"""TensorFlow tf.data pipeline — Pipeline L (grayscale, parameterized H/W/T) for Student.

Day 10 rewrite: JPEG decode + crop + resize move from PIL (Python, GIL-bound)
to ``tf.io.decode_jpeg`` + ``tf.image`` ops in the tf.data graph (C++,
GIL-free). Day 9 measured ~9 samples/sec aggregate on the PIL version
(throughput plateaued regardless of num_shards because GIL serialized the
Python work); the tf.io version targets 5-10x throughput so 30-epoch full
training fits a sensible SLURM wall.

Pipeline:
    1. Python generator (one per shard):
         * reads HDF5 vlen JPEG bytes -- h5py releases GIL during chunked read
         * samples crop params + flip flag (numpy random)
         * yields (jpegs[T] string, label, do_flip, top, left, side)
    2. tf.data.map(num_parallel_calls=AUTOTUNE):
         * tf.io.decode_jpeg per frame (C++)
         * tf.image.crop_to_bounding_box + tf.image.resize
         * normalize to [-1, 1]
         * conditional horizontal flip
    3. Shuffle + batch + prefetch.

This module also exposes ``build_kd_dataset`` -- a wrapper that adds
TA soft labels with flip-aware class-axis remap. (The old ``kd_dataset_tf.py``
is now a thin re-export of ``build_kd_dataset`` from here, for back-compat.)
"""

from __future__ import annotations

import os
from pathlib import Path

# Disable HDF5 file locking BEFORE importing h5py. On Lustre + multi-reader
# (e.g. 8 generator shards in this process + concurrent SLURM jobs), the
# default flock-based locking serializes opens and starves readers. The
# Jester HDF5 is read-only, so locking buys us nothing.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import tensorflow as tf

from jester_common import (
    FLIP_LABEL_MAP,
    H5_PATH,
    parse_split,
    sample_segment_indices,
)

AUTOTUNE = tf.data.AUTOTUNE


# ---- Python-side: read raw JPEG bytes + crop/flip params -------------------

def _gen_items(h5: h5py.File, items, T: int, training: bool, return_soft_idx: bool = False):
    """Pure-Python generator: yield raw JPEG bytes + augmentation params per clip.

    `items`: list of (vid, label) or (vid, label, soft_idx) tuples.
    No PIL, no decode -- those move to the tf.data graph.

    Day 10 perf measurement (jester_full.h5 on Lustre):
        - per-frame ds[i] vlen read:        12 clips/s  (0.5 MB/s)
        - batch ds[offs.tolist()] (8 idx):  24 clips/s  (1.0 MB/s)
        - full ds[:] read (all frames):     81 clips/s  (14.6 MB/s)
    HDF5 chunked vlen has per-chunk metadata overhead; reading the WHOLE clip
    once and slicing the offsets we need in numpy is 6.8x faster than reading
    8 frames individually. Jester clips average ~36 frames, so we trade
    ~4.5x more bytes-read for the speedup -- still a net win.
    """
    flip_map_arr = np.asarray(FLIP_LABEL_MAP, dtype=np.int32)
    for it in items:
        if return_soft_idx:
            vid, label, soft_idx = it
        else:
            vid, label = it
            soft_idx = -1
        ds = h5["clips"][str(vid)]
        n_frames = ds.shape[0]
        offsets = sample_segment_indices(n_frames, T, training)

        # ONE h5py call reads all frames; slice the T we need in numpy
        all_bytes = ds[:]                                     # (n_frames,) vlen
        jpegs = np.array(
            [bytes(all_bytes[int(o)]) for o in offsets.tolist()],
            dtype=object,
        )

        # Crop params: peek at first frame's spatial dims by decoding *header*
        # only would be ideal, but JPEG header parsing in pure Python is its
        # own can of worms. Jester clips are uniformly 100x176; we pass the
        # crop window in normalized coords and let TF figure exact dims after
        # decode. Use generator-side numpy random for reproducible per-clip
        # crop selection.
        if training:
            crop_frac = np.random.uniform(0.6, 1.0)
            aspect = np.random.uniform(0.9, 1.1)
            # Encode as ratios so graph can absorb any source H,W:
            #   target_h_frac = sqrt(crop_frac / aspect)
            #   target_w_frac = sqrt(crop_frac * aspect)
            target_h_frac = float(np.sqrt(crop_frac / aspect))
            target_w_frac = float(np.sqrt(crop_frac * aspect))
            # Clamp to [0.1, 1.0] just in case
            target_h_frac = max(0.1, min(1.0, target_h_frac))
            target_w_frac = max(0.1, min(1.0, target_w_frac))
            top_frac = float(np.random.uniform(0.0, 1.0 - target_h_frac))
            left_frac = float(np.random.uniform(0.0, 1.0 - target_w_frac))
        else:
            # Center square crop
            # Equivalent to top_frac = (1 - side_frac) / 2, side_frac = min(H/H, W/H) = min(1, W/H)
            # but we don't know H, W here -- the graph will compute it.
            target_h_frac = -1.0   # sentinel: graph uses center-crop branch
            target_w_frac = -1.0
            top_frac = 0.0
            left_frac = 0.0

        do_flip = bool(training and (np.random.rand() < 0.5))
        if do_flip:
            label = int(flip_map_arr[label])

        yield (
            jpegs,                                                 # (T,) object/string
            np.int32(label),
            np.bool_(do_flip),
            np.float32(top_frac),
            np.float32(left_frac),
            np.float32(target_h_frac),
            np.float32(target_w_frac),
            np.int64(soft_idx),
        )


def _make_shard_gen(items, h5_path: str, T: int, training: bool, return_soft_idx: bool):
    def _gen():
        with h5py.File(h5_path, "r", libver="latest", swmr=True) as h5:
            yield from _gen_items(h5, items, T, training, return_soft_idx)
    return _gen


# ---- Graph-side: decode + crop + resize + normalize -----------------------

def _make_decode_fn(T: int, img_size: int):
    """Return a tf.function that decodes + augments one clip.

    Inputs (all per-clip):
        jpegs:       (T,) string
        label:       () int32
        do_flip:     () bool
        top_frac:    () float32 in [0, 1]
        left_frac:   () float32 in [0, 1]
        target_h_frac: () float32 in (0, 1] -- or -1 to flag center crop
        target_w_frac: () float32 in (0, 1] -- or -1 to flag center crop

    Output: (frames[T,H,W,1] float32, label, do_flip)
    """
    img_size_t = tf.constant(img_size, dtype=tf.int32)

    @tf.function
    def _decode(jpegs, label, do_flip, top_frac, left_frac, th_frac, tw_frac):
        # Decode first frame to learn H, W (assumed uniform across clip)
        first = tf.io.decode_jpeg(jpegs[0], channels=1)        # (H, W, 1) uint8
        H = tf.shape(first)[0]
        W = tf.shape(first)[1]

        # Compute crop box (top, left, h, w) in pixel coords
        center_crop_flag = th_frac < 0.0  # training=False sentinel
        side = tf.minimum(H, W)
        # Center-crop branch
        cc_top = (H - side) // 2
        cc_left = (W - side) // 2
        cc_h = side
        cc_w = side
        # Random-crop branch
        rc_h = tf.cast(tf.cast(H, tf.float32) * th_frac, tf.int32)
        rc_w = tf.cast(tf.cast(W, tf.float32) * tw_frac, tf.int32)
        rc_top = tf.cast(tf.cast(H, tf.float32) * top_frac, tf.int32)
        rc_left = tf.cast(tf.cast(W, tf.float32) * left_frac, tf.int32)
        # Clamp random-crop to valid range
        rc_top = tf.minimum(rc_top, H - rc_h)
        rc_left = tf.minimum(rc_left, W - rc_w)
        rc_top = tf.maximum(rc_top, 0)
        rc_left = tf.maximum(rc_left, 0)
        rc_h = tf.maximum(rc_h, 1)
        rc_w = tf.maximum(rc_w, 1)

        top_px = tf.where(center_crop_flag, cc_top, rc_top)
        left_px = tf.where(center_crop_flag, cc_left, rc_left)
        h_px = tf.where(center_crop_flag, cc_h, rc_h)
        w_px = tf.where(center_crop_flag, cc_w, rc_w)

        def _process_one(jpg):
            img = tf.io.decode_jpeg(jpg, channels=1)                # (H, W, 1) uint8
            img = tf.image.crop_to_bounding_box(img, top_px, left_px, h_px, w_px)
            img = tf.image.resize(img, [img_size_t, img_size_t], method="bilinear")
            img = tf.cast(img, tf.float32) / 127.5 - 1.0           # [-1, 1]
            return img                                              # (img_size, img_size, 1)

        # Unroll T frames (T is python int, baked into graph)
        frames_list = [_process_one(jpegs[t]) for t in range(T)]
        frames = tf.stack(frames_list, axis=0)                      # (T, img_size, img_size, 1)

        # Conditional horizontal flip along W axis
        frames = tf.cond(
            do_flip,
            lambda: tf.reverse(frames, axis=[-2]),
            lambda: frames,
        )
        return frames, label, do_flip

    return _decode


# ---- Public dataset builder -----------------------------------------------

def build_tf_dataset(
    split: str,
    batch_size: int,
    T: int = 8,
    img_size: int = 64,
    training: bool = True,
    num_shards: int = 8,
    shuffle_buf: int = 128,
    h5_path: Path | str = H5_PATH,
    seed: int | None = None,
    mode: str = "sequence",
) -> tf.data.Dataset:
    """tf.data.Dataset yielding (clip, label) batches.

    Args:
        mode: 'sequence' (default) yields clip shape (B, T, H, W, 1) for Path
              A/C1/C3. 'stacked' yields (B, H, W, T) for Path C2 (T frames as
              channel axis — pure 2D CNN input).
    """
    items = parse_split(split)
    if training:
        rng = np.random.default_rng(seed)
        order = np.arange(len(items)); rng.shuffle(order)
        items = [items[i] for i in order]
    shards = [items[i::num_shards] for i in range(num_shards)]

    # output_signature of the GENERATOR (before .map)
    output_signature = (
        tf.TensorSpec(shape=(T,), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.float32),   # top_frac
        tf.TensorSpec(shape=(), dtype=tf.float32),   # left_frac
        tf.TensorSpec(shape=(), dtype=tf.float32),   # target_h_frac
        tf.TensorSpec(shape=(), dtype=tf.float32),   # target_w_frac
        tf.TensorSpec(shape=(), dtype=tf.int64),     # soft_idx (unused; -1 sentinel)
    )

    def _shard_ds(i: int) -> tf.data.Dataset:
        gen = _make_shard_gen(shards[i], str(h5_path), T, training, return_soft_idx=False)
        return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    ds_shards = [_shard_ds(i) for i in range(num_shards)]
    ds = tf.data.Dataset.sample_from_datasets(
        ds_shards, weights=[1.0 / num_shards] * num_shards, stop_on_empty_dataset=False,
    )

    decode_fn = _make_decode_fn(T, img_size)
    stacked_mode = (mode == "stacked")
    if mode not in ("sequence", "stacked"):
        raise ValueError(f"mode must be 'sequence' or 'stacked', got {mode!r}")

    def _map_drop_soft_idx(jpegs, label, do_flip, tf_, lf, thf, twf, soft_idx):
        frames, label_out, do_flip_out = decode_fn(jpegs, label, do_flip, tf_, lf, thf, twf)
        if stacked_mode:
            # (T, H, W, 1) -> (H, W, T): stack T frames as channels for C2
            frames = tf.squeeze(frames, axis=-1)           # (T, H, W)
            frames = tf.transpose(frames, perm=[1, 2, 0])  # (H, W, T)
        return frames, label_out

    ds = ds.map(_map_drop_soft_idx, num_parallel_calls=AUTOTUNE, deterministic=False)
    if training:
        ds = ds.shuffle(shuffle_buf, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_kd_dataset(
    split: str,
    batch_size: int,
    soft_labels_path: Path | str,
    T: int = 8,
    img_size: int = 64,
    training: bool = True,
    num_shards: int = 8,
    shuffle_buf: int = 128,
    h5_path: Path | str = H5_PATH,
    seed: int | None = None,
    mode: str = "sequence",
    ta_reduce: str | None = None,
) -> tf.data.Dataset:
    """tf.data.Dataset yielding (clip, label, soft_logits) batches.

    Args:
        mode: 'sequence' (Path A/C1/C3): clip (B,T,H,W,1), soft_logits (B,T_ta,K)
              'stacked' (Path C2): clip (B,H,W,T), soft_logits (B,K) after
              ta_reduce.
        ta_reduce: only used when mode='stacked' (video-level KD). 'last'
              (default for stacked) takes soft[:, -1, :]; 'mean' averages
              over T_ta. Per Day 9 sanity, TA frame[7] = 95.5% argmax match,
              'last' is the right default.

    Soft labels are class-axis-permuted via FLIP_LABEL_MAP when the clip is
    horizontally flipped (Day 6's hidden footgun).
    """
    items_base = parse_split(split)
    soft = np.load(str(soft_labels_path))  # (N, T_ta, K) fp16
    if soft.shape[0] != len(items_base):
        raise ValueError(
            f"soft labels shape {soft.shape} does not match split size {len(items_base)}"
        )
    T_ta = int(soft.shape[1])
    K = int(soft.shape[2])
    soft_fp32 = soft.astype(np.float32)  # one-time cast (51 MB → ~100 MB)

    items = [(vid, label, idx) for idx, (vid, label) in enumerate(items_base)]
    if training:
        rng = np.random.default_rng(seed)
        order = np.arange(len(items)); rng.shuffle(order)
        items = [items[i] for i in order]
    shards = [items[i::num_shards] for i in range(num_shards)]

    output_signature = (
        tf.TensorSpec(shape=(T,), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )

    def _shard_ds(i: int) -> tf.data.Dataset:
        gen = _make_shard_gen(shards[i], str(h5_path), T, training, return_soft_idx=True)
        return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    ds_shards = [_shard_ds(i) for i in range(num_shards)]
    ds = tf.data.Dataset.sample_from_datasets(
        ds_shards, weights=[1.0 / num_shards] * num_shards, stop_on_empty_dataset=False,
    )

    decode_fn = _make_decode_fn(T, img_size)
    soft_const = tf.constant(soft_fp32, dtype=tf.float32)            # (N, T_ta, K)
    flip_map_t = tf.constant(FLIP_LABEL_MAP, dtype=tf.int32)

    stacked_mode = (mode == "stacked")
    if mode not in ("sequence", "stacked"):
        raise ValueError(f"mode must be 'sequence' or 'stacked', got {mode!r}")
    if stacked_mode and ta_reduce is None:
        ta_reduce = "last"

    def _map(jpegs, label, do_flip, tf_, lf, thf, twf, soft_idx):
        frames, label, do_flip = decode_fn(jpegs, label, do_flip, tf_, lf, thf, twf)
        if stacked_mode:
            frames = tf.squeeze(frames, axis=-1)             # (T, H, W)
            frames = tf.transpose(frames, perm=[1, 2, 0])    # (H, W, T)
        soft_logits = tf.gather(soft_const, soft_idx, axis=0)        # (T_ta, K)
        soft_logits = tf.cond(
            do_flip,
            lambda: tf.gather(soft_logits, flip_map_t, axis=-1),
            lambda: soft_logits,
        )
        if stacked_mode:
            if ta_reduce == "last":
                soft_logits = soft_logits[-1, :]              # (K,)
            elif ta_reduce == "mean":
                soft_logits = tf.reduce_mean(soft_logits, axis=0)
            else:
                raise ValueError(f"unknown ta_reduce: {ta_reduce}")
        return frames, label, soft_logits

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE, deterministic=False)
    if training:
        ds = ds.shuffle(shuffle_buf, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---- Smoke test -----------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--kd", action="store_true")
    args = ap.parse_args()

    tf.get_logger().setLevel("ERROR")

    if args.kd:
        ds = build_kd_dataset(
            split=args.split, batch_size=args.batch_size,
            soft_labels_path="soft_labels/soft_labels_ta_train.npy",
            T=args.T, img_size=args.img_size, training=(args.split == "train"),
            num_shards=args.num_shards, seed=42,
        )
    else:
        ds = build_tf_dataset(
            split=args.split, batch_size=args.batch_size,
            T=args.T, img_size=args.img_size, training=(args.split == "train"),
            num_shards=args.num_shards, seed=42,
        )

    # Warmup: pull 1 batch (compiles graph + fills first samples)
    t0 = time.time()
    for batch in ds.take(1):
        pass
    warmup_t = time.time() - t0
    print(f"warmup (graph compile + first batch): {warmup_t:.2f}s")

    # Throughput measure
    n = 0; n_samples = 0
    t0 = time.time()
    for batch in ds.take(args.n_batches):
        n += 1
        n_samples += int(batch[0].shape[0])
    elapsed = time.time() - t0
    rate = n_samples / elapsed
    print(f"{n} batches ({n_samples} samples) in {elapsed:.2f}s -> {rate:.1f} samples/s")
    print(f"first batch shapes: clip={tuple(batch[0].shape)}  label={tuple(batch[1].shape)} "
          f"{('soft='+str(tuple(batch[2].shape))) if args.kd else ''}")
