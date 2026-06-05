"""Stream Jester V1 complete (tgz nested inside Kaggle zip) → single HDF5.

Pipeline: zip → inner tgz stream → tarfile streaming → per-clip flush → HDF5.

Bypasses extracting ~5M tiny JPGs to Lustre (would take 12+ h due to metadata
bottleneck). Frames are decoded on-demand at training time from in-memory bytes.

Source: sanjanatg26/20bn-jester-v1-complete (22.9 GB zip, contains one 22 GB tgz
inside, which contains ~148,092 clip directories, each with N frames as JPG).

Output:
  data/jester/jester_full.h5

Schema:
  /clips/<video_id>          vlen uint8, shape [N_frames]   raw JPEG bytes
    .attrs['num_frames']     int
  /video_ids                 int32  [N_clips]               all video IDs
  /num_frames                int16  [N_clips]               aligned with video_ids

Split CSVs (jester-v1-train.csv etc.) are sourced separately — Jester's archive
ships frames only.
"""

from __future__ import annotations

import sys
import tarfile
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np

ZIP_PATH = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/data/jester/_raw_full/20bn-jester-v1-complete.zip")
INNER_TGZ_NAME = "20bn-jester-v1-complete.tgz"
OUT_DIR = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/data/jester")
OUT_H5 = OUT_DIR / "jester_full.h5"

EXPECTED_CLIPS = 148_092


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.is_file():
        print(f"ERROR: zip not found: {ZIP_PATH}", flush=True)
        return 1

    print(f"Source: {ZIP_PATH} ({ZIP_PATH.stat().st_size/1e9:.2f} GB)", flush=True)
    print(f"Output: {OUT_H5}", flush=True)

    vlen_dt = h5py.vlen_dtype(np.uint8)

    t0 = time.time()
    video_ids: list[int] = []
    num_frames_list: list[int] = []
    flushed = 0

    # Buffer for current clip
    cur_clip: str | None = None
    cur_frames: dict[int, bytes] = {}

    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open(INNER_TGZ_NAME) as stream:
            tf = tarfile.open(fileobj=stream, mode="r|gz")
            h5 = h5py.File(OUT_H5, "w")
            clips_grp = h5.create_group("clips")

            def flush() -> None:
                nonlocal flushed
                if cur_clip is None or not cur_frames:
                    return
                sorted_idx = sorted(cur_frames.keys())
                buf = np.empty(len(sorted_idx), dtype=object)
                for i, fi in enumerate(sorted_idx):
                    buf[i] = np.frombuffer(cur_frames[fi], dtype=np.uint8)
                ds = clips_grp.create_dataset(cur_clip, data=buf, dtype=vlen_dt)
                ds.attrs["num_frames"] = len(sorted_idx)
                video_ids.append(int(cur_clip))
                num_frames_list.append(len(sorted_idx))
                flushed += 1
                if flushed % 500 == 0 or flushed == EXPECTED_CLIPS:
                    elapsed = time.time() - t0
                    rate = flushed / elapsed
                    eta_min = (EXPECTED_CLIPS - flushed) / rate / 60 if rate > 0 else 0
                    out_gb = OUT_H5.stat().st_size / 1e9
                    print(
                        f"  [{flushed:>6}/{EXPECTED_CLIPS}]  rate={rate:5.1f} clips/s  "
                        f"eta={eta_min:5.1f} min  out={out_gb:5.2f} GB",
                        flush=True,
                    )

            try:
                for member in tf:
                    if not member.isfile() or not member.name.endswith(".jpg"):
                        continue
                    parts = member.name.split("/")
                    if len(parts) < 3:
                        continue
                    video_id = parts[-2]
                    try:
                        frame_idx = int(parts[-1].split(".")[0])
                    except ValueError:
                        continue

                    if video_id != cur_clip:
                        flush()
                        cur_clip = video_id
                        cur_frames = {}

                    f = tf.extractfile(member)
                    if f is not None:
                        cur_frames[frame_idx] = f.read()

                flush()  # final clip

                h5.create_dataset("video_ids", data=np.array(video_ids, dtype=np.int32))
                h5.create_dataset("num_frames", data=np.array(num_frames_list, dtype=np.int16))

            finally:
                h5.close()
                tf.close()

    print(
        f"\nDONE: {flushed} clips in {(time.time()-t0)/60:.1f} min  "
        f"file={OUT_H5.stat().st_size/1e9:.2f} GB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
