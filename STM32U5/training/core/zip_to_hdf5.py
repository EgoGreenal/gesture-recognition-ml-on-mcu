"""Stream the Jester zip (toxicmender/20bn-jester) directly into three HDF5
files — bypasses extracting 2.4M tiny JPGs to Lustre (which would take 12+ h
due to Lustre metadata bottleneck).

Output:
  data/jester/train.h5        # ~118562 clips
  data/jester/validation.h5   # ~14787 clips
  data/jester/test.h5         # ~14743 clips

Each HDF5 contains:
  /clips/<video_id>          vlen uint8, shape [N_frames]   raw JPEG bytes per frame
    .attrs['label_id']       int8  (-1 for test which has no labels)
    .attrs['num_frames']     int16
  /video_ids                 int32  [N_clips]
  /label_ids                 int8   [N_clips]   (label_id per clip, -1 if missing)
  /num_frames                int16  [N_clips]
  /labels                    str    [27]        (only in train.h5; index = label_id)

Reading example (in dataloader):
  with h5py.File('train.h5') as f:
      ds = f[f'clips/{video_id}']        # one clip
      jpgs = [bytes(ds[i]) for i in range(ds.shape[0])]
      pil = [Image.open(BytesIO(j)) for j in jpgs]
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RAW_DIR = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/data/jester/_raw")
OUT_DIR = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/data/jester")
ZIP_PATH = RAW_DIR / "20bn-jester.zip"

# (csv basename inside zip, frame-folder prefix inside zip, output split name)
SPLITS = [
    ("Train.csv", "Train", "train"),
    ("Validation.csv", "Validation", "validation"),
    ("Test.csv", "Test", "test"),
]


def convert_split(zf: zipfile.ZipFile, csv_arc: str, arc_prefix: str, out_path: Path) -> None:
    print(f"\n=== {out_path.name} ===", flush=True)
    with zf.open(csv_arc) as f:
        df = pd.read_csv(f)
    # Normalize: Test.csv uses 'id', Train/Validation use 'video_id'
    if "video_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "video_id"})
    n_clips = len(df)
    has_label = "label_id" in df.columns and not df["label_id"].isna().all()
    print(f"  {n_clips} rows in {csv_arc}  (labels={'yes' if has_label else 'no'})", flush=True)

    vlen_dt = h5py.vlen_dtype(np.uint8)
    t0 = time.time()

    video_ids = np.empty(n_clips, dtype=np.int32)
    label_ids = np.full(n_clips, -1, dtype=np.int8)
    num_frames_arr = np.empty(n_clips, dtype=np.int16)

    with h5py.File(out_path, "w") as h5:
        clips_grp = h5.create_group("clips")
        # Disable HDF5 chunk-cache flushing chatter — vlen writes go through global heap

        for i, row in enumerate(df.itertuples(index=False)):
            video_id = int(row.video_id)
            expected_frames = int(row.frames)
            lab = int(row.label_id) if has_label else -1

            # Read N frames sequentially from zip
            buf = np.empty(expected_frames, dtype=object)
            actual = 0
            for fnum in range(1, expected_frames + 1):
                arcname = f"{arc_prefix}/{video_id}/{fnum:05d}.jpg"
                try:
                    b = zf.read(arcname)
                except KeyError:
                    print(f"  WARN: missing {arcname}", flush=True)
                    break
                buf[actual] = np.frombuffer(b, dtype=np.uint8)
                actual += 1
            buf = buf[:actual]

            ds = clips_grp.create_dataset(str(video_id), data=buf, dtype=vlen_dt)
            ds.attrs["label_id"] = lab
            ds.attrs["num_frames"] = actual

            video_ids[i] = video_id
            label_ids[i] = lab
            num_frames_arr[i] = actual

            if (i + 1) % 500 == 0 or i + 1 == n_clips:
                rate = (i + 1) / (time.time() - t0)
                eta_s = (n_clips - i - 1) / rate if rate > 0 else 0
                cur_mb = out_path.stat().st_size / 1e6
                print(
                    f"  [{i+1:>6}/{n_clips}]  rate={rate:6.1f} clips/s  "
                    f"eta={eta_s/60:5.1f} min  out={cur_mb:7.1f} MB",
                    flush=True,
                )

        h5.create_dataset("video_ids", data=video_ids)
        h5.create_dataset("label_ids", data=label_ids)
        h5.create_dataset("num_frames", data=num_frames_arr)

        if has_label:
            unique_pairs = (
                df.sort_values("label_id")
                .drop_duplicates("label_id")[["label_id", "label"]]
                .values
            )
            label_names = [name for _, name in unique_pairs]
            assert len(label_names) == 27, f"expected 27 classes, got {len(label_names)}"
            h5.create_dataset(
                "labels",
                data=label_names,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

    size_gb = out_path.stat().st_size / 1e9
    print(f"  WROTE {out_path}  {size_gb:.2f} GB  in {(time.time()-t0)/60:.1f} min", flush=True)


def main() -> int:
    if not ZIP_PATH.is_file():
        print(f"ERROR: zip not found: {ZIP_PATH}", flush=True)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {ZIP_PATH}  ({ZIP_PATH.stat().st_size/1e9:.2f} GB)", flush=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for csv_arc, arc_prefix, out_split in SPLITS:
            out_path = OUT_DIR / f"{out_split}.h5"
            convert_split(zf, csv_arc, arc_prefix, out_path)

    print("\n[zip_to_hdf5] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
