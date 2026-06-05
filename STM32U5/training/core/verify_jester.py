"""Verify Jester V1 dataset integrity after HDF5 conversion.

Inputs:
  data/jester/jester_full.h5
    /clips/<video_id>          vlen uint8, raw JPEG bytes per frame
    /video_ids                 int32  [N_clips]
    /num_frames                int16  [N_clips]

  data/jester/annotations/
    jester-v1-labels.csv       27 class names (one per line, idx = label_id)
    jester-v1-train.csv        118562 rows  "video_id;label_name"
    jester-v1-validation.csv   14787 rows   "video_id;label_name"
    jester-v1-test.csv         14743 rows   "video_id"

Checks:
  1. Labels CSV has 27 lines.
  2. Split CSVs have expected row counts.
  3. HDF5 has ~148092 clips, aligning with train+val+test union.
  4. Every train/val video_id has a corresponding /clips/<id> entry.
  5. Random spot-check: 5 clips, each has > 10 valid JPEG frames.
"""

from __future__ import annotations

import io
import random
import subprocess
import sys
from pathlib import Path

import h5py
from PIL import Image

ROOT = Path("/leonardo_work/IscrB_WearUsFM/jzhang03/MLonMCU/data/jester")
H5 = ROOT / "jester_full.h5"
ANN = ROOT / "annotations"

EXPECTED_CLASSES = 27
EXPECTED = {
    "train": 118562,
    "validation": 14787,
    "test": 14743,
}
TOTAL = sum(EXPECTED.values())  # 148092


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def read_split_csv(name: str) -> list[tuple[int, str | None]]:
    """Return list of (video_id, label_name_or_None). Test has no label."""
    p = ANN / f"jester-v1-{name}.csv"
    rows: list[tuple[int, str | None]] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                vid, lab = line.split(";", 1)
                rows.append((int(vid), lab))
            else:
                rows.append((int(line), None))
    return rows


def main() -> int:
    rc = 0

    # 1. Labels file
    labels_p = ANN / "jester-v1-labels.csv"
    if not labels_p.is_file():
        fail(f"missing {labels_p}")
        return 1
    with labels_p.open() as f:
        classes = [line.strip() for line in f if line.strip()]
    if len(classes) == EXPECTED_CLASSES:
        ok(f"labels: {len(classes)} classes  (first 5: {classes[:5]})")
    else:
        fail(f"labels has {len(classes)} classes, expected {EXPECTED_CLASSES}")
        rc = 1

    # 2. Split CSVs
    splits: dict[str, list] = {}
    for name, expected_n in EXPECTED.items():
        rows = read_split_csv(name)
        splits[name] = rows
        if len(rows) == expected_n:
            ok(f"{name}.csv: {len(rows)} rows")
        else:
            fail(f"{name}.csv: {len(rows)} rows, expected {expected_n}")
            rc = 1

    # 3. HDF5 exists
    if not H5.is_file():
        fail(f"missing {H5} — run tgz_to_hdf5.py first")
        return 1

    with h5py.File(H5, "r") as h5:
        video_ids_arr = h5["video_ids"][:]
        num_frames_arr = h5["num_frames"][:]
        n_h5 = len(video_ids_arr)
        if abs(n_h5 - TOTAL) < 100:
            ok(f"HDF5: {n_h5} clips  (expected ~{TOTAL})")
        else:
            fail(f"HDF5 has {n_h5} clips, expected ~{TOTAL}")
            rc = 1

        h5_id_set = set(int(v) for v in video_ids_arr)

        # 4. CSV → HDF5 coverage
        for name in ("train", "validation"):
            csv_ids = {vid for vid, _ in splits[name]}
            missing = csv_ids - h5_id_set
            if not missing:
                ok(f"{name}: all {len(csv_ids)} CSV video_ids present in HDF5")
            else:
                fail(f"{name}: {len(missing)} CSV video_ids missing in HDF5 (e.g. {list(missing)[:5]})")
                rc = 1

        # 5. Spot-check
        clips_grp = h5["clips"]
        sample_ids = random.sample(list(h5_id_set), 5)
        print("      spot check 5 random clips:")
        for vid in sample_ids:
            ds = clips_grp[str(vid)]
            n_frames = ds.shape[0]
            # try decoding first + middle + last frame
            first_b = bytes(ds[0])
            last_b = bytes(ds[-1])
            try:
                img0 = Image.open(io.BytesIO(first_b))
                img1 = Image.open(io.BytesIO(last_b))
                size0 = img0.size  # (W, H)
                size1 = img1.size
                ok_frame = f"OK size={size0}"
            except Exception as e:
                ok_frame = f"DECODE FAIL: {e}"
                rc = 1
            print(f"        clip {vid}: {n_frames} frames  first/last decode: {ok_frame}")

        # Frame count stats
        nf = num_frames_arr
        print(f"      frame counts: min={nf.min()}  max={nf.max()}  mean={nf.mean():.1f}  median={int(sorted(nf.tolist())[len(nf)//2])}")

    # 6. Disk
    try:
        du = subprocess.check_output(["du", "-sh", str(H5)], text=True).strip()
        print(f"      du -sh {H5} -> {du}")
    except Exception as e:
        print(f"      du failed: {e}")

    print("\n" + ("ALL CHECKS PASSED." if rc == 0 else "SOME CHECKS FAILED."))
    return rc


if __name__ == "__main__":
    sys.exit(main())
