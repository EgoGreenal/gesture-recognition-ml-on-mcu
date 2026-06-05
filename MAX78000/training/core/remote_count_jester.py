from pathlib import Path

base = Path("/path/to/20bn-jester-v1-complete")
frames = base / "20bn-jester-v1"
ann = Path("/path/to/20bn-jester/annotations")

frame_dirs = {p.name for p in frames.iterdir() if p.is_dir()}
print("frame_dirs", len(frame_dirs))

for name in [
    "jester-v1-labels.csv",
    "jester-v1-train.csv",
    "jester-v1-validation.csv",
    "jester-v1-test.csv",
]:
    path = ann / name
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if name.endswith("labels.csv"):
        print(name, "labels", len(lines))
        continue

    ids = []
    labeled = 0
    for line in lines:
        if ";" in line:
            vid = line.split(";", 1)[0].strip()
            labeled += 1
        elif "," in line:
            vid = line.split(",", 1)[0].strip()
            labeled += 1
        else:
            vid = line.strip()
        ids.append(vid)

    existing = sum(1 for vid in ids if vid in frame_dirs)
    print(
        name,
        "rows", len(lines),
        "parsed", len(ids),
        "labeled_rows", labeled,
        "existing_frame_dirs", existing,
        "missing_frame_dirs", len(ids) - existing,
    )
