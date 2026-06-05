#!/usr/bin/env bash
set -u

BASE="/path/to/20bn-jester-v1-complete-parent"
DATASET="$BASE/versions/1/20bn-jester-v1-complete"
FRAMES="$DATASET/20bn-jester-v1"
ARCHIVE="$BASE/versions/1/20bn-jester-v1-complete.tgz"

echo "BASE=$BASE"
echo "DATASET=$DATASET"
echo "ARCHIVE=$ARCHIVE"

echo "all small metadata files under BASE:"
find "$BASE" -type f \( -name "*.csv" -o -name "*.json" -o -name "*.txt" \) -printf "%p %s bytes\n" | sort | head -200

echo "archive label-like members:"
if [[ -f "$ARCHIVE" ]]; then
  tar -tzf "$ARCHIVE" | grep -Ei "csv|json|label|train|valid|test|jester" | head -200
else
  echo "archive missing"
fi

echo "frame dir count:"
find "$FRAMES" -maxdepth 1 -mindepth 1 -type d | wc -l

echo "sample directories and frame counts:"
find "$FRAMES" -maxdepth 1 -mindepth 1 -type d | head -8 | while read -r d; do
  echo "$(basename "$d") $(find "$d" -maxdepth 1 -type f | wc -l)"
  find "$d" -maxdepth 1 -type f | sort | head -5
done
