#!/usr/bin/env bash
# Extract frames at 10 FPS (official preprocessing recipe) and build annotations.
set -euo pipefail
cd "$(dirname "$0")"

python scripts/extract_frames.py \
    --video_root data/videos \
    --frame_root data/frames \
    --fps 10 \
    --variants 10min 30min 60min \
    --workers 4

python scripts/make_annotation.py \
    --frame_root data/frames \
    --ann_root data/annotations \
    --variants 10min 30min 60min
