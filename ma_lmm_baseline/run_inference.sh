#!/usr/bin/env bash
# Run MA-LMM zero-shot inference on each duration variant.
set -euo pipefail
cd "$(dirname "$0")"

# Memory-bank size and sampled-frame count: defaults match the paper's
# captioning configs (memory_bank_length=40, num_frames=80).
MBL="${MBL:-40}"
NFRAMES="${NFRAMES:-80}"
CKPT="${CKPT:-}"   # set to a fine-tuned MA-LMM .pth to bypass zero-shot

CKPT_ARG=()
if [ -n "${CKPT}" ]; then
    CKPT_ARG=(--ckpt_path "${CKPT}")
fi

mkdir -p outputs

for V in 10min 30min 60min; do
    echo "=== Running on ${V} ==="
    python scripts/infer.py \
        --frame_dir data/frames/"${V}" \
        --ann_path  data/annotations/ego4d_"${V}".json \
        --output    outputs/captions_"${V}".json \
        --memory_bank_length "${MBL}" \
        --num_frames "${NFRAMES}" \
        "${CKPT_ARG[@]}"
done

echo "All variants done. Captions live under outputs/."
