#!/usr/bin/env bash
# =============================================================================
# Experiment 2 — Frame-budget vs quality / compute trade-off sweep.
#
# Runs MA-LMM at multiple (num_frames, memory_bank_length) configurations to
# map the quality–compute Pareto frontier. All runs use the 10-min variant
# (fast per-video inference; 5 videos provide enough signal for mean ± std).
#
# Key insight being tested:
#   Increasing num_frames and memory_bank_length improves caption quality,
#   but inference time and GPU memory grow rapidly.  The sweep lets us show
#   that the quality delta beyond the baseline (100 frames, MBL=40) may not
#   justify the added compute.
#
# Configuration grid (mbl ≤ num_frames is always enforced):
#
#   num_frames  memory_bank_length   label
#   ----------  ------------------   -----
#      100             40            baseline (paper default ~80f/40mbl)
#      100             80            larger memory, same input frames
#      200             40            2× frames, baseline memory
#      200             80            2× both
#      200            160            2× frames, full memory (no compression)
#      400             80            4× frames, 2× memory
#      400            160            4× frames, 4× memory
#      400            320            4× frames, near-full memory
#
# Each configuration writes one JSON to outputs/exp2_frame_sweep/ containing
# per-video captions, inference_time_sec, and gpu_peak_memory_gb.
#
# Usage:
#   bash run_frame_sweep.sh                   # run all 8 configs
#   VARIANT=30min bash run_frame_sweep.sh     # sweep on 30-min videos instead
#
# After completion, generate plots:
#   python scripts/plot_sweep.py \
#       --sweep_dir outputs/exp2_frame_sweep \
#       --plot_dir  outputs/exp2_frame_sweep/plots
#
# With caption quality metrics (requires Ego4D narration.json):
#   python scripts/plot_sweep.py \
#       --sweep_dir  outputs/exp2_frame_sweep \
#       --plot_dir   outputs/exp2_frame_sweep/plots \
#       --narration  /path/to/ego4d/v2/annotations/narration.json \
#       --ann_path   data/annotations/ego4d_10min.json
#
# Prerequisites:
#   - conda activate malmm_baseline
#   - bash prepare_frames.sh  (frames must exist under data/frames/10min/)
#   - GPU with ≥ 40 GB VRAM recommended for the 400-frame configs
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VARIANT="${VARIANT:-10min}"
OUT_DIR="outputs/exp2_frame_sweep"
mkdir -p "${OUT_DIR}"

FRAME_DIR="data/frames/${VARIANT}"
ANN_PATH="data/annotations/ego4d_${VARIANT}.json"

if [ ! -d "${FRAME_DIR}" ]; then
    echo "ERROR: ${FRAME_DIR} not found. Run prepare_frames.sh first." >&2
    exit 1
fi

# Each entry: "num_frames memory_bank_length"
# Constraint: mbl <= num_frames (if mbl > num_frames, the memory bank never
# compresses — effectively the same as mbl = num_frames).
CONFIGS=(
    "100  40"
    "100  80"
    "200  40"
    "200  80"
    "200 160"
    "400  80"
    "400 160"
    "400 320"
)

total=${#CONFIGS[@]}
echo "============================================================"
echo "Experiment 2: Frame-budget sweep"
echo "  Variant  : ${VARIANT}"
echo "  Configs  : ${total}"
echo "  Output   : ${OUT_DIR}"
echo "============================================================"

idx=0
for cfg in "${CONFIGS[@]}"; do
    read -r NF MBL <<< "${cfg}"
    idx=$(( idx + 1 ))
    OUT_FILE="${OUT_DIR}/captions_f${NF}_m${MBL}.json"

    if [ -f "${OUT_FILE}" ]; then
        echo "[${idx}/${total}] SKIP  num_frames=${NF}  mbl=${MBL}  (already done)"
        continue
    fi

    echo ""
    echo "[${idx}/${total}] RUN  num_frames=${NF}  mbl=${MBL}"
    python scripts/infer.py \
        --frame_dir          "${FRAME_DIR}" \
        --ann_path           "${ANN_PATH}" \
        --output             "${OUT_FILE}" \
        --num_frames         "${NF}" \
        --memory_bank_length "${MBL}"

    echo "  → ${OUT_FILE}"
done

echo ""
echo "============================================================"
echo "Sweep complete (${total} configurations)."
echo ""
echo "Generate plots (compute only):"
echo "  python scripts/plot_sweep.py \\"
echo "      --sweep_dir ${OUT_DIR} \\"
echo "      --plot_dir  ${OUT_DIR}/plots"
echo ""
echo "Generate plots (+ caption quality):"
echo "  python scripts/plot_sweep.py \\"
echo "      --sweep_dir  ${OUT_DIR} \\"
echo "      --plot_dir   ${OUT_DIR}/plots \\"
echo "      --narration  /path/to/ego4d/v2/annotations/narration.json \\"
echo "      --ann_path   ${ANN_PATH}"
echo "============================================================"
