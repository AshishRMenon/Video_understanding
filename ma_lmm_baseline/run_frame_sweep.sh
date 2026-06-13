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
#   bash run_frame_sweep.sh                              # run all 8 configs + compute plots
#   VARIANT=30min bash run_frame_sweep.sh                # sweep on 30-min videos instead
#   OUT_PARENT=outputs/2026-06-13 bash run_frame_sweep.sh  # custom output parent folder
#
#   # Full pipeline: inference + ROUGE-L + QA evaluation + all plots:
#   NARRATION_PATH=/workspace/ego4d_meta/v2/annotations/narration.json \
#   ANTHROPIC_API_KEY=sk-ant-... \
#   bash run_frame_sweep.sh
#
#   # Override QA model (default: claude-haiku-4-5-20251001):
#   QA_MODEL=claude-sonnet-4-6 NARRATION_PATH=... ANTHROPIC_API_KEY=... bash run_frame_sweep.sh
#
# Plots written to outputs/exp2_frame_sweep/plots/:
#   inference_time.png      always
#   gpu_memory.png          always
#   caption_quality.png     requires NARRATION_PATH
#   quality_vs_compute.png  requires NARRATION_PATH
#   qa_score.png            requires NARRATION_PATH + ANTHROPIC_API_KEY
#
# Prerequisites:
#   - conda activate malmm_baseline
#   - bash prepare_frames.sh  (frames must exist under data/frames/10min/)
#   - GPU with ≥ 40 GB VRAM recommended for the 400-frame configs
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VARIANT="${VARIANT:-10min}"
OUT_PARENT="${OUT_PARENT:-outputs}"
OUT_DIR="${OUT_PARENT}/exp2_frame_sweep"
PLOT_DIR="${OUT_DIR}/plots"
# Optional: set NARRATION_PATH to enable ROUGE-L + QA quality plots.
#   NARRATION_PATH=/workspace/ego4d_meta/v2/annotations/narration.json bash run_frame_sweep.sh
NARRATION_PATH="${NARRATION_PATH:-}"
# Optional: override LLM model used for QA evaluation (default: claude-haiku-4-5-20251001).
#   QA_MODEL=claude-sonnet-4-6 bash run_frame_sweep.sh
QA_MODEL="${QA_MODEL:-claude-haiku-4-5-20251001}"
# Optional: number of questions per video for QA eval (default: 12).
QA_N_QUESTIONS="${QA_N_QUESTIONS:-12}"
mkdir -p "${OUT_DIR}"

FRAME_DIR="data/frames/${VARIANT}"
ANN_PATH="data/annotations/ego4d_${VARIANT}.json"

if [ ! -d "${FRAME_DIR}" ]; then
    echo "ERROR: ${FRAME_DIR} not found. Run prepare_frames.sh first." >&2
    exit 1
fi

# Each entry: "num_frames memory_bank_length"
# Configs form a linear scale-up: both num_frames and memory grow together.
CONFIGS=(
    " 100  40"
    " 200  80"
    " 500 150"
    "1000 300"
)

total=${#CONFIGS[@]}
echo "============================================================"
echo "Experiment 2: Frame-budget sweep"
echo "  Variant      : ${VARIANT}"
echo "  Configs      : ${total}"
echo "  Output       : ${OUT_DIR}"
echo "  Narration GT : ${NARRATION_PATH:-"(not set — quality + QA plots skipped)"}"
if [ -n "${NARRATION_PATH}" ]; then
echo "  QA model     : ${QA_MODEL}  (questions: ${QA_N_QUESTIONS})"
fi
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
echo "============================================================"

# ---- Step 2: QA evaluation (only when narration + API key are available) --
if [ -n "${NARRATION_PATH}" ]; then
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo ""
        echo "WARNING: ANTHROPIC_API_KEY not set — skipping QA evaluation."
        echo "  Export the key and re-run to add QA scores:"
        echo "  ANTHROPIC_API_KEY=sk-ant-... NARRATION_PATH=${NARRATION_PATH} bash run_frame_sweep.sh"
    else
        echo ""
        echo "Running QA evaluation (model: ${QA_MODEL}, questions: ${QA_N_QUESTIONS})…"
        python scripts/qa_eval.py \
            --sweep_dir   "${OUT_DIR}" \
            --narration   "${NARRATION_PATH}" \
            --ann_path    "${ANN_PATH}" \
            --model       "${QA_MODEL}" \
            --n_questions "${QA_N_QUESTIONS}"
    fi
fi

# ---- Step 3: Plotting -----------------------------------------------------
echo ""
echo "Generating plots…"

PLOT_CMD=(
    python scripts/plot_sweep.py
    --sweep_dir "${OUT_DIR}"
    --plot_dir  "${PLOT_DIR}"
)

if [ -n "${NARRATION_PATH}" ]; then
    PLOT_CMD+=(--narration "${NARRATION_PATH}" --ann_path "${ANN_PATH}")
fi

"${PLOT_CMD[@]}"

echo ""
echo "Plots written to: ${PLOT_DIR}"
echo "============================================================"
