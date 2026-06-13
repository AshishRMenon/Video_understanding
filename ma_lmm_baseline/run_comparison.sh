#!/usr/bin/env bash
# =============================================================================
# Experiment 3 — Chunked captioning vs. full-video captioning (QA comparison).
#
# Compares two approaches on 10-min Ego4D videos:
#
#   A) CHUNKED   — Split each video into temporal chunks, caption each chunk
#                  with MA-LMM, merge with Vicuna into one video-level summary.
#
#   B) FULL      — Caption the entire video at once with MA-LMM using the best
#                  (num_frames, memory_bank_length) config from the frame sweep.
#
# Quality is measured with our QA-based coverage metric (qa_eval.py):
#   generate questions from the ground-truth narration → check how many the
#   predicted caption can answer → QA score = fraction answered.
#
# Usage:
#   NARRATION_PATH=/workspace/ego4d_meta/v2/annotations/narration.json \
#   ANTHROPIC_API_KEY=sk-ant-... \
#   bash run_comparison.sh
#
# Key env-var overrides (all optional):
#   NARRATION_PATH    Ego4D narration.json                  (required for QA)
#   ANTHROPIC_API_KEY Anthropic API key                     (required for QA)
#   BEST_NF           num_frames for full-video run         (default: 200)
#   BEST_MBL          memory_bank_length for full-video run (default: 160)
#   CHUNK_MINS        chunk duration in minutes             (default: 2)
#   VIDEO_FPS         FPS used during frame extraction      (default: 10)
#   QA_MODEL          LLM for QA evaluation                 (default: claude-haiku-4-5-20251001)
#   QA_N_QUESTIONS    questions generated per video         (default: 12)
#   VARIANT           frame/annotation variant              (default: 10min)
#   OUT_PARENT        parent output dir                     (default: outputs)
#
# Outputs written to outputs/exp3_comparison/:
#   chunked/                    per-video JSONs from infer_chunked.py
#   full_captions.json          per-video captions from infer.py
#   chunked_captions.json       chunked summaries in flat [{video_id, caption}] format
#   qa_cache.json               shared question cache (generated once, reused for both)
#   chunked_qa.json             QA scores for chunked approach
#   full_qa.json                QA scores for full-video approach
#   plots/qa_comparison.png     bar chart comparing the two approaches
#
# Prerequisites:
#   - conda activate malmm_baseline
#   - bash prepare_frames.sh  (frames must exist under data/frames/10min/)
#   - bash run_frame_sweep.sh  (sets baseline for BEST_NF / BEST_MBL choice)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- Configuration -----------------------------------------------------------
VARIANT="${VARIANT:-10min}"
OUT_PARENT="${OUT_PARENT:-outputs}"
OUT_DIR="${OUT_PARENT}/exp3_comparison"
PLOT_DIR="${OUT_DIR}/plots"

BEST_NF="${BEST_NF:-200}"
BEST_MBL="${BEST_MBL:-160}"
CHUNK_MINS="${CHUNK_MINS:-2}"
VIDEO_FPS="${VIDEO_FPS:-10}"

NARRATION_PATH="${NARRATION_PATH:-}"
QA_MODEL="${QA_MODEL:-claude-haiku-4-5-20251001}"
QA_N_QUESTIONS="${QA_N_QUESTIONS:-12}"

FRAME_DIR="data/frames/${VARIANT}"
ANN_PATH="data/annotations/ego4d_${VARIANT}.json"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/chunked" "${PLOT_DIR}"

if [ ! -d "${FRAME_DIR}" ]; then
    echo "ERROR: ${FRAME_DIR} not found. Run prepare_frames.sh first." >&2
    exit 1
fi

echo "============================================================"
echo "Experiment 3: Chunked vs. Full-video captioning"
echo "  Variant      : ${VARIANT}"
echo "  Full config  : num_frames=${BEST_NF}  mbl=${BEST_MBL}"
echo "  Chunk size   : ${CHUNK_MINS} min  (fps=${VIDEO_FPS})"
echo "  Output       : ${OUT_DIR}"
echo "  Narration GT : ${NARRATION_PATH:-"(not set — QA eval skipped)"}"
if [ -n "${NARRATION_PATH}" ]; then
echo "  QA model     : ${QA_MODEL}  (questions: ${QA_N_QUESTIONS})"
fi
echo "============================================================"

# ---- Step 1A: Chunked inference ---------------------------------------------
echo ""
echo "[Step 1A] Chunked inference (${CHUNK_MINS}-min chunks)…"

# Read video IDs from annotation JSON and loop over them
mapfile -t VIDEO_IDS < <(python3 -c "
import json, sys
with open('${ANN_PATH}') as f:
    anns = json.load(f)
for a in anns:
    # strip _10min / _30min suffix to match frame directory names
    vid = a['video_id']
    for sfx in ('_10min', '_30min', '_60min'):
        vid = vid.replace(sfx, '')
    print(vid)
")

n_chunked=0
n_chunked_skip=0
for VID in "${VIDEO_IDS[@]}"; do
    VID_FRAME_DIR="${FRAME_DIR}/${VID}"
    OUT_FILE="${OUT_DIR}/chunked/${VID}.json"

    if [ ! -d "${VID_FRAME_DIR}" ]; then
        echo "  [skip] ${VID}: frame dir not found"
        continue
    fi

    if [ -f "${OUT_FILE}" ]; then
        echo "  [skip] ${VID}: already done"
        n_chunked_skip=$(( n_chunked_skip + 1 ))
        continue
    fi

    echo "  Captioning ${VID}…"
    python scripts/infer_chunked.py \
        --frame_dir            "${VID_FRAME_DIR}" \
        --output               "${OUT_FILE}" \
        --chunk_duration_min   "${CHUNK_MINS}" \
        --video_fps            "${VIDEO_FPS}" \
        --num_frames_per_chunk 80 \
        --memory_bank_length   40

    n_chunked=$(( n_chunked + 1 ))
done

echo "  Chunked: ${n_chunked} run, ${n_chunked_skip} skipped."

# ---- Step 1B: Full-video inference ------------------------------------------
FULL_OUT="${OUT_DIR}/full_captions.json"
echo ""
echo "[Step 1B] Full-video inference (num_frames=${BEST_NF}, mbl=${BEST_MBL})…"

if [ -f "${FULL_OUT}" ]; then
    echo "  Already done — skipping (delete ${FULL_OUT} to re-run)."
else
    python scripts/infer.py \
        --frame_dir          "${FRAME_DIR}" \
        --ann_path           "${ANN_PATH}" \
        --output             "${FULL_OUT}" \
        --num_frames         "${BEST_NF}" \
        --memory_bank_length "${BEST_MBL}"
fi

# ---- Step 2: Normalize chunked outputs to flat [{video_id, caption}] --------
CHUNKED_FLAT="${OUT_DIR}/chunked_captions.json"
echo ""
echo "[Step 2] Normalizing chunked outputs to flat caption format…"
python3 -c "
import json, pathlib, sys

chunked_dir = pathlib.Path('${OUT_DIR}/chunked')
records = []
for p in sorted(chunked_dir.glob('*.json')):
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        print(f'  [skip] {p.name}: {e}', file=sys.stderr)
        continue
    vid  = d.get('video_id', p.stem)
    # Prefer Vicuna summary; fall back to concatenated chunk captions
    cap  = d.get('summary') or ' '.join(
        c['caption'] for c in d.get('chunks', []) if c.get('caption')
    )
    if cap:
        records.append({'video_id': vid, 'caption': cap.strip()})

pathlib.Path('${CHUNKED_FLAT}').write_text(json.dumps(records, indent=2))
print(f'  {len(records)} chunked captions written to ${CHUNKED_FLAT}')
"

# ---- Step 3: QA evaluation --------------------------------------------------
if [ -z "${NARRATION_PATH}" ]; then
    echo ""
    echo "WARNING: NARRATION_PATH not set — skipping QA evaluation."
    echo "  Set NARRATION_PATH and ANTHROPIC_API_KEY to enable it."
elif [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo ""
    echo "WARNING: ANTHROPIC_API_KEY not set — skipping QA evaluation."
else
    SHARED_CACHE="${OUT_DIR}/qa_cache.json"

    echo ""
    echo "[Step 3A] QA evaluation — chunked approach…"
    python scripts/qa_eval.py \
        --captions_json "${CHUNKED_FLAT}" \
        --narration     "${NARRATION_PATH}" \
        --ann_path      "${ANN_PATH}" \
        --output        "${OUT_DIR}/chunked_qa.json" \
        --cache         "${SHARED_CACHE}" \
        --model         "${QA_MODEL}" \
        --n_questions   "${QA_N_QUESTIONS}"

    echo ""
    echo "[Step 3B] QA evaluation — full-video approach…"
    python scripts/qa_eval.py \
        --captions_json "${FULL_OUT}" \
        --narration     "${NARRATION_PATH}" \
        --ann_path      "${ANN_PATH}" \
        --output        "${OUT_DIR}/full_qa.json" \
        --cache         "${SHARED_CACHE}" \
        --model         "${QA_MODEL}" \
        --n_questions   "${QA_N_QUESTIONS}"

    # ---- Step 4: Plot comparison --------------------------------------------
    echo ""
    echo "[Step 4] Generating comparison plot…"
    python scripts/plot_comparison.py \
        --chunked_qa "${OUT_DIR}/chunked_qa.json" \
        --full_qa    "${OUT_DIR}/full_qa.json" \
        --plot_dir   "${PLOT_DIR}" \
        --chunk_mins "${CHUNK_MINS}" \
        --best_nf    "${BEST_NF}" \
        --best_mbl   "${BEST_MBL}"
fi

echo ""
echo "============================================================"
echo "Done.  Results in: ${OUT_DIR}"
echo "  Plots         : ${PLOT_DIR}/qa_comparison.png"
echo "  Chunked QA    : ${OUT_DIR}/chunked_qa.json"
echo "  Full-video QA : ${OUT_DIR}/full_qa.json"
echo "============================================================"
