#!/usr/bin/env bash
# =============================================================================
# Experiment 4 — Chunk-size sweep on 30-min videos.
#
# Runs chunked captioning at three chunk sizes on 30-min Ego4D videos and
# evaluates each with the full metric suite:
#   • ROUGE-L    (lexical overlap with GT summaries)
#   • BERTScore  (semantic similarity with GT summaries)
#   • Macro QA   (main-theme coverage — LLM judge)
#   • Micro QA   (fine-grained detail coverage — LLM judge)
#
# All four metrics are plotted together on a single graph (Graph 2):
#   X-axis: chunk size (2 min / 5 min / 10 min)
#   Y-axis: score (0–1)
#
# Usage:
#   NARRATION_PATH=/workspace/ego4d_meta/v2/annotations/narration.json \
#   ANTHROPIC_API_KEY=sk-ant-... \
#   bash run_chunk_sweep.sh
#
# Env-var overrides (all optional):
#   CHUNK_SIZES       space-separated list of chunk durations in minutes
#                     (default: "2 5 10")
#   VIDEO_FPS         FPS used during frame extraction     (default: 10)
#   NFRAMES_PER_CHUNK MA-LMM frames sampled per chunk      (default: 80)
#   MBL_PER_CHUNK     memory_bank_length per chunk         (default: 40)
#   NARRATION_PATH    Ego4D narration.json                 (required for metrics)
#   ANTHROPIC_API_KEY Anthropic key for QA eval            (required for QA)
#   QA_MODEL          LLM model for QA eval                (default: claude-haiku-4-5-20251001)
#   QA_N_MACRO        macro questions per video            (default: 8)
#   QA_N_MICRO        micro questions per video            (default: 8)
#   OUT_PARENT        parent output directory              (default: outputs)
#
# Output: outputs/exp4_chunk_sweep/
#   {2,5,10}min/                  per-video JSONs from infer_chunked.py
#   captions_{2,5,10}min.json     flat [{video_id, caption}] for each chunk size
#   qa_cache.json                 shared question cache across all chunk sizes
#   qa_{2,5,10}min.json           QA scores per chunk size
#   plots/chunk_sweep.png         combined metrics graph (Graph 2)
#
# Prerequisites:
#   - conda activate malmm_baseline
#   - bash prepare_frames.sh  (frames under data/frames/30min/ must exist)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VARIANT="30min"
CHUNK_SIZES="${CHUNK_SIZES:-2 5 10}"
VIDEO_FPS="${VIDEO_FPS:-10}"
NFRAMES_PER_CHUNK="${NFRAMES_PER_CHUNK:-80}"
MBL_PER_CHUNK="${MBL_PER_CHUNK:-40}"

NARRATION_PATH="${NARRATION_PATH:-}"
QA_MODEL="${QA_MODEL:-claude-haiku-4-5-20251001}"
QA_N_MACRO="${QA_N_MACRO:-8}"
QA_N_MICRO="${QA_N_MICRO:-8}"

OUT_PARENT="${OUT_PARENT:-outputs}"
OUT_DIR="${OUT_PARENT}/exp4_chunk_sweep"
PLOT_DIR="${OUT_DIR}/plots"
ANN_PATH="data/annotations/ego4d_${VARIANT}.json"
FRAME_ROOT="data/frames/${VARIANT}"

mkdir -p "${OUT_DIR}" "${PLOT_DIR}"

if [ ! -d "${FRAME_ROOT}" ]; then
    echo "ERROR: ${FRAME_ROOT} not found. Run prepare_frames.sh first." >&2
    exit 1
fi
if [ ! -f "${ANN_PATH}" ]; then
    echo "ERROR: ${ANN_PATH} not found." >&2
    exit 1
fi

echo "============================================================"
echo "Experiment 4: Chunk-size sweep on ${VARIANT} videos"
echo "  Chunk sizes  : ${CHUNK_SIZES} min"
echo "  Frames/chunk : ${NFRAMES_PER_CHUNK}  MBL/chunk: ${MBL_PER_CHUNK}"
echo "  Output       : ${OUT_DIR}"
echo "  Narration GT : ${NARRATION_PATH:-"(not set — metrics skipped)"}"
echo "============================================================"

# Read video IDs (strip variant suffix to match frame dir names)
mapfile -t VIDEO_IDS < <(python3 -c "
import json
with open('${ANN_PATH}') as f:
    anns = json.load(f)
for a in anns:
    vid = a['video_id']
    for sfx in ('_10min','_30min','_60min'):
        vid = vid.replace(sfx,'')
    print(vid)
")

# ---- Step 1: Inference for each chunk size ---------------------------------
for CHUNK_MINS in ${CHUNK_SIZES}; do
    CHUNK_DIR="${OUT_DIR}/${CHUNK_MINS}min"
    FLAT_JSON="${OUT_DIR}/captions_${CHUNK_MINS}min.json"
    mkdir -p "${CHUNK_DIR}"

    echo ""
    echo "--- Chunk size: ${CHUNK_MINS} min ---"

    n_run=0; n_skip=0
    for VID in "${VIDEO_IDS[@]}"; do
        VID_FRAME_DIR="${FRAME_ROOT}/${VID}"
        OUT_FILE="${CHUNK_DIR}/${VID}.json"

        if [ ! -d "${VID_FRAME_DIR}" ]; then
            echo "  [skip] ${VID}: frame dir missing"; continue
        fi
        if [ -f "${OUT_FILE}" ]; then
            echo "  [skip] ${VID}: already done"; n_skip=$((n_skip+1)); continue
        fi

        echo "  Captioning ${VID} (${CHUNK_MINS}-min chunks)…"
        python scripts/infer_chunked.py \
            --frame_dir            "${VID_FRAME_DIR}" \
            --output               "${OUT_FILE}" \
            --chunk_duration_min   "${CHUNK_MINS}" \
            --video_fps            "${VIDEO_FPS}" \
            --num_frames_per_chunk "${NFRAMES_PER_CHUNK}" \
            --memory_bank_length   "${MBL_PER_CHUNK}"
        n_run=$((n_run+1))
    done
    echo "  Done: ${n_run} run, ${n_skip} skipped."

    # Flatten: extract summary (or join chunk captions) → flat JSON
    if [ ! -f "${FLAT_JSON}" ] || [ "${n_run}" -gt 0 ]; then
        python3 -c "
import json, pathlib, sys
records = []
for p in sorted(pathlib.Path('${CHUNK_DIR}').glob('*.json')):
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        print(f'  [skip] {p.name}: {e}', file=sys.stderr); continue
    vid = d.get('video_id', p.stem)
    cap = d.get('summary') or ' '.join(
        c['caption'] for c in d.get('chunks',[]) if c.get('caption'))
    if cap:
        records.append({'video_id': vid, 'caption': cap.strip()})
pathlib.Path('${FLAT_JSON}').write_text(json.dumps(records, indent=2))
print(f'  Flattened {len(records)} captions → ${FLAT_JSON}')
"
    fi
done

# ---- Step 2: QA evaluation (if API key available) --------------------------
SHARED_CACHE="${OUT_DIR}/qa_cache.json"

if [ -z "${NARRATION_PATH}" ]; then
    echo ""
    echo "WARNING: NARRATION_PATH not set — skipping all metric evaluation."
elif [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo ""
    echo "WARNING: ANTHROPIC_API_KEY not set — skipping QA evaluation."
    echo "  ROUGE-L and BERTScore will still be computed during plotting."
else
    for CHUNK_MINS in ${CHUNK_SIZES}; do
        FLAT_JSON="${OUT_DIR}/captions_${CHUNK_MINS}min.json"
        QA_OUT="${OUT_DIR}/qa_${CHUNK_MINS}min.json"

        if [ -f "${QA_OUT}" ]; then
            echo ""; echo "QA ${CHUNK_MINS}min: already done (delete to re-run)."; continue
        fi

        echo ""; echo "[QA] chunk size ${CHUNK_MINS} min…"
        python scripts/qa_eval.py \
            --captions_json "${FLAT_JSON}" \
            --narration     "${NARRATION_PATH}" \
            --ann_path      "${ANN_PATH}" \
            --output        "${QA_OUT}" \
            --cache         "${SHARED_CACHE}" \
            --model         "${QA_MODEL}" \
            --n_macro       "${QA_N_MACRO}" \
            --n_micro       "${QA_N_MICRO}"
    done
fi

# ---- Step 3: Combined metrics plot (Graph 2) --------------------------------
echo ""
echo "Generating Graph 2 (chunk_sweep.png)…"

NARR_ARGS=()
if [ -n "${NARRATION_PATH}" ]; then
    NARR_ARGS=(--narration "${NARRATION_PATH}" --ann_path "${ANN_PATH}")
fi

python scripts/plot_chunk_sweep.py \
    --out_dir     "${OUT_DIR}" \
    --chunk_sizes ${CHUNK_SIZES} \
    --plot_dir    "${PLOT_DIR}" \
    "${NARR_ARGS[@]}"

echo ""
echo "============================================================"
echo "Done.  Graph 2 → ${PLOT_DIR}/chunk_sweep.png"
echo "============================================================"
