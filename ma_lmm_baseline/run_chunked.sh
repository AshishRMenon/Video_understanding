#!/usr/bin/env bash
# =============================================================================
# Experiment 1 — Hierarchical chunk-level captioning of long videos.
#
# Instead of running MA-LMM once on a full 60-minute video, this script splits
# each video into fixed-duration temporal chunks, captions each chunk with
# MA-LMM independently, then uses Vicuna (already loaded) to merge the chunk
# captions into a single coherent video-level summary.
#
# Two canonical configurations for a 60-minute video:
#   (a) 6 chunks × 10 min  →  CHUNK_MINS=10  (default)
#   (b) 60 chunks × 1 min  →  CHUNK_MINS=1
#
# Usage:
#   bash run_chunked.sh                     # 10-min chunks, 60-min videos
#   CHUNK_MINS=1  bash run_chunked.sh       # 1-min chunks, 60-min videos
#   VARIANT=30min bash run_chunked.sh       # run on 30-min videos instead
#   NO_SUMMARIZE=1 bash run_chunked.sh      # skip Vicuna summarization step
#
# Env-var overrides (all optional):
#   CHUNK_MINS      chunk duration in minutes           (default: 10)
#   VARIANT         which duration split to process     (default: 60min)
#   NFRAMES         MA-LMM frames sampled per chunk     (default: 80)
#   MBL             memory_bank_length per chunk        (default: 40)
#   VIDEO_FPS       FPS used during frame extraction    (default: 10)
#   NO_SUMMARIZE    set to 1 to skip Vicuna summary     (default: off)
#
# Output directory:
#   outputs/exp1_chunked/<CHUNK_MINS>min_chunks/<video_id>.json
#
# Each JSON file contains:
#   - chunk-level captions with timing and compute stats
#   - a Vicuna-generated summary of the full video (unless NO_SUMMARIZE=1)
#
# Prerequisites:
#   - conda activate malmm_baseline
#   - bash prepare_frames.sh  (frames must exist under data/frames/<VARIANT>/)
#
# To run both configurations back-to-back:
#   CHUNK_MINS=10 bash run_chunked.sh
#   CHUNK_MINS=1  bash run_chunked.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

CHUNK_MINS="${CHUNK_MINS:-10}"
VARIANT="${VARIANT:-60min}"
NFRAMES="${NFRAMES:-80}"
MBL="${MBL:-40}"
VIDEO_FPS="${VIDEO_FPS:-10}"
NO_SUMMARIZE="${NO_SUMMARIZE:-}"

OUT_DIR="outputs/exp1_chunked/${CHUNK_MINS}min_chunks"
FRAME_ROOT="data/frames/${VARIANT}"

mkdir -p "${OUT_DIR}"

if [ ! -d "${FRAME_ROOT}" ]; then
    echo "ERROR: ${FRAME_ROOT} not found. Run prepare_frames.sh first." >&2
    exit 1
fi

NO_SUM_ARG=()
if [ -n "${NO_SUMMARIZE}" ]; then
    NO_SUM_ARG=(--no_summarize)
fi

echo "============================================================"
echo "Experiment 1: Chunk-level captioning"
echo "  Variant      : ${VARIANT}"
echo "  Chunk size   : ${CHUNK_MINS} min"
echo "  NFRAMES/chunk: ${NFRAMES}"
echo "  MBL/chunk    : ${MBL}"
echo "  Output dir   : ${OUT_DIR}"
echo "============================================================"

n_processed=0
n_skipped=0

for vid_dir in "${FRAME_ROOT}"/*/; do
    [ -d "${vid_dir}" ] || continue
    vid_id=$(basename "${vid_dir}")
    out_file="${OUT_DIR}/${vid_id}.json"

    if [ -f "${out_file}" ]; then
        echo "[skip] ${vid_id} (output already exists)"
        n_skipped=$(( n_skipped + 1 ))
        continue
    fi

    echo ""
    echo "--- ${vid_id} ---"
    python scripts/infer_chunked.py \
        --frame_dir          "${vid_dir}" \
        --output             "${out_file}" \
        --chunk_duration_min "${CHUNK_MINS}" \
        --video_fps          "${VIDEO_FPS}" \
        --num_frames_per_chunk "${NFRAMES}" \
        --memory_bank_length "${MBL}" \
        "${NO_SUM_ARG[@]}"

    n_processed=$(( n_processed + 1 ))
done

echo ""
echo "============================================================"
echo "Done.  Processed: ${n_processed}  Skipped: ${n_skipped}"
echo "Results : ${OUT_DIR}/"
echo ""
echo "To run the 1-min chunk configuration:"
echo "  CHUNK_MINS=1 bash run_chunked.sh"
echo "============================================================"
