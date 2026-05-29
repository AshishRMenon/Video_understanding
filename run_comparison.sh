#!/bin/bash
# Master comparison script: runs MA-LMM and VideoRecap on the same 5 Ego4D videos,
# then produces a side-by-side comparison report.
#
# Prerequisites (must be done once):
#   1. bash setup_env.sh          — conda env for MA-LMM
#   2. bash setup_vidrecap.sh     — clone VideoRecap, download LaViLa encoder
#   3. Manually download VideoRecap checkpoints from Google Drive (see setup_vidrecap.sh)
#   4. Download Vicuna-7b → MA-LMM/llm/vicuna-7b/
#   5. python3 scripts/download_ego4d_5videos.py  — get 5 videos + annotations
#
# Run:
#   bash run_comparison.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VIDEO_DIR="data/ego4d/videos"
FRAME_DIR="data/ego4d/frames"
ANN_PATH="data/ego4d/annotations/ego4d_eval.json"
NARRATION_JSON="data/ego4d/annotations/ego4d_5videos_annotations.json"
MALMM_OUT="outputs/exp01_memory_saturation"
VIDRECAP_OUT="outputs/vidrecap"
COMPARISON_OUT="outputs/comparison"

echo "================================================"
echo " MA-LMM vs VideoRecap Comparison Pipeline"
echo "================================================"

# ── Prerequisite checks ──────────────────────────────
echo ""
echo "[prereq] Checking prerequisites..."

MISSING=0

if [ -z "$(ls $VIDEO_DIR/*.mp4 2>/dev/null)" ]; then
    echo "  MISSING: Ego4D videos in $VIDEO_DIR"
    echo "    Run: python3 scripts/download_ego4d_5videos.py"
    MISSING=1
fi

if [ ! -d "MA-LMM/llm/vicuna-7b" ]; then
    echo "  MISSING: Vicuna-7b at MA-LMM/llm/vicuna-7b/"
    echo "    See README for download instructions"
    MISSING=1
fi

if [ ! -d "VideoRecap" ]; then
    echo "  MISSING: VideoRecap repo — run: bash setup_vidrecap.sh"
    MISSING=1
fi

for ckpt in videorecap_clip videorecap_segment videorecap_video; do
    if [ ! -f "VideoRecap/pretrained_models/${ckpt}.pt" ]; then
        echo "  MISSING: VideoRecap/pretrained_models/${ckpt}.pt"
        echo "    Download from: https://drive.google.com/drive/folders/1KlIbqhZ2lfngs0hc32zK2nnMVquYfzaC"
        MISSING=1
    fi
done

if [ ! -f "VideoRecap/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth" ]; then
    echo "  MISSING: LaViLa encoder checkpoint — run: bash setup_vidrecap.sh"
    MISSING=1
fi

if [ "$MISSING" = "1" ]; then
    echo ""
    echo "Fix the above prerequisites then re-run this script."
    exit 1
fi

echo "  All prerequisites OK."

# ── PART A: MA-LMM ──────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo " PART A: MA-LMM Inference"
echo "════════════════════════════════════════════════"

echo ""
echo "[A1] Extracting frames at 1 FPS..."
python3 scripts/extract_frames_ego4d.py \
    --video_dir "$VIDEO_DIR" \
    --out_dir "$FRAME_DIR" \
    --fps 1.0 \
    --short_side 256 \
    --min_duration 1800

echo ""
echo "[A2] Creating annotation JSON..."
HCAP_FLAG=""
if [ -f "$NARRATION_JSON" ]; then
    HCAP_FLAG="--ego4d_hcap_path $NARRATION_JSON"
fi
python3 scripts/create_ego4d_annotation.py \
    --frame_dir "$FRAME_DIR" \
    --out_path "$ANN_PATH" \
    $HCAP_FLAG

echo ""
echo "[A3] Running MA-LMM inference..."
CKPT_FLAG=""
if [ -f "MA-LMM/saved_model/youcook2/checkpoint_best.pth" ]; then
    CKPT_FLAG="--ckpt_path MA-LMM/saved_model/youcook2/checkpoint_best.pth"
fi
mkdir -p "$MALMM_OUT"
python3 scripts/run_inference_ego4d.py \
    --video_dir "$FRAME_DIR" \
    --ann_path "$ANN_PATH" \
    --llm_model "MA-LMM/llm/vicuna-7b" \
    --memory_bank_length 40 \
    --num_frames 80 \
    --output_dir "$MALMM_OUT" \
    --prompt "Describe in detail what the person does throughout this entire video from start to finish." \
    $CKPT_FLAG

# ── PART B: VideoRecap ──────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo " PART B: VideoRecap Inference"
echo "════════════════════════════════════════════════"

echo ""
echo "[B1] Running VideoRecap 3-level pipeline..."
mkdir -p "$VIDRECAP_OUT"
python3 scripts/run_inference_vidrecap.py \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$VIDRECAP_OUT" \
    --vidrecap_repo "VideoRecap" \
    --clip_ckpt "VideoRecap/pretrained_models/videorecap_clip.pt" \
    --segment_ckpt "VideoRecap/pretrained_models/videorecap_segment.pt" \
    --video_ckpt "VideoRecap/pretrained_models/videorecap_video.pt" \
    --lavila_ckpt "VideoRecap/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth"

# ── PART C: Comparison ──────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo " PART C: Side-by-Side Comparison"
echo "════════════════════════════════════════════════"

echo ""
mkdir -p "$COMPARISON_OUT"
python3 scripts/compare_baselines.py \
    --malmm_dir "$MALMM_OUT" \
    --vidrecap_dir "$VIDRECAP_OUT" \
    --out_dir "$COMPARISON_OUT" \
    --narration_json "$NARRATION_JSON"

echo ""
echo "════════════════════════════════════════════════"
echo " All done!"
echo " MA-LMM results  : $MALMM_OUT/"
echo " VidRecap results: $VIDRECAP_OUT/"
echo " Comparison      : $COMPARISON_OUT/comparison_report.json"
echo " LaTeX table     : $COMPARISON_OUT/comparison_table.tex"
echo "════════════════════════════════════════════════"
