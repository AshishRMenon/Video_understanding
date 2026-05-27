#!/bin/bash
# Master script for Exp-01: MA-LMM on Ego4D hour-long videos
# Run from: /home/ashish95/video_captioning_coding_exploration_ground/MA_LMM_coding/

set -e
CONDA_ENV="malmm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "Exp-01: MA-LMM Memory Saturation on Ego4D"
echo "================================================"

# ── STEP 0: Prerequisites check ──────────────────────
echo -e "\n[Step 0] Checking prerequisites..."

if [ ! -d "MA-LMM/llm/vicuna-7b" ]; then
    echo "ERROR: Vicuna-7b not found at MA-LMM/llm/vicuna-7b"
    echo "Download from: https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md"
    exit 1
fi

if [ -z "$(ls data/ego4d/videos/*.mp4 2>/dev/null)" ]; then
    echo "WARNING: No Ego4D videos found in data/ego4d/videos/"
    echo "Get access at: https://ego4d-data.org/docs/start-here/"
    echo "Then download 4-5 long videos and place .mp4 files in data/ego4d/videos/"
    exit 1
fi
echo "  Prerequisites OK"

# ── STEP 1: Extract frames ────────────────────────────
echo -e "\n[Step 1] Extracting frames at 1 FPS..."
python scripts/extract_frames_ego4d.py \
    --video_dir data/ego4d/videos \
    --out_dir data/ego4d/frames \
    --fps 1.0 \
    --short_side 256 \
    --min_duration 1800

# ── STEP 2: Create annotations ───────────────────────
echo -e "\n[Step 2] Creating annotation JSON..."
EGO4D_HCAP=""  # Set this if you have the Ego4D-HCap GT captions JSON

if [ -f "data/ego4d/annotations/ego4dhcap_gt.json" ]; then
    EGO4D_HCAP="--ego4d_hcap_path data/ego4d/annotations/ego4dhcap_gt.json"
fi

python scripts/create_ego4d_annotation.py \
    --frame_dir data/ego4d/frames \
    --out_path data/ego4d/annotations/ego4d_eval.json \
    $EGO4D_HCAP

# ── STEP 3: Run MA-LMM inference ─────────────────────
echo -e "\n[Step 3] Running MA-LMM inference..."
CKPT=""
if [ -f "MA-LMM/saved_model/youcook2/checkpoint_best.pth" ]; then
    CKPT="--ckpt_path MA-LMM/saved_model/youcook2/checkpoint_best.pth"
fi

python scripts/run_inference_ego4d.py \
    --video_dir data/ego4d/frames \
    --ann_path data/ego4d/annotations/ego4d_eval.json \
    --llm_model MA-LMM/llm/vicuna-7b \
    --memory_bank_length 40 \
    --num_frames 80 \
    --output_dir outputs/exp01_memory_saturation \
    --prompt "Describe in detail what the person does throughout this entire video from start to finish." \
    $CKPT

# ── STEP 4: Analyze results ──────────────────────────
echo -e "\n[Step 4] Analyzing memory saturation..."
python scripts/analyze_memory_saturation.py \
    --results_dir outputs/exp01_memory_saturation \
    --out_dir outputs/exp01_memory_saturation/analysis

echo -e "\n================================================"
echo "Experiment complete!"
echo "Results: outputs/exp01_memory_saturation/"
echo "Analysis: outputs/exp01_memory_saturation/analysis/"
echo "================================================"
