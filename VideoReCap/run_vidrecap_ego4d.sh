#!/bin/bash
# =============================================================================
#  VideoRecap — Ego4D Baseline Pipeline
#  Usage:  conda activate malmm_baseline && bash run_vidrecap_ego4d.sh
#
#  Edit the CONFIG block below to point to different data or checkpoints.
#  Run a single step:   bash run_vidrecap_ego4d.sh --step 2
# =============================================================================

set -e   # stop on first error

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — PATHS  (edit these to point to your data / checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

# Input videos — directory containing *_10min.mp4 files
VIDEO_DIR="/workspace/data/ego4d/videos/v2/full_scale"

# Pretrained model checkpoints
LAVILA_CKPT="/workspace/MA_LMM_coding/VideoRecap/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth"
CLIP_CKPT="/workspace/MA_LMM_coding/VideoRecap/pretrained_models/videorecap_clip.pt"
SEGMENT_CKPT="/workspace/MA_LMM_coding/VideoRecap/pretrained_models/videorecap_segment.pt"
VIDEO_CKPT="/workspace/MA_LMM_coding/VideoRecap/pretrained_models/videorecap_video.pt"

# Where intermediate and final outputs are saved
FEATURE_DIR="/workspace/data/ego4d/vidrecap_features"    # step 1 → .npy files
OUTPUT_DIR="/workspace/Video_understanding/VideoReCap/outputs"

# MA-LMM results for comparison (step 5)
MALMM_DIR="/workspace/MA_LMM_coding/outputs/exp01_memory_saturation"

# Script directory (resolved relative to this file so you can run from anywhere)
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — HYPERPARAMETERS  (tweak to experiment)
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_GLOB="*_10min.mp4"   # which videos to process (glob, relative to VIDEO_DIR)
FEATURE_STEP=4             # seconds between extracted features  (step 1)
FRAMES_PER_CLIP=4          # frames fed to encoder per clip      (step 1)
CLIP_DURATION=4            # seconds per clip for captioning     (step 2)
SEGMENT_DURATION=180       # seconds per segment window          (step 3)

BATCH_SIZE_FEAT=16         # batch size for feature extraction   (step 1)
BATCH_SIZE_CLIP=8          # batch size for clip captioning      (step 2)
BATCH_SIZE_SEG=8           # batch size for segment descriptions (step 3)
BATCH_SIZE_VID=4           # batch size for video summaries      (step 4)

NUM_WORKERS=4              # DataLoader worker processes

# ─────────────────────────────────────────────────────────────────────────────
#  Parse --step argument (optional — runs a single step)
# ─────────────────────────────────────────────────────────────────────────────

START_STEP=1
END_STEP=5
while [[ $# -gt 0 ]]; do
    case $1 in
        --step) START_STEP=$2; END_STEP=$2; shift 2 ;;
        *)      echo "Unknown argument: $1"; echo "Usage: $0 [--step N]"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR/logs"

# ─────────────────────────────────────────────────────────────────────────────
#  Prerequisite check
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  VideoRecap  —  Ego4D Baseline Pipeline"
echo "============================================================"

MISSING=0
for FILE in "$LAVILA_CKPT" "$CLIP_CKPT" "$SEGMENT_CKPT" "$VIDEO_CKPT"; do
    [[ -f "$FILE" ]] || { echo "  MISSING: $FILE"; MISSING=1; }
done
VIDEO_COUNT=$(ls "$VIDEO_DIR"/$VIDEO_GLOB 2>/dev/null | wc -l)
[[ $VIDEO_COUNT -gt 0 ]] || { echo "  MISSING: no videos match $VIDEO_DIR/$VIDEO_GLOB"; MISSING=1; }
[[ $MISSING -eq 0 ]] || { echo ""; echo "Fix missing files and retry."; exit 1; }

echo "  Videos  : $VIDEO_COUNT  ($VIDEO_DIR/$VIDEO_GLOB)"
echo "  Running : steps $START_STEP–$END_STEP"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: announce a step, run Python, log output
# ─────────────────────────────────────────────────────────────────────────────

run_step() {
    local STEP=$1 ; local TITLE=$2 ; shift 2   # remaining args are the python command

    [[ $STEP -ge $START_STEP && $STEP -le $END_STEP ]] || return 0

    echo "------------------------------------------------------------"
    echo "  STEP $STEP — $TITLE"
    echo "------------------------------------------------------------"
    "$@" 2>&1 | tee "$OUTPUT_DIR/logs/step${STEP}.log"
    echo ""
    echo "  Step $STEP complete.  Log → $OUTPUT_DIR/logs/step${STEP}.log"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Extract LaViLa CLS features
#  Reads  : $VIDEO_DIR/*.mp4
#  Writes : $FEATURE_DIR/<vid>.npy   (shape: [n_clips, 768])
# ─────────────────────────────────────────────────────────────────────────────

run_step 1 "Extract CLS features (LaViLa TimeSformer)" \
    python3 "$SCRIPTS/01_extract_cls_features.py" \
        --video_dir    "$VIDEO_DIR"    \
        --output_dir   "$FEATURE_DIR"  \
        --lavila_ckpt  "$LAVILA_CKPT"  \
        --video_glob   "$VIDEO_GLOB"   \
        --feature_step $FEATURE_STEP   \
        --num_frames   $FRAMES_PER_CLIP \
        --batch_size   $BATCH_SIZE_FEAT \
        --num_workers  $NUM_WORKERS

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Level 1: Clip captions
#  Reads  : $VIDEO_DIR/*.mp4  (raw pixels)
#  Writes : $OUTPUT_DIR/clip_captions.pkl
#           $OUTPUT_DIR/clip_captions_summary.json
# ─────────────────────────────────────────────────────────────────────────────

run_step 2 "Generate clip captions (Level 1 — pixel-based)" \
    python3 "$SCRIPTS/02_generate_clip_captions.py" \
        --video_dir      "$VIDEO_DIR"   \
        --clip_ckpt      "$CLIP_CKPT"   \
        --output_dir     "$OUTPUT_DIR"  \
        --video_glob     "$VIDEO_GLOB"  \
        --clip_duration  $CLIP_DURATION \
        --batch_size     $BATCH_SIZE_CLIP \
        --num_workers    $NUM_WORKERS

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Level 2: Segment descriptions
#  Reads  : $OUTPUT_DIR/clip_captions.pkl   (from step 2)
#           $FEATURE_DIR/<vid>.npy           (from step 1)
#  Writes : $OUTPUT_DIR/segment_descriptions.pkl
#           $OUTPUT_DIR/segment_descriptions_summary.json
# ─────────────────────────────────────────────────────────────────────────────

run_step 3 "Generate segment descriptions (Level 2 — CLS features + clip captions)" \
    python3 "$SCRIPTS/03_generate_segment_descriptions.py" \
        --feature_dir       "$FEATURE_DIR"                         \
        --clip_captions     "$OUTPUT_DIR/clip_captions.pkl"        \
        --segment_ckpt      "$SEGMENT_CKPT"                        \
        --output_dir        "$OUTPUT_DIR"                          \
        --segment_duration  $SEGMENT_DURATION                      \
        --batch_size        $BATCH_SIZE_SEG                        \
        --num_workers       $NUM_WORKERS

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Level 3: Video summaries
#  Reads  : $OUTPUT_DIR/segment_descriptions.pkl  (from step 3)
#           $FEATURE_DIR/<vid>.npy                 (from step 1)
#  Writes : $OUTPUT_DIR/video_summaries.json
# ─────────────────────────────────────────────────────────────────────────────

run_step 4 "Generate video summaries (Level 3 — CLS features + segment descriptions)" \
    python3 "$SCRIPTS/04_generate_video_summaries.py" \
        --feature_dir    "$FEATURE_DIR"                              \
        --segment_descs  "$OUTPUT_DIR/segment_descriptions.pkl"     \
        --video_ckpt     "$VIDEO_CKPT"                              \
        --output_dir     "$OUTPUT_DIR"                              \
        --batch_size     $BATCH_SIZE_VID                            \
        --num_workers    $NUM_WORKERS

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Analyse and compare vs MA-LMM
#  Reads  : $MALMM_DIR/*_result.json
#           $OUTPUT_DIR/video_summaries.json
#  Writes : $OUTPUT_DIR/comparison_malmm_vs_vidrecap.json
# ─────────────────────────────────────────────────────────────────────────────

run_step 5 "Analyse results and compare vs MA-LMM" \
    python3 "$SCRIPTS/05_analyze_results.py" \
        --malmm_dir          "$MALMM_DIR"                              \
        --vidrecap_summaries "$OUTPUT_DIR/video_summaries.json"        \
        --vidrecap_segments  "$OUTPUT_DIR/segment_descriptions.pkl"    \
        --vidrecap_clips     "$OUTPUT_DIR/clip_captions.pkl"           \
        --output_dir         "$OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  Pipeline complete."
echo "============================================================"
echo ""
echo "  Key output files:"
echo "    $OUTPUT_DIR/clip_captions.pkl"
echo "    $OUTPUT_DIR/clip_captions_summary.json"
echo "    $OUTPUT_DIR/segment_descriptions.pkl"
echo "    $OUTPUT_DIR/segment_descriptions_summary.json"
echo "    $OUTPUT_DIR/video_summaries.json           ← main result"
echo "    $OUTPUT_DIR/comparison_malmm_vs_vidrecap.json"
echo ""
