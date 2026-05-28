#!/bin/bash
# Quick end-to-end test of the MA-LMM captioning pipeline on YouTube clips.
#
# Downloads a few short YouTube segments, extracts frames, runs inference,
# and prints generated captions.
#
# USAGE:
#   bash scripts/test_youtube_pipeline.sh
#
# PREREQUISITES:
#   pip install yt-dlp
#   ffmpeg installed (apt install ffmpeg)
#   MA-LMM cloned to MA-LMM/  and pip install -e MA-LMM/
#   Vicuna-7b weights at MA-LMM/llm/vicuna-7b/
#
# To test with your own URLs, edit the VIDEOS array below.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

# ── Config ────────────────────────────────────────────────
CLIP_START=60          # start trimming at this second (skip intros)
CLIP_DURATION=60       # length of each clip in seconds
FPS=1.0                # frames per second to extract
VIDEO_DIR="data/youtube_test/videos"
FRAME_DIR="data/youtube_test/frames"
ANN_PATH="data/youtube_test/annotations/yt_eval.json"
OUT_DIR="outputs/youtube_test"
LLM_MODEL="MA-LMM/llm/vicuna-7b"

# ── Test videos: (label, YouTube URL) ─────────────────────
# Mix of eventful and sparse/repetitive — mirrors the Ego4D experiment intent.
declare -A VIDEOS=(
    ["cooking_tasty"]="https://www.youtube.com/watch?v=SIBs4PcqEaE"
    ["city_walk_pov"]="https://www.youtube.com/watch?v=P0RcELiJmzc"
    ["basketball_game"]="https://www.youtube.com/watch?v=8wYsavdZDRc"
)

# ─────────────────────────────────────────────────────────

echo "================================================"
echo " MA-LMM YouTube Clip Test Pipeline"
echo "================================================"

# ── Check dependencies ───────────────────────────────────
echo ""
echo "[check] Dependencies..."
for cmd in yt-dlp ffmpeg python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "  MISSING: $cmd"
        echo "  Install: pip install yt-dlp  /  apt install ffmpeg"
        exit 1
    fi
done
echo "  yt-dlp, ffmpeg, python3 — OK"

if [ ! -d "$LLM_MODEL" ]; then
    echo ""
    echo "WARNING: Vicuna-7b not found at $LLM_MODEL"
    echo "  Will run frame extraction only (no inference)."
    echo "  To get weights: see README section 'Download Vicuna-7b weights'"
    SKIP_INFERENCE=1
fi

# ── Step 1: Download and trim clips ─────────────────────
echo ""
echo "[1/4] Downloading and trimming YouTube clips..."
mkdir -p "$VIDEO_DIR"

for label in "${!VIDEOS[@]}"; do
    url="${VIDEOS[$label]}"
    out_path="$VIDEO_DIR/${label}.mp4"

    if [ -f "$out_path" ]; then
        echo "  [skip] $label already exists"
        continue
    fi

    echo "  --> $label"
    echo "      URL: $url"

    # Download best mp4 up to 720p (fast), then trim with ffmpeg
    tmp_path="/tmp/yt_raw_${label}.mp4"

    yt-dlp \
        -f "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best" \
        --merge-output-format mp4 \
        -o "$tmp_path" \
        "$url" \
        2>&1 | grep -E "Downloading|Merging|ERROR|WARNING" || true

    if [ ! -f "$tmp_path" ]; then
        echo "  ERROR: Download failed for $label — skipping"
        continue
    fi

    # Trim to CLIP_DURATION seconds starting at CLIP_START
    ffmpeg -y -ss "$CLIP_START" -i "$tmp_path" -t "$CLIP_DURATION" \
        -c:v libx264 -c:a aac \
        "$out_path" -loglevel error

    rm -f "$tmp_path"
    duration=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$out_path" 2>/dev/null | cut -d. -f1)
    echo "  OK  $out_path  (${duration}s)"
done

echo ""
ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null || { echo "No clips downloaded — exiting."; exit 1; }

# ── Step 2: Extract frames ───────────────────────────────
echo ""
echo "[2/4] Extracting frames at ${FPS} FPS..."
python3 scripts/extract_frames_ego4d.py \
    --video_dir "$VIDEO_DIR" \
    --out_dir "$FRAME_DIR" \
    --fps "$FPS" \
    --short_side 256 \
    --min_duration 30          # override: accept clips as short as 30s

echo ""
echo "  Frames:"
for d in "$FRAME_DIR"/*/; do
    count=$(find "$d" -name "*.jpg" | wc -l)
    echo "  $d  ($count frames)"
done

# ── Step 3: Create annotations ───────────────────────────
echo ""
echo "[3/4] Creating annotation JSON..."
mkdir -p "$(dirname "$ANN_PATH")"
python3 scripts/create_ego4d_annotation.py \
    --frame_dir "$FRAME_DIR" \
    --out_path "$ANN_PATH"
echo "  Saved: $ANN_PATH"

# ── Step 4: Run inference ────────────────────────────────
echo ""
if [ "${SKIP_INFERENCE:-0}" = "1" ]; then
    echo "[4/4] Skipping inference (no Vicuna weights)."
    echo "  Frame extraction succeeded — pipeline is working."
    echo "  Once you have Vicuna-7b weights, run:"
    echo ""
    echo "  python3 scripts/run_inference_ego4d.py \\"
    echo "      --video_dir $FRAME_DIR \\"
    echo "      --ann_path $ANN_PATH \\"
    echo "      --llm_model $LLM_MODEL \\"
    echo "      --memory_bank_length 40 \\"
    echo "      --num_frames 16 \\"
    echo "      --output_dir $OUT_DIR \\"
    echo "      --prompt \"Describe in detail what happens in this video.\""
else
    echo "[4/4] Running MA-LMM inference..."
    mkdir -p "$OUT_DIR"
    python3 scripts/run_inference_ego4d.py \
        --video_dir "$FRAME_DIR" \
        --ann_path "$ANN_PATH" \
        --llm_model "$LLM_MODEL" \
        --memory_bank_length 40 \
        --num_frames 16 \
        --output_dir "$OUT_DIR" \
        --prompt "Describe in detail what happens in this video."

    echo ""
    echo "================================================"
    echo " GENERATED CAPTIONS"
    echo "================================================"
    for f in "$OUT_DIR"/*_result.json; do
        [ -f "$f" ] || continue
        vid=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('video_id','?'))")
        cap=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('caption','N/A'))")
        div=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('memory_stats',{}).get('diversity_score','?'))")
        echo ""
        echo "[$vid]"
        echo "  Caption:   $cap"
        echo "  Diversity: $div"
    done
fi

echo ""
echo "================================================"
echo " Done. Results: $OUT_DIR"
echo "================================================"
