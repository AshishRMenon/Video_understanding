#!/bin/bash
# Setup script for VideoRecap (official repo)
# Run from: MA_LMM_coding/
#
# After this script, manually download the 3 model checkpoints from Google Drive
# (see STEP 3 below) — Drive downloads cannot be automated reliably.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "================================================"
echo " VideoRecap Setup"
echo "================================================"

# ── STEP 1: Clone repo ──────────────────────────────
echo ""
echo "[1/4] Cloning VideoRecap repo..."
if [ -d "VideoRecap" ]; then
    echo "  [skip] VideoRecap/ already exists"
else
    git clone https://github.com/md-mohaiminul/VideoRecap.git
    echo "  Cloned to VideoRecap/"
fi

# ── STEP 2: Install dependencies ────────────────────
echo ""
echo "[2/4] Installing VideoRecap dependencies..."
pip install -q \
    transformers \
    evaluate \
    einops \
    timm \
    decord \
    h5py \
    moviepy \
    absl-py

# Install VideoRecap itself
pip install -q -e VideoRecap/
echo "  Done."

# ── STEP 3: Download LaViLa encoder checkpoint ──────
echo ""
echo "[3/4] Downloading LaViLa video encoder checkpoint..."
mkdir -p VideoRecap/pretrained_models

LAVILA_CKPT="VideoRecap/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth"
if [ -f "$LAVILA_CKPT" ]; then
    echo "  [skip] LaViLa checkpoint already exists"
else
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.baseline.ep_0003.pth" \
        -O "$LAVILA_CKPT"
    echo "  Saved to $LAVILA_CKPT"
fi

# ── STEP 4: Manual downloads required ───────────────
echo ""
echo "[4/4] *** MANUAL DOWNLOAD REQUIRED ***"
echo ""
echo "  Download the 3 VideoRecap model checkpoints from Google Drive:"
echo "  https://drive.google.com/drive/folders/1KlIbqhZ2lfngs0hc32zK2nnMVquYfzaC"
echo ""
echo "  Place them at:"
echo "    VideoRecap/pretrained_models/videorecap_clip.pt"
echo "    VideoRecap/pretrained_models/videorecap_segment.pt"
echo "    VideoRecap/pretrained_models/videorecap_video.pt"
echo ""
echo "  Alternatively, download the unified checkpoint (all 3 levels in one):"
echo "  https://drive.google.com/file/d/1hsfm8ayCsOTFlVjvI-LCPXfS9MPstUwP"
echo "  Place at: VideoRecap/pretrained_models/videorecap_u.pt"
echo ""

# Check what's already there
echo "  Current state of VideoRecap/pretrained_models/:"
ls -lh VideoRecap/pretrained_models/ 2>/dev/null || echo "    (empty)"

echo ""
echo "================================================"
echo " Once checkpoints are downloaded, run:"
echo "   bash run_comparison.sh"
echo "================================================"
