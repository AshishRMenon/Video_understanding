#!/bin/bash
# Download 5 long Ego4D videos on RunPod (or any Linux server)
#
# USAGE:
#   export EGO4D_AWS_KEY="your_access_key_id"
#   export EGO4D_AWS_SECRET="your_secret_access_key"
#   bash scripts/download_ego4d_runpod.sh
#
# Credentials are read from environment variables — never hardcode them.
# Get credentials after signing the Ego4D license at:
#   https://ego4d-data.org/docs/start-here/#cli-download

set -e

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/data/ego4d/videos}"
META_DIR="${META_DIR:-/workspace/ego4d_meta}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-3600}"   # 1 hour minimum

# ── Validate credentials ──────────────────────────────────
if [ -z "$EGO4D_AWS_KEY" ] || [ -z "$EGO4D_AWS_SECRET" ]; then
    echo ""
    echo "ERROR: AWS credentials not set."
    echo ""
    echo "Export them before running this script:"
    echo "  export EGO4D_AWS_KEY=\"your_access_key_id\""
    echo "  export EGO4D_AWS_SECRET=\"your_secret_access_key\""
    echo ""
    echo "Get credentials after signing the Ego4D license at:"
    echo "  https://ego4d-data.org/docs/start-here/#cli-download"
    exit 1
fi

AWS_REGION="${AWS_REGION:-us-west-1}"

echo "================================================"
echo " Ego4D Long Video Downloader"
echo " Output dir : $OUTPUT_DIR"
echo " Min duration: ${MIN_DURATION_SEC}s ($(( MIN_DURATION_SEC / 60 )) min)"
echo "================================================"

# Step 1: Write AWS credentials (session-scoped, cleared at end)
echo "[1/5] Configuring AWS credentials..."
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = ${EGO4D_AWS_KEY}
aws_secret_access_key = ${EGO4D_AWS_SECRET}
EOF
cat > ~/.aws/config << EOF
[default]
region = ${AWS_REGION}
EOF
echo "  Done."

# Ensure credentials are cleaned up even on failure
cleanup() {
    echo ""
    echo "[cleanup] Removing ~/.aws/credentials ..."
    rm -f ~/.aws/credentials
}
trap cleanup EXIT

# Step 2: Install dependencies
echo "[2/5] Installing ego4d CLI and awscli..."
pip install -q ego4d awscli
echo "  Done."

# Step 3: Download metadata
echo "[3/5] Downloading Ego4D metadata (annotations only)..."
mkdir -p "$META_DIR"
ego4d \
    --output_directory "$META_DIR" \
    --datasets annotations \
    2>&1 | tail -5
echo "  Done."

# Step 4: Select 5 diverse long videos
echo "[4/5] Selecting 5 long videos..."

UIDS_FILE="/tmp/ego4d_selected_uids.txt"

python3 << PYEOF
import json, pathlib, sys

meta_candidates = list(pathlib.Path("${META_DIR}").rglob("ego4d.json"))
if not meta_candidates:
    print("ERROR: ego4d.json not found under ${META_DIR}")
    sys.exit(1)

with open(meta_candidates[0]) as f:
    d = json.load(f)

long_videos = sorted(
    [(v["video_uid"], v["duration_sec"]) for v in d["videos"]
     if v.get("duration_sec", 0) >= ${MIN_DURATION_SEC}],
    key=lambda x: -x[1]
)

# Pick diverse durations: 2 very long (>=90min), 3 medium-long (~60-90min)
very_long = [v for v in long_videos if v[1] >= 5400][:2]
medium    = [v for v in long_videos if 3600 <= v[1] < 5400][:3]
selected  = very_long + medium
if len(selected) < 5:
    selected = long_videos[:5]

print(f"  {'UID':<40} {'Duration (min)'}")
print(f"  {'-'*55}")
for uid, dur in selected:
    print(f"  {uid}  {dur/60:.1f} min")

with open("${UIDS_FILE}", "w") as f:
    for uid, _ in selected:
        f.write(uid + "\n")
print(f"\n  UIDs written to ${UIDS_FILE}")
PYEOF

if [ ! -f "$UIDS_FILE" ]; then
    echo "ERROR: UID selection failed."
    exit 1
fi

# Step 5: Download videos
echo ""
echo "[5/5] Downloading videos to $OUTPUT_DIR ..."
mkdir -p "$OUTPUT_DIR"

while IFS= read -r uid; do
    echo ""
    echo "  --> $uid"
    ego4d \
        --output_directory "$OUTPUT_DIR" \
        --datasets full_scale \
        --video_uids "$uid" \
        2>&1 | grep -v "^$" | tail -10
done < "$UIDS_FILE"

echo ""
echo "================================================"
echo " Download complete!"
echo " Videos in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo " (check ego4d output above for file locations)"
echo "================================================"
# Note: ~/.aws/credentials is removed automatically by the EXIT trap above
