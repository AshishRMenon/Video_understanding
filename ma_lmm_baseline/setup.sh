#!/usr/bin/env bash
# Clean MA-LMM baseline setup. Clones the official repo, creates a conda env,
# installs deps exactly as the official README prescribes, and grabs Vicuna-7b.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${HERE}/MA-LMM"
ENV_NAME="${ENV_NAME:-malmm_baseline}"
PY_VER="${PY_VER:-3.9}"

# 1. Clone the official repo verbatim
if [ ! -d "${REPO_DIR}" ]; then
    echo "[1/4] Cloning official MA-LMM repo..."
    git clone https://github.com/boheumd/MA-LMM.git "${REPO_DIR}"
else
    echo "[1/4] MA-LMM already cloned at ${REPO_DIR}"
fi

# 2. Conda env. Python 3.9 matches what the LAVIS-derived deps were tested with.
echo "[2/4] Creating conda env '${ENV_NAME}' (python ${PY_VER})..."
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi

# Source conda for the current shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 3. Install. The official repo uses `pip install -e .` with its requirements.txt.
# Pin PyTorch to a version compatible with the older deps (timm==0.4.12, fairscale==0.4.4).
echo "[3/4] Installing PyTorch 2.0.1 + CUDA 11.8 wheels..."
pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo "      Installing MA-LMM (editable) + its requirements..."
pip install -e "${REPO_DIR}"

# Extras we need that aren't in MA-LMM's requirements
pip install ffmpeg-python imageio

# 4. Vicuna-7b weights (the official model used in the paper)
LLM_DIR="${REPO_DIR}/llm/vicuna-7b"
if [ ! -d "${LLM_DIR}" ] || [ -z "$(ls -A "${LLM_DIR}" 2>/dev/null)" ]; then
    echo "[4/4] Downloading Vicuna-7b-v1.5 weights..."
    pip install huggingface_hub
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id='lmsys/vicuna-7b-v1.5', local_dir='${LLM_DIR}')
PY
else
    echo "[4/4] Vicuna weights already present at ${LLM_DIR}"
fi

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Next step:            bash prepare_frames.sh"
