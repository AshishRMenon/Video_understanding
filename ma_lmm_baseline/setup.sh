#!/usr/bin/env bash
# Full bootstrap: installs conda + ffmpeg if missing, then sets up MA-LMM.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${HERE}/MA-LMM"
ENV_NAME="${ENV_NAME:-malmm_baseline}"
PY_VER="${PY_VER:-3.9}"

# ── 1. ffmpeg ─────────────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    echo "[1/6] ffmpeg already installed: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "[1/6] Installing ffmpeg..."
    apt-get update -qq && apt-get install -y --no-install-recommends ffmpeg
fi

# ── 2. Conda ──────────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    echo "[2/6] conda already available: $(conda --version)"
else
    echo "[2/6] Installing Miniconda..."
    MINICONDA_SH="/tmp/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o "${MINICONDA_SH}"
    bash "${MINICONDA_SH}" -b -p "${HOME}/miniconda3"
    rm -f "${MINICONDA_SH}"
    export PATH="${HOME}/miniconda3/bin:${PATH}"
    conda init bash
    echo "  Miniconda installed at ${HOME}/miniconda3"
    echo "  NOTE: open a new shell (or run: source ~/.bashrc) after this script"
    echo "        if 'conda activate' is not found in subsequent terminals."
fi

# Make conda available in this shell session
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── 3. Clone MA-LMM ───────────────────────────────────────────────────────────
if [ ! -d "${REPO_DIR}" ]; then
    echo "[3/6] Cloning official MA-LMM repo..."
    git clone https://github.com/boheumd/MA-LMM.git "${REPO_DIR}"
else
    echo "[3/6] MA-LMM already cloned at ${REPO_DIR}"
fi

# ── 4. Conda environment ──────────────────────────────────────────────────────
echo "[4/6] Creating conda env '${ENV_NAME}' (python ${PY_VER})..."
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi
conda activate "${ENV_NAME}"

# ── 5. Python dependencies ────────────────────────────────────────────────────
echo "[5/6] Installing PyTorch 2.0.1 + CUDA 11.8 wheels..."
pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo "      Pre-pinning spacy and transformers to compatible versions..."
pip install "spacy<3.8.0" "transformers>=4.28.0,<4.46.0"

echo "      Installing MA-LMM (editable) + its requirements..."
pip install -e "${REPO_DIR}"

pip install ffmpeg-python imageio huggingface_hub tqdm

# ── 6. Vicuna-7b-v1.3 weights ─────────────────────────────────────────────────
# IMPORTANT: must be v1.3 — the pretrained Q-Former checkpoint
# (instruct_blip_vicuna7b_trimmed.pth) was trained against v1.3's embedding
# space. Using v1.5 causes degenerate output (repeated "O"s / "1. 1. 1.").
LLM_DIR="${REPO_DIR}/llm/vicuna-7b-v1.3"
SHARD1="${LLM_DIR}/pytorch_model-00001-of-00002.bin"
SHARD2="${LLM_DIR}/pytorch_model-00002-of-00002.bin"
if [ -f "${SHARD1}" ] && [ -f "${SHARD2}" ]; then
    echo "[6/6] Vicuna-7b-v1.3 weights already complete at ${LLM_DIR} — skipping download."
else
    echo "[6/6] Downloading Vicuna-7b-v1.3 weights (~13 GB)..."
    mkdir -p "${LLM_DIR}"
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id='lmsys/vicuna-7b-v1.3', local_dir='${LLM_DIR}')
PY
fi

echo
echo "================================================================"
echo " Setup complete."
echo " Activate env : conda activate ${ENV_NAME}"
echo " Next steps   :"
echo "   bash prepare_frames.sh   (extract frames + build annotations)"
echo "   bash run_inference.sh    (or run infer.py directly for 10min)"
echo "================================================================"
