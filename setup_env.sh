#!/bin/bash
# Environment setup for MA-LMM + Ego4D experiments
set -e

echo "=== Setting up MA-LMM experiment environment ==="

# 1. Create conda env
conda create -n malmm python=3.9 -y
conda activate malmm

# 2. Install MA-LMM
cd MA-LMM
pip install -e .
cd ..

# 3. Install additional experiment deps
pip install ffmpeg-python tqdm pandas matplotlib seaborn jupyter

# 4. Check GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

echo ""
echo "=== Next steps ==="
echo "1. Download Vicuna-7b weights:"
echo "   See: https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md"
echo "   Place at: MA-LMM/llm/vicuna-7b"
echo ""
echo "2. (Optional) Download fine-tuned checkpoints:"
echo "   https://drive.google.com/file/d/1mq6fg69Ofm32-1HjEunoFtPg8ymAIcOp/view"
echo "   Extract to: MA-LMM/saved_model/"
echo ""
echo "3. Get Ego4D access:"
echo "   https://ego4d-data.org/docs/start-here/#cli-download"
echo "   Then run: python scripts/download_ego4d_videos.py"
