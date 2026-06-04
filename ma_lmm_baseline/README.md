# MA-LMM Baseline — Ego4D 10 / 30 / 60 min

Zero-shot captioning of long Ego4D videos using the official
[MA-LMM](https://github.com/boheumd/MA-LMM) pipeline — EVA-CLIP-G vision
encoder → Q-Former with memory bank → Vicuna-7b-v1.3 LLM. No fine-tuning,
no workarounds.

---

## How MA-LMM works (briefly)

MA-LMM processes video frames sequentially through a Q-Former that maintains
a **memory bank** compressing visual context across time. The accumulated
visual embeddings are projected as prefix tokens into Vicuna-7b, which
generates a single coherent caption. This is end-to-end vision-language —
Vicuna receives visual tokens, not text.

---

## Repository layout

```
ma_lmm_baseline/
├── setup.sh                  # one-time bootstrap (conda, deps, Vicuna download)
├── prepare_frames.sh         # FFmpeg frame extraction + annotation JSON
├── run_inference.sh          # runs infer.py on 10/30/60-min variants
├── preflight.sh              # optional: scan /workspace/ for pre-existing assets
├── scripts/
│   ├── infer.py              # main inference driver (uses real MALMM pipeline)
│   ├── infer_v2.py           # two-stage workaround (NOT the MALMM pipeline — kept for reference)
│   ├── extract_frames.py     # FFmpeg @ 10 FPS
│   ├── make_annotation.py    # builds MA-LMM-style annotation JSON
│   └── preflight.py          # scans /workspace/ for assets
├── configs/
│   └── cap_ego4d.yaml        # reference MA-LMM config
└── data/
    ├── videos/{10min,30min,60min}/   # put your MP4s here
    ├── frames/{10min,30min,60min}/   # produced by prepare_frames.sh
    └── annotations/                  # produced by prepare_frames.sh
```

`MA-LMM/` (cloned by `setup.sh`) and `data/` are gitignored — they contain
large binaries.

---

## Prerequisites

| Requirement | Value |
|---|---|
| GPU VRAM | ≥ 24 GB (40 GB+ recommended for 60-min @ 80 frames) |
| CUDA | 11.8 |
| Python | 3.9 or 3.10 |
| Disk | ≥ 60 GB (Vicuna ~13 GB + frames) |

**Recommended RunPod template:**
`runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`

Avoid PyTorch ≥ 2.1 — `fairscale==0.4.4` and `timm==0.4.12` hit breaking
API changes (`torch._six`, `Tensor.storage()`) on newer versions.

---

## Step-by-step setup

### 1. Clone this repo (only_MALMM branch)

```bash
git clone -b only_MALMM https://github.com/AshishRMenon/Video_understanding.git
cd Video_understanding/ma_lmm_baseline
```

### 2. (Optional) Preflight scan

If you are on a pod that may already have Vicuna weights or videos, run this
first. It scans `/workspace/` and prints symlink suggestions so you don't
re-download what's already there.

```bash
WORKSPACE=/workspace bash preflight.sh
# Then follow the suggestions, e.g.:
#   mkdir -p MA-LMM/llm
#   ln -s /workspace/existing_vicuna  MA-LMM/llm/vicuna-7b-v1.3
#   ln -s /workspace/videos/10min     data/videos/10min
```

### 3. Run setup.sh

```bash
bash setup.sh
conda activate malmm_baseline
```

This script (idempotent — safe to re-run):
1. Installs `ffmpeg` if missing
2. Installs Miniconda if missing
3. Clones the official MA-LMM repo into `MA-LMM/`
4. Creates conda env `malmm_baseline` (Python 3.9)
5. Installs PyTorch 2.0.1 + CUDA 11.8, then `pip install -e MA-LMM/`
6. Downloads **Vicuna-7b-v1.3** (~13 GB) to `MA-LMM/llm/vicuna-7b-v1.3/`

> **Why v1.3 specifically?** The pretrained Q-Former checkpoint
> (`instruct_blip_vicuna7b_trimmed.pth`) has its projection layer trained
> against Vicuna v1.3's embedding space. Using v1.5 causes degenerate output
> (captions filled with `OOOOO...` or `1. 1. 1. 1.`).

### 4. Verify the LAVIS config points to v1.3

After cloning, confirm this line in
`MA-LMM/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml`:

```yaml
llm_model: "/workspace/Video_understanding/ma_lmm_baseline/MA-LMM/llm/vicuna-7b-v1.3"
```

If it points anywhere else, update it to match your actual path to the v1.3
weights. This is the single most common cause of garbled output.

### 5. Put your videos in place

```
data/videos/
├── 10min/  ← 5 × ~10-min Ego4D clips (.mp4)
├── 30min/  ← 5 × ~30-min clips
└── 60min/  ← 5 × ~60-min clips
```

Filenames become the `video_id` in outputs. You can symlink instead of copying.

### 6. Extract frames and build annotations

```bash
bash prepare_frames.sh
```

Extracts at 10 FPS (`scale=-1:256`) and writes
`data/annotations/ego4d_{10min,30min,60min}.json`.

### 7. Run inference

```bash
bash run_inference.sh
```

Or target a single duration directly:

```bash
conda activate malmm_baseline
python scripts/infer.py \
    --frame_dir data/frames/10min \
    --ann_path  data/annotations/ego4d_10min.json \
    --output    outputs/captions_10min.json \
    --memory_bank_length 40 \
    --num_frames 80
```

Tune via env vars:

```bash
MBL=20 NFRAMES=120 bash run_inference.sh
# Fine-tuned checkpoint (optional):
CKPT=/path/to/checkpoint_best.pth bash run_inference.sh
```

---

## Expected output format

```json
[
  {
    "video_id": "12c36350-aec9-4570-8367-7163ef4f68ca",
    "n_frames_available": 4605,
    "num_frames_sampled": 100,
    "memory_bank_length": 40,
    "caption": "The video captures a close-up view of a person's hands working on a loom...",
    "inference_time_sec": 10.1,
    "gpu_peak_memory_gb": 15.873
  }
]
```

Captions land in `outputs/captions_{10min,30min,60min}.json`.

---

## Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `--memory_bank_length` | 40 | Compress Q-Former memory after this many frames |
| `--num_frames` | 80 | Uniformly sampled frames per video |
| `--num_beams` | 5 | Beam search width |
| `--max_len` | 256 | Max caption tokens |
| `--prompt` | "Describe what happens in this video in detail." | Instruction prefix |

---

## Bugs found and fixed vs. the original infer.py

Two bugs were present in the original `scripts/infer.py` that produced
silent wrong results:

1. **Frame tensor dimension order** — `load_video_frames` returns `[T, C, H, W]`
   but `model.generate` expects `[B, C, T, H, W]`. Fixed with
   `.permute(1, 0, 2, 3).unsqueeze(0)` instead of `.unsqueeze(0)` alone.

2. **Wrong sample dict key** — the generate call was passing
   `"text_input": [prompt]` but the MALMM model reads `"prompt"` (a bare
   string). Fixed by using `"prompt": args.prompt`.

---

## About infer_v2.py

`scripts/infer_v2.py` is kept for reference but is **not** the MALMM pipeline.
It uses a two-stage workaround:

- Stage 1: InstructBLIP-FlanT5-XL captions each frame independently
- Stage 2: Vicuna-7b receives those text descriptions and summarizes them

Vicuna never sees any video frames in this approach — it is a pure text
summarizer. Results look like English but do not reflect the MALMM memory
bank architecture. Use `infer.py` for actual MALMM results.

---

## Dependency pins

| Package | Pin | Reason |
|---|---|---|
| `torch` | 2.0.1+cu118 | `fairscale==0.4.4` breaks on ≥ 2.1 |
| `fairscale` | 0.4.4 | Uses deprecated `torch._six` / `Tensor.storage()` |
| `timm` | 0.4.12 | Required by LAVIS EVA-CLIP loader |
| `transformers` | ≥4.28.0, <4.46.0 | LLaMA tokenizer compatibility window |
| `spacy` | <3.8.0 | Conflicts in newer versions with LAVIS deps |
| Vicuna | v1.3 | Must match the Q-Former pretrained checkpoint |
