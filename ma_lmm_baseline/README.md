# MA-LMM Baseline — Ego4D 10 / 30 / 60 min

A clean folder that **clones the official [MA-LMM](https://github.com/boheumd/MA-LMM)
repo as-is** and runs its zero-shot captioning pipeline on three duration
variants (10 / 30 / 60 minutes) of 5 hour-long Ego4D videos.

No memory-instrumentation, no diversity metric, no analysis stage —
this is the vanilla baseline.

---

## Layout

```
ma_lmm_baseline/
├── setup.sh                  # clones MA-LMM, conda env, installs deps, downloads Vicuna
├── prepare_frames.sh         # 10 FPS frame extraction + annotation JSON
├── run_inference.sh          # runs zero-shot captioning on 10/30/60 min variants
├── configs/
│   └── cap_ego4d.yaml        # minimal MA-LMM config (used by infer.py for the model)
├── scripts/
│   ├── extract_frames.py     # ffmpeg @ 10 fps (matches official preprocess recipe)
│   ├── make_annotation.py    # builds MA-LMM-style annotation JSON
│   └── infer.py              # inference driver using lavis.models.load_model_and_preprocess
├── MA-LMM/                   # cloned by setup.sh (gitignored)
└── data/
    ├── videos/{10min,30min,60min}/      # YOU put your videos here
    ├── frames/{10min,30min,60min}/      # produced by prepare_frames.sh
    └── annotations/ego4d_{10min,30min,60min}.json
```

---

## Data layout you need to provide

Put the 5 videos × 3 durations under `data/videos/`:

```
data/videos/
├── 10min/
│   ├── <video_id_1>.mp4
│   ├── <video_id_2>.mp4
│   ├── <video_id_3>.mp4
│   ├── <video_id_4>.mp4
│   └── <video_id_5>.mp4
├── 30min/
│   └── ...
└── 60min/
    └── ...
```

Filenames don't matter — whatever stem you use becomes the `video_id`.
Keeping the stems consistent across the three folders is recommended so
you can join results across durations.

---

## Quickstart

```bash
# 0) Clone this repo on the only_MALMM branch
git clone -b only_MALMM https://github.com/AshishRMenon/Video_understanding.git
cd Video_understanding/ma_lmm_baseline

# 1) one-time setup (~20 min, dominated by Vicuna-7b download)
bash setup.sh
conda activate malmm_baseline

# 2) frames + annotations
bash prepare_frames.sh

# 3) inference (zero-shot)
bash run_inference.sh
```

Tweak the inference knobs via env vars:

```bash
MBL=20 NFRAMES=120 bash run_inference.sh
# or load a fine-tuned MA-LMM checkpoint instead of zero-shot:
CKPT=/path/to/checkpoint_best.pth bash run_inference.sh
```

Captions land in `outputs/captions_{10min,30min,60min}.json`.

---

## RunPod template recommendation

**Pick: `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`** (or
the equivalent "PyTorch 2.0.1" template on the marketplace).

Why this combo:

| Constraint | Pin | Source |
|---|---|---|
| `torch >= 1.10.0` | torch 2.0.1 | `MA-LMM/requirements.txt` |
| `transformers >= 4.28.0` | 4.28.x is the lowest known good | requirements.txt |
| `fairscale == 0.4.4` | needs torch ≤ 2.0.x — breaks on 2.1+ | requirements.txt |
| `timm == 0.4.12` | very old, but layer-norm import works through torch 2.0 | requirements.txt |
| Vicuna-7b fp16 + ViT-G + Q-Former + 80-frame video | ≥ 24 GB VRAM, **40 GB+ recommended** for 60-min @ 80 frames | empirical |

**CUDA / PyTorch / Python:**
- **CUDA 11.8**, **PyTorch 2.0.1**, **Python 3.9 or 3.10** — `setup.sh`
  installs `torch==2.0.1+cu118` from the official wheel index.
- Avoid PyTorch ≥ 2.1: `fairscale==0.4.4` and `timm==0.4.12` start hitting
  `torch._six`/`Tensor.storage()` deprecations.
- Avoid CUDA 12.x base images unless you also bump PyTorch.

**GPU choice on RunPod:**
- **A100 40 GB** or **A100 80 GB** — safest. The paper trained on 4× A100.
- **A6000 (48 GB)** / **L40S (48 GB)** — fine for inference with
  `num_frames=80, memory_bank_length=40`.
- **RTX 4090 (24 GB)** — works for the 10-min variant; will OOM on the
  60-min × 80-frame setting unless you drop `num_frames` to ~40.

**Disk:** allocate ≥ 60 GB. Vicuna-7b is ~13 GB, raw Ego4D videos +
frames at 10 fps can balloon quickly (1 hr × 10 fps ≈ 36 k JPEGs per
video).

---

## What `setup.sh` does

1. `git clone https://github.com/boheumd/MA-LMM.git` into `./MA-LMM/`
2. Creates conda env `malmm_baseline` (Python 3.9)
3. Installs PyTorch 2.0.1 + CUDA 11.8 wheels, then `pip install -e MA-LMM/`
4. Downloads `lmsys/vicuna-7b-v1.5` to `MA-LMM/llm/vicuna-7b/`

If you already have Vicuna weights elsewhere, symlink them and re-run.

---

## Notes vs. the official repo

- Inference uses **zero-shot InstructBLIP + MA-LMM** by default (no
  `--ckpt_path`). For task-specific quality, pass a fine-tuned checkpoint
  from the [saved_model.tar](https://drive.google.com/file/d/1mq6fg69Ofm32-1HjEunoFtPg8ymAIcOp/view?usp=sharing)
  bundle distributed by the authors.
- Frame extraction at 10 FPS replicates `MA-LMM/data/extract_frames.py`.
- `memory_bank_length=40` and `num_frames=80` match the captioning configs
  shipped in `MA-LMM/lavis/projects/malmm/cap_youcook2.yaml`.
