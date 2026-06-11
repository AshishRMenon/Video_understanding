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
├── run_inference.sh          # baseline: runs infer.py on 10/30/60-min variants
├── run_chunked.sh            # Experiment 1: hierarchical chunk-level captioning
├── run_frame_sweep.sh        # Experiment 2: frame-budget vs quality/compute sweep
├── preflight.sh              # optional: scan /workspace/ for pre-existing assets
├── scripts/
│   ├── infer.py              # main inference driver (uses real MALMM pipeline)
│   ├── infer_chunked.py      # Experiment 1: chunk a video, caption each chunk
│   ├── plot_sweep.py         # Experiment 2: generate plots from sweep results
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

## Experiment 1 — Hierarchical chunk-level captioning

**Motivation.** MA-LMM's memory bank compresses hour-long context into 32
visual tokens, but this compression is lossy for very long videos. An
alternative strategy is to divide the video into short temporal chunks, caption
each chunk independently, then merge the chunk captions into a single
video-level summary with Vicuna.

**Two configurations for a 60-min video:**

| Configuration | Chunks | Captions before merge |
|---|---|---|
| 10-min chunks | 6 | 6 chunk captions → 1 summary |
| 1-min chunks | 60 | 60 chunk captions → 1 summary |

**How it works (`scripts/infer_chunked.py`):**
1. Sort all frame files in a video directory chronologically.
2. Divide into fixed-size temporal windows (`chunk_duration_min × video_fps` frames each).
3. Run MA-LMM on each chunk independently (same `num_frames` subsampling per chunk).
4. Feed all chunk captions as text to the already-loaded Vicuna LLM for a single-paragraph summary.

**Run 10-min chunks (default):**

```bash
conda activate malmm_baseline
bash run_chunked.sh
```

**Run 1-min chunks:**

```bash
CHUNK_MINS=1 bash run_chunked.sh
```

**Run both back-to-back:**

```bash
CHUNK_MINS=10 bash run_chunked.sh
CHUNK_MINS=1  bash run_chunked.sh
```

**Environment variable overrides:**

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_MINS` | `10` | Chunk duration in minutes |
| `VARIANT` | `60min` | Duration split to process (`10min`, `30min`, `60min`) |
| `NFRAMES` | `80` | MA-LMM frames sampled per chunk |
| `MBL` | `40` | `memory_bank_length` per chunk |
| `NO_SUMMARIZE` | _(unset)_ | Set to `1` to skip Vicuna summarization |

**Output** (`outputs/exp1_chunked/<CHUNK_MINS>min_chunks/<video_id>.json`):

```json
{
  "video_id": "12c36350-...",
  "chunk_duration_min": 10,
  "n_chunks": 6,
  "num_frames_per_chunk": 80,
  "memory_bank_length": 40,
  "total_inference_time_sec": 63.4,
  "summary": "The video shows a person working in a kitchen...",
  "chunks": [
    {
      "chunk_idx": 0,
      "start_min": 0,
      "end_min": 10,
      "n_frames_in_chunk": 6000,
      "num_frames_sampled": 80,
      "caption": "The person prepares ingredients...",
      "inference_time_sec": 10.1,
      "gpu_peak_memory_gb": 15.87
    }
  ]
}
```

---

## Experiment 2 — Frame-budget vs quality / compute trade-off

**Motivation.** The paper's baseline uses 80–100 uniformly sampled frames and
a memory bank of 40 frames. Increasing both parameters is expected to improve
caption quality, but at the cost of longer inference time and higher GPU memory.
This experiment characterizes the quality–compute Pareto frontier so we can
quantify whether the quality gains justify the added compute.

**What is swept:**

| `num_frames` | `memory_bank_length` | Notes |
|---|---|---|
| 100 | 40 | baseline (paper default) |
| 100 | 80 | larger memory, same input frames |
| 200 | 40 | 2× frames, baseline memory |
| 200 | 80 | 2× both |
| 200 | 160 | 2× frames, full memory (no compression fires) |
| 400 | 80 | 4× frames, 2× memory |
| 400 | 160 | 4× frames, 4× memory |
| 400 | 320 | 4× frames, near-full memory |

The sweep runs on the 10-min variant (fast per-video turnaround; 5 videos give
sufficient signal for mean ± std error bars). Each configuration records
per-video `inference_time_sec` and `gpu_peak_memory_gb`.

**Run the sweep:**

```bash
conda activate malmm_baseline
bash run_frame_sweep.sh
```

This runs all 8 configurations sequentially and saves one JSON per config to
`outputs/exp2_frame_sweep/`. Already-completed configs are skipped
automatically on re-run.

**Generate plots (compute metrics only):**

```bash
python scripts/plot_sweep.py \
    --sweep_dir outputs/exp2_frame_sweep \
    --plot_dir  outputs/exp2_frame_sweep/plots
```

**Generate plots with caption quality (ROUGE-L):**

```bash
python scripts/plot_sweep.py \
    --sweep_dir  outputs/exp2_frame_sweep \
    --plot_dir   outputs/exp2_frame_sweep/plots \
    --narration  /path/to/ego4d/v2/annotations/narration.json \
    --ann_path   data/annotations/ego4d_10min.json
```

**Output plots** (saved to `outputs/exp2_frame_sweep/plots/`):

| File | Contents |
|---|---|
| `inference_time.png` | Wall-clock inference time vs `num_frames`, one line per `MBL` |
| `gpu_memory.png` | GPU peak memory vs `num_frames`, one line per `MBL` |
| `caption_quality.png` | ROUGE-L vs `num_frames` (requires ground truth) |
| `quality_vs_compute.png` | Pareto scatter: ROUGE-L vs inference time per config |

`plot_sweep.py` also prints a summary table to stdout suitable for a paper
appendix.

> **GPU note.** The 400-frame configurations require ≥ 40 GB VRAM. If you
> have only 24 GB, cap the sweep at `num_frames=200` by commenting out the
> 400-frame rows in `run_frame_sweep.sh`.

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
