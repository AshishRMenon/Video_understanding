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
├── run_frame_sweep.sh        # Experiment 2: frame-budget vs all-metrics sweep
├── run_chunk_sweep.sh        # Experiment 4: chunk-size sweep on 30-min videos
├── run_comparison.sh         # chunked vs. full-video head-to-head comparison
├── preflight.sh              # optional: scan /workspace/ for pre-existing assets
├── scripts/
│   ├── infer.py              # main inference driver (uses real MALMM pipeline)
│   ├── infer_chunked.py      # Experiment 1: chunk a video, caption each chunk
│   ├── infer_v2.py           # two-stage workaround (NOT the MALMM pipeline — kept for reference)
│   ├── plot_sweep.py         # Experiment 2: combined metrics graph (Graph 1)
│   ├── plot_chunk_sweep.py   # Experiment 4: combined metrics graph (Graph 2)
│   ├── plot_comparison.py    # chunked vs. full-video comparison plot
│   ├── qa_eval.py            # QA-based caption quality evaluation (macro + micro)
│   ├── evaluate.py           # standalone ROUGE + BERTScore comparison script
│   ├── extract_frames.py     # FFmpeg @ 10 FPS
│   ├── make_annotation.py    # builds MA-LMM-style annotation JSON
│   └── preflight.py          # scans /workspace/ for assets
├── configs/
│   └── cap_ego4d.yaml        # reference MA-LMM config
└── data/
    ├── videos/{10min,30min,60min}/   # put your MP4s here
    ├── frames/{10min,30min,60min}/   # produced by prepare_frames.sh
    └── annotations/                  # produced by prepare_frames.sh
                                      # ego4d_30min.json auto-generated if missing
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

## Evaluation metrics

All experiments share four quality metrics evaluated against Ego4D ground truth.
ROUGE-L and BERTScore require only the narration file; Macro/Micro QA additionally
require an Anthropic API key (free tier is sufficient — total cost < $0.10).

| Metric | What it measures | Ground truth source |
|---|---|---|
| **ROUGE-L** | Lexical overlap between caption and GT | Segment summaries (5-min blocks) |
| **BERTScore** | Semantic similarity (roberta-large F1) | Segment summaries |
| **Macro QA** | Coverage of main themes and activity | Segment summaries → LLM questions |
| **Micro QA** | Coverage of specific details and actions | Fine-grained narrations → LLM questions |

**Ground truth path (Ego4D narration.json):**
```
/workspace/ego4d_meta/v2/annotations/narration.json
```

### How the QA metric works

The Macro QA and Micro QA scores are a custom **QAGS-style** evaluation metric
designed to test whether captions can support Video Question Answering.

**Step 1 — Question generation** (LLM call, done once per video, cached):

Claude is given the ground-truth narration and generates yes/no questions:
- **Macro questions** (8 per video) — from segment summaries. High-level:
  *"Does the video show a person weaving fabric on a loom?"*
- **Micro questions** (8 per video) — from fine-grained narrations. Specific:
  *"Does the person use a weaving pin to separate threads on the loom?"*

Questions are cached in `qa_cache.json` and reused across all configs,
so generation only runs once.

**Step 2 — Answer checking** (LLM call, once per video per config):

Claude is given the predicted caption and all 16 questions and answers YES/NO
for each. This requires semantic understanding that keyword matching cannot provide —
a caption saying *"working on a loom"* correctly answers *"weaving fabric?"* even
without exact word overlap.

**Step 3 — Scoring** (no API, just arithmetic):

```
Macro QA score = answered macro questions / 8   (per video, then averaged)
Micro QA score = answered micro questions / 8
Overall QA     = average(Macro QA, Micro QA)
```

**API cost estimate** — using `claude-haiku-4-5-20251001` (default):

| Step | Calls |
|---|---|
| Question generation (5 videos, cached) | 5 |
| Answer checking — frame sweep (4 configs × 5 videos) | 20 |
| Answer checking — chunk sweep (3 sizes × 5 videos) | 15 |
| **Total** | **~40 calls, < $0.10** |

**Run QA evaluation standalone:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python scripts/qa_eval.py \
    --captions_json outputs/captions_10min.json \
    --narration     /workspace/ego4d_meta/v2/annotations/narration.json \
    --ann_path      data/annotations/ego4d_10min.json
```

Use `--model claude-sonnet-4-6` for higher-accuracy evaluation on
borderline answers.

---

## Experiment 2 — Frame-budget vs. all metrics (Graph 1)

**Motivation.** Both `num_frames` and `memory_bank_length` are scaled together
to test whether more visual context improves caption quality across all four
metrics, and whether the quality gain justifies the additional compute.

**Configs swept** (10-min variant, 5 videos):

| `num_frames` | `memory_bank_length` | Scale |
|---|---|---|
| 100 | 40 | 1× baseline |
| 200 | 80 | 2× |
| 500 | 150 | 5× |
| 1000 | 300 | 10× |

**Full pipeline** (inference + QA eval + plot):

```bash
conda activate malmm_baseline

NARRATION_PATH=/workspace/ego4d_meta/v2/annotations/narration.json \
ANTHROPIC_API_KEY=sk-ant-... \
bash run_frame_sweep.sh
```

Without the API key, ROUGE-L and BERTScore are still computed; QA scores are skipped.

**Output — Graph 1** (`outputs/exp2_frame_sweep/plots/metrics_vs_frames.png`):

A single graph with four metric lines on the same axes:

```
Y-axis: Score (0–1)
X-axis: (num_frames, memory_bank_length) config label

Lines:  ● ROUGE-L    ■ BERTScore    ▲ Macro QA    ◆ Micro QA
```

A summary table is also printed to stdout.

> **GPU note.** The 1000-frame config requires ≥ 40 GB VRAM. Use
> `--no_bertscore` flag to skip the BERTScore model download on first run.

---

## Experiment 4 — Chunk-size sweep on 30-min videos (Graph 2)

**Motivation.** Chunked captioning quality depends heavily on chunk size:
too short and each chunk lacks context; too long and the memory bank must
compress more aggressively. This experiment sweeps chunk sizes on 30-min
videos and evaluates all four metrics to find the optimal granularity.

**Chunk sizes compared** (30-min variant, 5 videos):

| Chunk size | Chunks per video | Frames per chunk |
|---|---|---|
| 2 min | 15 | 80 (subsampled) |
| 5 min | 6 | 80 |
| 10 min | 3 | 80 |

Each chunk is captioned independently with MA-LMM (`num_frames=80, MBL=40`),
then Vicuna merges all chunk captions into one video-level summary. All four
metrics are computed on the final summary.

**Full pipeline:**

```bash
conda activate malmm_baseline

NARRATION_PATH=/workspace/ego4d_meta/v2/annotations/narration.json \
ANTHROPIC_API_KEY=sk-ant-... \
bash run_chunk_sweep.sh
```

**Environment variable overrides:**

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_SIZES` | `"2 5 10"` | Space-separated chunk durations in minutes |
| `NFRAMES_PER_CHUNK` | `80` | MA-LMM frames sampled per chunk |
| `MBL_PER_CHUNK` | `40` | `memory_bank_length` per chunk |
| `QA_MODEL` | `claude-haiku-4-5-20251001` | LLM for QA evaluation |

**Output — Graph 2** (`outputs/exp4_chunk_sweep/plots/chunk_sweep.png`):

A single graph with four metric lines on the same axes:

```
Y-axis: Score (0–1)
X-axis: Chunk size (2 min / 5 min / 10 min)

Lines:  ● ROUGE-L    ■ BERTScore    ▲ Macro QA    ◆ Micro QA
```

Questions are shared via a single `qa_cache.json` across all chunk sizes,
so question generation (5 API calls) only happens once.

**Expected finding:** Micro QA should be highest for smaller chunks
(each chunk gets dense per-segment coverage) while Macro QA may plateau
or improve with larger chunks (more context for the overall summary).
This gap is the baseline for future models to beat.

---

## Experiment 3 — Chunked vs. full-video head-to-head comparison

**Motivation.** Graph 1 shows how full-video captioning scales with frame budget.
Graph 2 shows how chunked captioning scales with chunk size. This experiment
directly compares the best of each approach on the same set of 10-min videos,
answering: *does hierarchical chunking actually beat end-to-end MA-LMM?*

**What is compared:**

| Approach | Config |
|---|---|
| Chunked | `infer_chunked.py`, 2-min chunks, `num_frames=80, MBL=40` per chunk |
| Full-video | `infer.py`, best frame budget from Experiment 2 (`num_frames=200, MBL=80`) |

Both approaches are evaluated with all four metrics (ROUGE-L, BERTScore,
Macro QA, Micro QA). The `qa_cache.json` from Experiment 2 is reused, so
no additional question-generation API calls are needed.

**Run:**

```bash
conda activate malmm_baseline

NARRATION_PATH=/workspace/ego4d_meta/v2/annotations/narration.json \
ANTHROPIC_API_KEY=sk-ant-... \
bash run_comparison.sh
```

**Environment variable overrides:**

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_MINS` | `2` | Chunk duration for the chunked approach |
| `BEST_NF` | `200` | `num_frames` for the full-video approach |
| `BEST_MBL` | `80` | `memory_bank_length` for the full-video approach |

**Output** (`outputs/exp3_comparison/plots/`):
- `comparison.png` — grouped bar chart: 4 bars per video (chunked-macro,
  chunked-micro, full-macro, full-micro), with a printed winner table and
  per-metric deltas.

---

## Bugs found and fixed vs. the original infer.py

Three bugs were present that either produced silent wrong results or crashed
with a CUDA assertion:

1. **Frame tensor dimension order** — `load_video_frames` returns `[T, C, H, W]`
   but `model.generate` expects `[B, C, T, H, W]`. Fixed with
   `.permute(1, 0, 2, 3).unsqueeze(0)` instead of `.unsqueeze(0)` alone.

2. **Wrong sample dict key** — the generate call was passing
   `"text_input": [prompt]` but the MALMM model reads `"prompt"` (a bare
   string). Fixed by using `"prompt": args.prompt`.

3. **`image_pe` positional embedding overflow** — `nn.Embedding(max_num_frames=120, 1408)`
   is hardcoded in `blip2_vicuna_instruct.py`. Any `num_frames > 120` triggers:
   ```
   Assertion `srcIndex < srcSelectDimSize` failed.
   RuntimeError: CUDA error: device-side assert triggered
   ```
   The embedding is zero-initialized (a learnable bias that starts at zero),
   so expanding it to a larger table with zeros is safe and has no effect on
   outputs. Both `infer.py` and `infer_chunked.py` now patch the embedding
   after model load if `num_frames > model.image_pe.num_embeddings`.

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
| `anthropic` | latest | Anthropic Python SDK for QA evaluation calls |
| `bert-score` | latest | BERTScore metric; downloads roberta-large (~500 MB) on first run |
| `rouge-score` | latest | ROUGE-L computation; installed automatically by plot scripts |
