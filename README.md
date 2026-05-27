# MA-LMM Memory Saturation — Exp-01: Ego4D Hour-Long Videos

Experiment infrastructure for **Exp-01**: running MA-LMM inference on hour-long egocentric videos to demonstrate memory saturation in memory-augmented captioning approaches.

This is part of a broader research effort arguing that memory-based methods (MA-LMM) and hierarchical methods (VidRecap) both fail on long sparse video, motivating efficient one-shot attention accumulation across frames.

---

## Research Hypothesis

On hour-long Ego4D videos:
1. MA-LMM's memory bank saturates — early context is overwritten
2. Sparse video (repeated walking, commuting) causes redundant memory tokens via cosine-similarity merging
3. Generated captions are biased toward late-video events
4. Caption quality degrades vs. short-clip baselines

The memory compression mechanism (`memory_bank_compress` in `blip2.py`) merges the most cosine-similar adjacent memory slots. On sparse video, near-identical frames are repeatedly merged, progressively diluting early unique event representations until they disappear.

---

## Repository Structure

```
MA_LMM_coding/
├── configs/
│   └── cap_ego4d.yaml          # Custom MA-LMM config for Ego4D inference
├── scripts/
│   ├── download_ego4d_runpod.sh     # Download 5 long Ego4D videos
│   ├── extract_frames_ego4d.py      # 1 FPS frame extraction via ffmpeg
│   ├── create_ego4d_annotation.py   # Build MA-LMM-compatible annotation JSON
│   ├── run_inference_ego4d.py       # Standalone inference + memory monitoring
│   └── analyze_memory_saturation.py # Report: diversity scores, captions, LaTeX table
├── run_experiment.sh           # Master pipeline (steps 1–4)
├── setup_env.sh                # Conda environment setup
└── README.md
```

Data, outputs, model weights, and the MA-LMM repo itself are excluded from version control (see `.gitignore`).

---

## Prerequisites

| Dependency | Where to get |
|---|---|
| MA-LMM repo | `git clone https://github.com/boheumd/MA-LMM.git` |
| Vicuna-7b weights | [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) → place at `MA-LMM/llm/vicuna-7b/` |
| Ego4D license | [ego4d-data.org](https://ego4d-data.org/docs/start-here/#cli-download) |
| Ego4D videos | Download via `scripts/download_ego4d_runpod.sh` → `data/ego4d/videos/` |

---

## Setup

### 1. Clone this repo and MA-LMM

```bash
git clone https://github.com/AshishRMenon/Video_understanding.git -b Video_captioning_exploration
cd Video_understanding

# Clone MA-LMM into the expected location
git clone https://github.com/boheumd/MA-LMM.git MA-LMM
```

### 2. Create the conda environment

```bash
bash setup_env.sh
conda activate malmm
```

### 3. Copy custom config into MA-LMM

```bash
cp configs/cap_ego4d.yaml MA-LMM/lavis/projects/malmm/cap_ego4d.yaml
```

### 4. Download Vicuna-7b weights

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lmsys/vicuna-7b-v1.5',
    local_dir='MA-LMM/llm/vicuna-7b',
)
"
```

---

## Downloading Ego4D Videos

After signing the Ego4D license, you receive AWS credentials. Export them and run the download script:

```bash
export EGO4D_AWS_KEY="your_access_key_id"
export EGO4D_AWS_SECRET="your_secret_access_key"

bash scripts/download_ego4d_runpod.sh
```

The script:
- Downloads the metadata JSON (fast)
- Selects 5 videos with varied durations (2 × ≥90min, 3 × ~60min) for experimental diversity
- Downloads them to `data/ego4d/videos/`
- Cleans up AWS credentials from disk automatically on exit

Override defaults with environment variables:
```bash
OUTPUT_DIR=/my/custom/path MIN_DURATION_SEC=2700 bash scripts/download_ego4d_runpod.sh
```

---

## Running the Experiment

```bash
# From the repo root (with conda env active)
bash run_experiment.sh
```

Pipeline steps:
1. **Prerequisite check** — verifies Vicuna weights and video files exist
2. **Frame extraction** — 1 FPS via ffmpeg (→ `data/ego4d/frames/`)
3. **Annotation creation** — builds MA-LMM-compatible JSON (→ `data/ego4d/annotations/`)
4. **MA-LMM inference** — standalone script, records memory diversity stats per video
5. **Analysis** — prints saturation report, generates LaTeX table

Results land in `outputs/exp01_memory_saturation/`.

### RunPod tip

Use `tmux` so inference keeps running if your SSH session drops:
```bash
tmux new -s malmm
bash run_experiment.sh 2>&1 | tee outputs/run_log.txt
# Ctrl+B D to detach
```

Monitor GPU memory during first video:
```bash
watch -n 2 nvidia-smi
```

---

## Key Config Parameters (`configs/cap_ego4d.yaml`)

| Parameter | Value | Notes |
|---|---|---|
| `memory_bank_length` | 40 | Fixed memory slots; on 1hr video each slot ≈ 90s of content |
| `num_frames` | 80 | Frames sampled from full video (1 frame per 45s for 1hr video) |
| `max_num_frames` | 200 | Positional embedding limit (extended from default 120) |
| `batch_size_eval` | 1 | One video at a time |

Ablation: vary `memory_bank_length` over {10, 20, 40, 80} to show degradation curve.

---

## Memory Saturation Metric

The inference script computes a **diversity score** from the final memory bank state:

```
diversity = 1 - mean(cosine_similarity between all memory slot pairs)
```

- Score near **1.0** → memory slots represent distinct visual content
- Score near **0.0** → memory has collapsed to near-identical representations (saturated)

A diversity score below 0.3 on a long sparse video is direct evidence supporting the hypothesis.

---

## Expected Outputs

```
outputs/exp01_memory_saturation/
├── <video_id>_result.json      # Per-video: caption, memory stats, GPU usage
└── analysis/
    ├── saturation_analysis.json
    └── saturation_table.tex    # LaTeX table for paper
```

Each result JSON contains:
- `caption` — generated caption
- `memory_stats.diversity_score` — memory saturation indicator
- `memory_stats.avg_inter_slot_similarity`
- `n_frames_available` / `num_frames_processed`
- `gpu_peak_memory_gb`

---

## References

- [MA-LMM (CVPR 2024)](https://boheumd.github.io/MA-LMM/) — Memory-Augmented Large Multimodal Model
- [VidRecap](https://sites.google.com/view/vidrecap) — Hierarchical video captioning baseline
- [Ego4D Dataset](https://ego4d-data.org/) — Egocentric video dataset with hour-long recordings
- [Ego4D-HCap](https://ego4d-data.org/) — Long-range video summary annotations
