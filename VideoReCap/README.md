# VideoRecap Ego4D Baseline

VideoRecap baseline on the same 5 Ego4D 10-minute videos used in the MA-LMM memory saturation experiment.

## What this does

Runs the full 3-level [VideoRecap](https://arxiv.org/abs/2402.13250) hierarchy:

| Level | Model | Input | Output |
|---|---|---|---|
| 1 – Clip | `videorecap_clip.pt` | Raw pixels (4s clips) | Short caption per clip |
| 2 – Segment | `videorecap_segment.pt` | CLS features + clip captions | Description per ~3min segment |
| 3 – Video | `videorecap_video.pt` | CLS features + segment descs | Full video summary |

All inference reuses the code from `/workspace/MA_LMM_coding/VideoRecap/` (official repo).

## Dataset

Same 5 Ego4D videos (10-minute clips) as the MA-LMM experiment:

```
/workspace/data/ego4d/videos/v2/full_scale/
    12c36350-aec9-4570-8367-7163ef4f68ca_10min.mp4
    786ced75-8534-469a-be44-4e7354b1c9b8_10min.mp4
    878666b9-5f3d-476a-960f-3a5d02da7619_10min.mp4
    c26cda75-ed47-4f46-9076-25cf75bf9f1d_10min.mp4
    e0f75a35-4dde-44ef-aee7-cab38e4524f6_10min.mp4
```

## Prerequisites

- Conda env `malmm_baseline` (has torch 2.0.1+cu118, decord, transformers, timm)
- Pretrained checkpoints already at `/workspace/MA_LMM_coding/VideoRecap/pretrained_models/`:
  - `clip_openai_timesformer_base.baseline.ep_0003.pth` (LaViLa encoder, 678 MB)
  - `videorecap_clip.pt` (2.7 GB)
  - `videorecap_segment.pt` (3.0 GB)
  - `videorecap_video.pt` (3.0 GB)

## Quick Start

```bash
conda activate malmm_baseline
cd /workspace/Video_understanding/VideoReCap
bash run_vidrecap_ego4d.sh
```

Expected wall-clock on a single GPU (A100-class):
- Step 1 (feature extraction): ~5 min
- Step 2 (clip captions):       ~15–25 min  (750 clips total)
- Step 3 (segment desc):        ~2 min       (15 segments total)
- Step 4 (video summaries):     ~1 min       (5 videos total)
- Step 5 (analysis):            instant

## Running Individual Steps

```bash
bash run_vidrecap_ego4d.sh --step 1   # feature extraction only
bash run_vidrecap_ego4d.sh --step 2   # clip captions only (needs step 1 done first)
bash run_vidrecap_ego4d.sh --step 5   # analysis only (needs steps 1-4)
```

Or run scripts directly:

```bash
conda activate malmm_baseline
python3 scripts/01_extract_cls_features.py
python3 scripts/02_generate_clip_captions.py
python3 scripts/03_generate_segment_descriptions.py
python3 scripts/04_generate_video_summaries.py
python3 scripts/05_analyze_results.py
```

## Output Files

```
outputs/
├── clip_captions.pkl                     # list of (vid, start, end, caption)
├── clip_captions_summary.json            # human-readable, keyed by vid
├── segment_descriptions.pkl              # list of dicts per segment
├── segment_descriptions_summary.json
├── video_summaries.json                  # final per-video summaries (main result)
├── comparison_malmm_vs_vidrecap.json     # side-by-side vs MA-LMM
└── logs/
    ├── step1.log ... step5.log
```

## Repository Layout

```
VideoReCap/
├── README.md
├── run_vidrecap_ego4d.sh          # master pipeline
├── scripts/
│   ├── 01_extract_cls_features.py
│   ├── 02_generate_clip_captions.py
│   ├── 03_generate_segment_descriptions.py
│   ├── 04_generate_video_summaries.py
│   └── 05_analyze_results.py
└── outputs/                       # generated at runtime
```

All model code is sourced from `/workspace/MA_LMM_coding/VideoRecap/` via `sys.path`; nothing is duplicated here.
