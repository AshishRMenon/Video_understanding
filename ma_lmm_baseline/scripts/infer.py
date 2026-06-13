"""
Run MA-LMM inference on a single duration variant of Ego4D videos.

This wraps the official LAVIS model loader (`load_model_and_preprocess`)
from the cloned MA-LMM repo. No memory instrumentation, no custom
metrics — just captioning to provide a clean baseline.

Usage:
    python scripts/infer.py \
        --frame_dir data/frames/30min \
        --ann_path  data/annotations/ego4d_30min.json \
        --output    outputs/30min.json \
        --memory_bank_length 40 \
        --num_frames 80
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent.parent / "MA-LMM"
sys.path.insert(0, str(REPO_DIR))


def load_video_frames(frame_dir: Path, num_frames: int, image_size: int = 224) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    files = sorted(frame_dir.glob("frame*.jpg"))
    if not files:
        raise FileNotFoundError(f"No frames in {frame_dir}")
    idx = np.linspace(0, len(files) - 1, num_frames, dtype=int)
    return torch.stack([transform(Image.open(files[i]).convert("RGB")) for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--ann_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--memory_bank_length", type=int, default=40)
    ap.add_argument("--num_frames", type=int, default=80)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--prompt", default="Describe what happens in this video in detail.")
    ap.add_argument("--ckpt_path", default=None,
                    help="Optional MA-LMM fine-tuned checkpoint (.pth). "
                         "Omit for zero-shot InstructBLIP+MA-LMM.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.ann_path) as f:
        anns = json.load(f)

    print(f"Loading MA-LMM model on {args.device}...")
    from lavis.models import load_model_and_preprocess
    model, _, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct_malmm",
        model_type="vicuna7b",
        is_eval=True,
        device=args.device,
    )
    model.memory_bank_length = args.memory_bank_length
    model.use_memory_bank = args.memory_bank_length > 0
    model.num_frames = args.num_frames

    # image_pe is nn.Embedding(max_num_frames=120, 1408) by default.
    # Frames beyond index 119 trigger a CUDA out-of-bounds assertion.
    # The weights are zero-initialized (positional bias starts at 0), so
    # expanding to a larger table with zeros is safe and has no effect on outputs.
    if args.num_frames > model.image_pe.num_embeddings:
        import torch.nn as nn
        new_pe = nn.Embedding(args.num_frames, model.image_pe.embedding_dim)
        nn.init.constant_(new_pe.weight, 0.0)
        model.image_pe = new_pe.to(args.device)
        print(f"  [info] image_pe expanded to {args.num_frames} positions "
              f"(was {model.image_pe.num_embeddings})")

    if args.ckpt_path:
        print(f"Loading checkpoint {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = []
    for ann in tqdm(anns, desc=f"infer({Path(args.frame_dir).name})"):
        vid = ann["video_id"]
        fdir = Path(args.frame_dir) / vid
        if not fdir.exists():
            print(f"[skip] {fdir}: missing")
            continue

        # load_video_frames returns [T, C, H, W]; model.generate expects [B, C, T, H, W]
        frames = load_video_frames(fdir, args.num_frames).permute(1, 0, 2, 3).unsqueeze(0).to(args.device)
        sample = {
            "image": frames,
            "prompt": args.prompt,   # generate() reads "prompt", not "text_input"
            "num_frames": args.num_frames,
            "is_video": True,
        }

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                sample,
                use_nucleus_sampling=False,
                num_beams=args.num_beams,
                max_length=args.max_len,
                min_length=args.min_len,
            )
        elapsed = time.time() - t0
        caption = out[0] if isinstance(out, list) else str(out)
        peak_gb = (torch.cuda.max_memory_allocated(args.device) / 1024**3) \
            if torch.cuda.is_available() else 0.0

        rec = {
            "video_id": vid,
            "n_frames_available": ann.get("n_frames"),
            "num_frames_sampled": args.num_frames,
            "memory_bank_length": args.memory_bank_length,
            "caption": caption,
            "inference_time_sec": round(elapsed, 2),
            "gpu_peak_memory_gb": round(peak_gb, 3),
        }
        results.append(rec)
        print(f"  {vid}: {caption[:140]}...")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
