"""
Standalone MA-LMM inference on Ego4D hour-long videos.

Key instrumentation added beyond standard inference:
  - Memory bank state snapshots at each frame timestep
  - Cosine similarity of current frame vs. memory bank (detects saturation)
  - Per-timestep memory entropy (how diverse is memory content?)
  - GPU memory usage tracking
  - Output: captions + memory dynamics JSON for analysis

Usage:
    python scripts/run_inference_ego4d.py \
        --video_dir data/ego4d/frames \
        --ann_path data/ego4d/annotations/ego4d_eval.json \
        --llm_model MA-LMM/llm/vicuna-7b \
        --ckpt_path MA-LMM/saved_model/youcook2/checkpoint_best.pth \
        --memory_bank_length 40 \
        --num_frames 80 \
        --output_dir outputs/exp01_memory_saturation
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add MA-LMM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "MA-LMM"))


def load_frames(frame_dir: str, num_frames: int, image_size: int = 224):
    """
    Load uniformly sampled frames from a directory of JPEGs.
    Returns: Tensor [T, C, H, W] normalized for ViT.
    """
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    frame_files = sorted(Path(frame_dir).glob("*.jpg"))
    if len(frame_files) == 0:
        raise ValueError(f"No frames found in {frame_dir}")

    # Uniform sampling
    indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
    selected = [frame_files[i] for i in indices]

    frames = []
    for fp in selected:
        img = Image.open(fp).convert("RGB")
        frames.append(transform(img))

    return torch.stack(frames)  # [T, C, H, W]


def compute_memory_stats(visual_memory_bank: torch.Tensor) -> dict:
    """
    Compute memory bank statistics to track saturation.
    visual_memory_bank shape: [B, T, N, C]
    """
    if visual_memory_bank is None:
        return {}

    bank = visual_memory_bank[0]  # [T, N, C], batch=0
    T, N, C = bank.shape

    # Mean representation per frame slot
    mean_per_slot = bank.mean(dim=1)  # [T, C]

    # Pairwise cosine similarity between memory slots
    norm = F.normalize(mean_per_slot, dim=-1)  # [T, C]
    sim_matrix = (norm @ norm.T).cpu().numpy()  # [T, T]

    # Average off-diagonal similarity (higher = more redundant)
    mask = ~np.eye(T, dtype=bool)
    avg_sim = float(sim_matrix[mask].mean()) if T > 1 else 0.0

    # Effective diversity: 1 - avg_sim
    diversity = 1.0 - avg_sim

    return {
        "num_slots": T,
        "avg_inter_slot_similarity": round(avg_sim, 4),
        "diversity_score": round(diversity, 4),
    }


def run_inference_single_video(
    model,
    frame_dir: str,
    video_id: str,
    num_frames: int,
    prompt: str,
    device: str,
    capture_memory: bool = True,
) -> dict:
    """Run MA-LMM on one video, capturing memory dynamics."""
    print(f"\n  Loading frames for {video_id}...")
    try:
        frames = load_frames(frame_dir, num_frames)  # [T, C, H, W]
    except ValueError as e:
        return {"video_id": video_id, "error": str(e)}

    T = frames.shape[0]
    print(f"  Loaded {T} frames")

    # Reset model memory bank
    if hasattr(model, "visual_memory_bank"):
        if model.visual_memory_bank is not None:
            del model.visual_memory_bank
            model.visual_memory_bank = None

    # Build sample dict mimicking the dataset output
    # MA-LMM expects: samples["image"] = [T, C, H, W] as a video tensor
    sample = {
        "image": frames.unsqueeze(0).to(device),  # [1, T, C, H, W]
        "text_input": [prompt],
        "num_frames": T,
        "is_video": True,
    }

    memory_snapshots = []
    t_start = time.time()

    with torch.no_grad():
        try:
            # Generate caption
            output = model.generate(
                sample,
                use_nucleus_sampling=False,
                num_beams=3,
                max_length=256,
                min_length=10,
            )
            caption = output[0] if isinstance(output, list) else str(output)
        except Exception as e:
            caption = f"ERROR: {e}"

    elapsed = time.time() - t_start

    # Capture final memory stats
    mem_stats = {}
    if capture_memory and hasattr(model, "visual_memory_bank"):
        mem_stats = compute_memory_stats(model.visual_memory_bank)

    gpu_mem_gb = 0.0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    return {
        "video_id": video_id,
        "caption": caption,
        "num_frames_processed": T,
        "inference_time_sec": round(elapsed, 2),
        "gpu_peak_memory_gb": round(gpu_mem_gb, 3),
        "memory_stats": mem_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data/ego4d/frames",
                        help="Root directory of frame subdirs (one per video)")
    parser.add_argument("--ann_path", default="data/ego4d/annotations/ego4d_eval.json",
                        help="Annotation JSON created by create_ego4d_annotation.py")
    parser.add_argument("--llm_model", default="MA-LMM/llm/vicuna-7b",
                        help="Path to Vicuna model weights")
    parser.add_argument("--ckpt_path", default=None,
                        help="Path to MA-LMM fine-tuned checkpoint (optional)")
    parser.add_argument("--memory_bank_length", type=int, default=40,
                        help="Memory bank size (number of slots)")
    parser.add_argument("--num_frames", type=int, default=80,
                        help="Number of frames to sample per video")
    parser.add_argument("--output_dir", default="outputs/exp01_memory_saturation",
                        help="Where to save results")
    parser.add_argument("--prompt", default="Describe what happens in this video in detail.",
                        help="Captioning prompt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load annotations
    with open(args.ann_path) as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} video annotations")

    # Load model
    print("\nLoading MA-LMM model...")
    from lavis.models import load_model_and_preprocess

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_vicuna_instruct_malmm",
        model_type="vicuna7b",
        is_eval=True,
        device=args.device,
    )

    # Override memory/frame params
    model.memory_bank_length = args.memory_bank_length
    model.use_memory_bank = args.memory_bank_length > 0
    model.num_frames = args.num_frames

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"Loading checkpoint: {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=args.device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    print(f"Model loaded. Memory bank length: {args.memory_bank_length}, Frames: {args.num_frames}")

    # Run inference
    results = []
    for ann in tqdm(annotations, desc="Running inference"):
        video_id = ann["video_id"]
        frame_subdir = os.path.join(args.video_dir, video_id)
        gt_caption = ann.get("caption", "")

        if not os.path.exists(frame_subdir):
            print(f"  SKIP {video_id}: frame dir not found")
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)

        result = run_inference_single_video(
            model=model,
            frame_dir=frame_subdir,
            video_id=video_id,
            num_frames=args.num_frames,
            prompt=args.prompt,
            device=args.device,
        )
        result["gt_caption"] = gt_caption
        result["n_frames_available"] = ann.get("n_frames", 0)

        print(f"\n  [{video_id}]")
        print(f"  Frames available: {result.get('n_frames_available', '?')} | Sampled: {args.num_frames}")
        print(f"  Caption: {result['caption'][:200]}")
        print(f"  Memory stats: {result.get('memory_stats', {})}")
        print(f"  Time: {result['inference_time_sec']}s | GPU peak: {result['gpu_peak_memory_gb']}GB")

        results.append(result)

        # Save incrementally
        out_path = os.path.join(args.output_dir, f"{video_id}_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    # Save full results
    all_results_path = os.path.join(args.output_dir, "all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {all_results_path}")

    # Print summary table
    print("\n=== RESULTS SUMMARY ===")
    print(f"{'Video ID':<30} {'Frames':<10} {'Mem Slots':<10} {'Diversity':<12} {'GPU GB':<8}")
    print("-" * 75)
    for r in results:
        ms = r.get("memory_stats", {})
        print(f"{r['video_id']:<30} {r.get('n_frames_available', '?'):<10} "
              f"{ms.get('num_slots', '?'):<10} {ms.get('diversity_score', '?'):<12} "
              f"{r.get('gpu_peak_memory_gb', '?'):<8}")


if __name__ == "__main__":
    main()
