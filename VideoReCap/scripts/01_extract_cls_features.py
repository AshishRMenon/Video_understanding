"""
Step 1 — Extract LaViLa CLS features from Ego4D 10-minute videos.

Runs the TimeSformer visual encoder (LaViLa dual-encoder) on every
FEATURE_STEP-second clip of each video and saves a single .npy per video:
    shape = (n_clips, 768)   where n_clips ≈ video_duration_s / feature_step
"""

import sys
import os
import argparse
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

VIDRECAP_ROOT = "/workspace/MA_LMM_coding/VideoRecap"
sys.path.insert(0, VIDRECAP_ROOT)

from src.data.video_transforms import Permute
from src.models.model_utils import remap_keys
from src.models.openai_model import QuickGELU
from src.models.openai_clip import load as load_openai_clip
from src.models.timesformer import SpaceTimeTransformer
from src.data.datasets import VideoCaptionDataset


def build_vision_model(lavila_ckpt: str, num_frames: int, device: str):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init="zeros",
        attention_style="frozen-in-time",
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=False,
    )
    clip_model, _ = load_openai_clip("ViT-B/16", "cpu")
    remapped = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    vision_model.load_state_dict(remapped, strict=False)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    for p in vision_model.parameters():
        p.requires_grad = False

    ckpt = torch.load(lavila_ckpt, map_location="cpu")
    state_dict = OrderedDict(
        {k.replace("module.visual.", ""): v
         for k, v in ckpt["state_dict"].items() if "visual" in k}
    )
    vision_model.load_state_dict(state_dict, strict=True)
    print(f"  Loaded LaViLa encoder from {lavila_ckpt}")
    vision_model.eval()
    vision_model.to(device)
    return vision_model


def extract_video_features(vid, video_dir, model, transform, feature_step,
                            num_frames, batch_size, num_workers, device):
    import decord
    video_path = os.path.join(video_dir, f"{vid}.mp4")
    vr = decord.VideoReader(video_path, num_threads=4)
    fps = vr.get_avg_fps()
    duration = len(vr) / fps

    metadata = []
    t = 0.0
    while t + feature_step <= duration:
        metadata.append([vid, t, t + feature_step])
        t += feature_step
    if t < duration:
        metadata.append([vid, t, duration])

    args_ns = argparse.Namespace(
        dataset="clip_caption",
        metadata=metadata,
        video_feature_type="pixel",
        video_feature_path=video_dir,
        video_loader_type="decord",
        chunk_len=-1,
        num_video_feat=num_frames,
        video_sampling_type="uniform",
        text_feature_type=None,
        num_text_feat=0,
        text_feature_path=None,
        text_feature_width=768,
        video_feature_width=768,
        part=None,
        unify_type=None,
        force_len=None,
        hier_type="recur",
    )

    dataset = VideoCaptionDataset(args_ns, transform=transform, is_training=False,
                                  extract_features=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    all_features = {}
    with torch.no_grad():
        for samples in tqdm(loader, desc=f"  {vid}", leave=False):
            frames = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous().to(device)
            feats = model.forward_features(frames, cls_at_last=True)
            for j in range(feats.shape[0]):
                start_sec = dataset.samples[samples["index"][j].item()][1]
                all_features[start_sec] = feats[j].cpu().numpy()

    seconds = sorted(all_features.keys())
    return np.stack([all_features[s] for s in seconds])


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    crop_size = 224
    transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(
            mean=[108.3272985, 116.7460125, 104.09373615000001],
            std=[68.5005327, 66.6321579, 70.32316305],
        ),
    ])

    model = build_vision_model(args.lavila_ckpt, args.num_frames, device)

    video_files = sorted(Path(args.video_dir).glob(args.video_glob))
    print(f"Found {len(video_files)} videos matching '{args.video_glob}' in {args.video_dir}")

    for vpath in video_files:
        vid = vpath.stem
        out_path = os.path.join(args.output_dir, f"{vid}.npy")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  [skip] {vid} (already exists, use --overwrite to redo)")
            continue

        print(f"\nExtracting: {vid}")
        try:
            feats = extract_video_features(
                vid, args.video_dir, model, transform,
                args.feature_step, args.num_frames,
                args.batch_size, args.num_workers, device,
            )
            np.save(out_path, feats)
            print(f"  Saved shape={feats.shape} → {out_path}")
        except Exception as e:
            print(f"  ERROR on {vid}: {e}")

    print("\nDone. Feature files:")
    for f in sorted(Path(args.output_dir).glob("*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: {arr.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LaViLa CLS features from videos")

    # ── paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--video_dir",    required=True,
                        help="Directory containing input .mp4 files")
    parser.add_argument("--output_dir",   required=True,
                        help="Directory to save per-video .npy feature files")
    parser.add_argument("--lavila_ckpt",  required=True,
                        help="Path to LaViLa dual-encoder checkpoint (.pth)")

    # ── data options ───────────────────────────────────────────────────────
    parser.add_argument("--video_glob",   default="*_10min.mp4",
                        help="Glob pattern to select videos (default: *_10min.mp4)")
    parser.add_argument("--feature_step", type=int, default=4,
                        help="Extract one feature every N seconds (default: 4)")
    parser.add_argument("--num_frames",   type=int, default=4,
                        help="Frames sampled per clip for the encoder (default: 4)")

    # ── compute options ────────────────────────────────────────────────────
    parser.add_argument("--batch_size",   type=int, default=16,
                        help="Clips per forward pass (default: 16)")
    parser.add_argument("--num_workers",  type=int, default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-extract even if .npy already exists")

    main(parser.parse_args())
