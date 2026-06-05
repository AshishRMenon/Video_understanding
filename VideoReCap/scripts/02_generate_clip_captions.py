"""
Step 2 — Level 1: Generate clip-level captions (one per N-second clip).

Uses the VideoRecap clip captioning model (videorecap_clip.pt) which runs the
TimeSformer directly on raw video pixels.

Output: pickle file of (vid, start_sec, end_sec, caption) tuples.
"""

import sys
import os
import argparse
import pickle
import json
import time
import numpy as np
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

VIDRECAP_ROOT = "/workspace/MA_LMM_coding/VideoRecap"
sys.path.insert(0, VIDRECAP_ROOT)

from src.data.video_transforms import Permute
from src.models.video_recap import VideoRecap
from src.data.datasets import VideoCaptionDataset, CaptionDataCollator
from transformers import AutoTokenizer


def decode_one(generated_ids, tokenizer):
    if tokenizer.eos_token_id == tokenizer.bos_token_id:
        eos_list = generated_ids[1:].tolist()
        eos_id = (eos_list.index(tokenizer.eos_token_id) + 1
                  if tokenizer.eos_token_id in eos_list
                  else len(generated_ids) - 1)
    elif tokenizer.eos_token_id in generated_ids.tolist():
        eos_id = generated_ids.tolist().index(tokenizer.eos_token_id)
    else:
        eos_id = len(generated_ids.tolist()) - 1
    return tokenizer.decode(generated_ids[1:eos_id].tolist())


def build_clip_metadata(video_dir, video_glob, clip_duration):
    import decord
    metadata = []
    for vpath in sorted(Path(video_dir).glob(video_glob)):
        vid = vpath.stem
        vr = decord.VideoReader(str(vpath), num_threads=4)
        duration = len(vr) / vr.get_avg_fps()
        t = 0.0
        while t + clip_duration <= duration:
            metadata.append((vid, t, t + clip_duration))
            t += clip_duration
        if t < duration:
            metadata.append((vid, t, duration))
    return metadata


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.clip_ckpt}")
    ckpt = torch.load(args.clip_ckpt, map_location="cpu")
    old_args = ckpt["args"]
    state_dict = OrderedDict({k.replace("module.", ""): v
                              for k, v in ckpt["state_dict"].items()})

    # Override paths / settings for our environment
    old_args.video_feature_path  = args.video_dir
    old_args.video_loader_type   = "decord"
    old_args.chunk_len           = -1          # full video, no chunking
    old_args.video_sampling_type = "uniform"
    old_args.dataset             = "clip_caption"
    old_args.video_feature_type  = "pixel"

    # Fill in fields that may be absent from older checkpoints
    for attr, default in [("finetune_mapper", False), ("query_width", 768),
                          ("vision_model_type", "timesformer"),
                          ("cross_attn_freq", 1), ("use_lora", False),
                          ("share_mapper", False), ("freeze_lm_entire", False)]:
        if not hasattr(old_args, attr):
            setattr(old_args, attr, default)

    tokenizer = AutoTokenizer.from_pretrained(old_args.decoder_name)

    crop_size = 224
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(
            mean=[108.3272985, 116.7460125, 104.09373615000001],
            std=[68.5005327, 66.6321579, 70.32316305],
        ),
    ])

    print(f"Scanning videos in {args.video_dir} (glob: {args.video_glob})")
    metadata = build_clip_metadata(args.video_dir, args.video_glob, args.clip_duration)
    print(f"  Total clips: {len(metadata)}")

    old_args.metadata = list(metadata)
    dataset = VideoCaptionDataset(old_args, transform=val_transform, is_training=False)
    collator = CaptionDataCollator(
        tokenizer,
        max_gen_tokens=old_args.max_gen_tokens,
        add_bos=True, add_eos=True, pad_token_id=0,
    )
    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=collator,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    print(f"Loading VideoRecap clip model...")
    # eval_only=True: vision encoder weights come from state_dict, not a separate path
    model = VideoRecap(old_args, eval_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}")

    t0 = time.time()
    print(f"\nGenerating clip captions ({len(dataset)} clips)...")
    with torch.no_grad():
        for samples in tqdm(loader):
            indices = samples["indices"]
            if hasattr(model, "vision_model"):
                image = samples["video_features"].permute(0, 2, 1, 3, 4).contiguous().to(device)
                samples["video_features"] = model.vision_model.forward_features(
                    image, use_checkpoint=False, cls_at_last=False)

            queries = model.map_features(samples)
            generated_ids, _ = model.generate(
                queries, tokenizer,
                do_sample=False,
                max_text_length=old_args.max_gen_tokens,
                num_return_sequences=1,
            )
            for j in range(generated_ids.shape[0]):
                caption = decode_one(generated_ids[j], tokenizer).strip()
                sample  = dataset.samples[indices[j].item()]
                dataset.samples[indices[j].item()] = list(sample) + [caption]

    results = [s for s in dataset.samples if len(s) == 4]

    # Save pickle (input for step 3)
    pkl_path = os.path.join(args.output_dir, "clip_captions.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved pickle → {pkl_path}  ({len(results)} clips, {time.time()-t0:.1f}s)")

    # Save human-readable JSON summary
    by_vid = {}
    for (vid, s, e, cap) in results:
        by_vid.setdefault(vid, []).append({"start": s, "end": e, "caption": cap})
    json_path = os.path.join(args.output_dir, "clip_captions_summary.json")
    with open(json_path, "w") as f:
        json.dump(by_vid, f, indent=2)
    print(f"Saved JSON  → {json_path}")

    for vid, clips in sorted(by_vid.items()):
        print(f"\n  [{vid}]  {len(clips)} clips")
        for c in clips[:3]:
            print(f"    [{c['start']:.0f}-{c['end']:.0f}s] {c['caption']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoRecap Level-1: clip captions")

    # ── paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--video_dir",      required=True,
                        help="Directory containing input .mp4 files")
    parser.add_argument("--clip_ckpt",      required=True,
                        help="Path to videorecap_clip.pt checkpoint")
    parser.add_argument("--output_dir",     required=True,
                        help="Directory to save clip_captions.pkl and JSON")

    # ── data options ───────────────────────────────────────────────────────
    parser.add_argument("--video_glob",     default="*_10min.mp4",
                        help="Glob to select videos (default: *_10min.mp4)")
    parser.add_argument("--clip_duration",  type=float, default=4.0,
                        help="Seconds per clip (default: 4.0)")

    # ── compute options ────────────────────────────────────────────────────
    parser.add_argument("--batch_size",     type=int, default=8,
                        help="Clips per forward pass (default: 8)")
    parser.add_argument("--num_workers",    type=int, default=4,
                        help="DataLoader workers (default: 4)")

    main(parser.parse_args())
