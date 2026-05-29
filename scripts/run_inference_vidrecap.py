"""
VideoRecap 3-level inference pipeline on arbitrary Ego4D videos.

Runs the full clip → segment → video hierarchy using official VideoRecap
checkpoints. Wraps the logic from VideoRecap/demo.ipynb into a batch script.

Usage:
    python3 scripts/run_inference_vidrecap.py \
        --video_dir data/ego4d/videos \
        --output_dir outputs/vidrecap \
        --vidrecap_repo VideoRecap \
        --clip_ckpt VideoRecap/pretrained_models/videorecap_clip.pt \
        --segment_ckpt VideoRecap/pretrained_models/videorecap_segment.pt \
        --video_ckpt VideoRecap/pretrained_models/videorecap_video.pt \
        --lavila_ckpt VideoRecap/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth

Output per video (saved to output_dir/<video_id>_result.json):
    {
        "video_id": "...",
        "video_caption": "...",            # final video-level summary
        "segment_captions": [...],         # one per ~3min segment
        "clip_captions": [...],            # one per 4s clip (first 20 shown)
        "n_clips": ...,
        "n_segments": ...,
        "processing_time_sec": ...
    }
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def load_vidrecap_modules(vidrecap_repo: str):
    """Add VideoRecap repo to path and import its modules."""
    repo_path = str(Path(vidrecap_repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from models.video_recapper import VideoRecapper
        from datasets.clip_caption import ClipCaptionDataset
        return VideoRecapper
    except ImportError as e:
        print(f"ERROR: Could not import VideoRecap modules from {repo_path}")
        print(f"  {e}")
        print("  Make sure VideoRecap is cloned and pip install -e VideoRecap/ was run.")
        sys.exit(1)


def load_model(ckpt_path: str, device: str):
    """Load a VideoRecap model checkpoint."""
    from models.video_recapper import VideoRecapper

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})

    model = VideoRecapper(args)
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip 'module.' prefix if saved from DDP
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, args


def load_lavila_encoder(lavila_ckpt: str, device: str):
    """Load the LaViLa TimeSformer video encoder for feature extraction."""
    try:
        from lavila.models import models as lavila_models
        import lavila.utils.distributed as dist_utils
    except ImportError:
        # LaViLa may be bundled inside VideoRecap
        try:
            from models.lavila_encoder import build_lavila_encoder
            encoder = build_lavila_encoder(lavila_ckpt, device)
            return encoder
        except ImportError:
            print("ERROR: LaViLa encoder not found. Check VideoRecap installation.")
            sys.exit(1)

    ckpt = torch.load(lavila_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Build vision-only encoder (we only need the visual side)
    model_name = ckpt["args"].model
    model = getattr(lavila_models, model_name)(
        text_use_cls_token=True,
        project_embed_dim=256,
        gated_xattn=False,
        timesformer_gated_xattn=False,
        timesformer_freeze_space=False,
        num_frames=4,
        drop_path_rate=0.0,
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.visual.to(device)
    model.eval()
    return model


def extract_video_clips(video_path: str, clip_duration: float = 4.0,
                        frames_per_clip: int = 4, image_size: int = 224):
    """
    Split video into clips and decode frames.
    Returns: list of tensors [frames_per_clip, C, H, W] and list of (start_sec, end_sec).
    """
    try:
        import decord
        decord.bridge.set_bridge("torch")
        from decord import VideoReader
    except ImportError:
        print("ERROR: decord not installed. Run: pip install decord")
        sys.exit(1)

    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    vr = VideoReader(video_path, num_threads=4)
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / fps

    clips = []
    clip_timestamps = []

    start = 0.0
    while start + clip_duration <= duration:
        end = start + clip_duration
        # Sample frames_per_clip evenly spaced within clip
        frame_indices = [
            min(int((start + (end - start) * i / (frames_per_clip - 1)) * fps), total_frames - 1)
            for i in range(frames_per_clip)
        ]
        frames = vr.get_batch(frame_indices).float() / 255.0  # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2)                   # [T, C, H, W]
        frames = torch.stack([transform(f) for f in frames])  # [T, C, H, W]
        clips.append(frames)
        clip_timestamps.append((start, end))
        start += clip_duration

    return clips, clip_timestamps, duration


def run_clip_level(video_path: str, clip_model, lavila_encoder,
                   device: str, max_new_tokens: int = 64) -> list:
    """
    Level 1: Generate one caption per 4-second clip.
    Returns list of {"start": float, "end": float, "caption": str, "cls_feature": tensor}
    """
    from transformers import AutoTokenizer

    clips, timestamps, duration = extract_video_clips(video_path)
    print(f"    {len(clips)} clips @ 4s each (video duration: {duration:.0f}s)")

    results = []
    with torch.no_grad():
        for i, (frames, (start, end)) in enumerate(zip(clips, timestamps)):
            frames = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]

            # Extract CLS feature from LaViLa encoder
            cls_feature = lavila_encoder(frames).cpu()  # [1, D]

            # Generate caption
            caption = clip_model.generate(
                video=frames,
                max_new_tokens=max_new_tokens,
            )
            if isinstance(caption, list):
                caption = caption[0]

            results.append({
                "start": start,
                "end": end,
                "caption": caption.strip(),
                "cls_feature": cls_feature,
            })

            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(clips)}] clips processed")

    return results


def run_segment_level(clip_results: list, segment_model,
                      device: str, segment_duration: float = 180.0,
                      max_new_tokens: int = 128) -> list:
    """
    Level 2: Group clips into ~3min segments, generate one description per segment.
    Returns list of {"start": float, "end": float, "description": str}
    """
    if not clip_results:
        return []

    # Group clips into segments of ~segment_duration seconds
    segments = []
    current_group = []
    seg_start = clip_results[0]["start"]

    for clip in clip_results:
        current_group.append(clip)
        if clip["end"] - seg_start >= segment_duration:
            segments.append(current_group)
            current_group = []
            if clip != clip_results[-1]:
                seg_start = clip_results[clip_results.index(clip) + 1]["start"]

    if current_group:
        segments.append(current_group)

    results = []
    with torch.no_grad():
        for seg in segments:
            start = seg[0]["start"]
            end = seg[-1]["end"]
            # CLS features: [num_clips_in_segment, D]
            cls_features = torch.cat([c["cls_feature"] for c in seg], dim=0).unsqueeze(0).to(device)
            clip_captions = " ".join([c["caption"] for c in seg])

            description = segment_model.generate(
                video=cls_features,
                prefix_text=clip_captions,
                max_new_tokens=max_new_tokens,
            )
            if isinstance(description, list):
                description = description[0]

            results.append({
                "start": start,
                "end": end,
                "description": description.strip(),
            })

    return results


def run_video_level(segment_results: list, clip_results: list,
                    video_model, device: str,
                    max_new_tokens: int = 256) -> str:
    """
    Level 3: Aggregate segment descriptions into one video-level summary.
    """
    if not segment_results:
        return ""

    # Sparse CLS features from clip results (one per segment)
    # Use one representative clip CLS per segment
    segment_cls = []
    for seg in segment_results:
        matching = [c for c in clip_results
                    if seg["start"] <= c["start"] < seg["end"]]
        if matching:
            mid_clip = matching[len(matching) // 2]
            segment_cls.append(mid_clip["cls_feature"])
        else:
            segment_cls.append(torch.zeros(1, clip_results[0]["cls_feature"].shape[-1]))

    cls_features = torch.cat(segment_cls, dim=0).unsqueeze(0).to(device)
    segment_text = " ".join([s["description"] for s in segment_results])

    with torch.no_grad():
        summary = video_model.generate(
            video=cls_features,
            prefix_text=segment_text,
            max_new_tokens=max_new_tokens,
        )
    if isinstance(summary, list):
        summary = summary[0]
    return summary.strip()


def run_single_video(video_path: str, clip_model, segment_model, video_model,
                     lavila_encoder, device: str) -> dict:
    video_id = Path(video_path).stem
    print(f"\n  Processing: {video_id}")
    t0 = time.time()

    print("    [Level 1] Clip captioning...")
    clip_results = run_clip_level(video_path, clip_model, lavila_encoder, device)

    print("    [Level 2] Segment descriptions...")
    segment_results = run_segment_level(clip_results, segment_model, device)

    print("    [Level 3] Video summary...")
    video_caption = run_video_level(segment_results, clip_results, video_model, device)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    print(f"    Video caption: {video_caption[:120]}...")

    return {
        "video_id": video_id,
        "video_caption": video_caption,
        "segment_captions": [
            {"start": s["start"], "end": s["end"], "description": s["description"]}
            for s in segment_results
        ],
        "clip_captions": [
            {"start": c["start"], "end": c["end"], "caption": c["caption"]}
            for c in clip_results[:20]  # save first 20 to keep JSON small
        ],
        "n_clips": len(clip_results),
        "n_segments": len(segment_results),
        "processing_time_sec": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data/ego4d/videos")
    parser.add_argument("--output_dir", default="outputs/vidrecap")
    parser.add_argument("--vidrecap_repo", default="VideoRecap")
    parser.add_argument("--clip_ckpt",
                        default="VideoRecap/pretrained_models/videorecap_clip.pt")
    parser.add_argument("--segment_ckpt",
                        default="VideoRecap/pretrained_models/videorecap_segment.pt")
    parser.add_argument("--video_ckpt",
                        default="VideoRecap/pretrained_models/videorecap_video.pt")
    parser.add_argument("--lavila_ckpt",
                        default="VideoRecap/pretrained_models/clip_openai_timesformer_base.baseline.ep_0003.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate checkpoints
    for ckpt_name, ckpt_path in [
        ("clip", args.clip_ckpt), ("segment", args.segment_ckpt),
        ("video", args.video_ckpt), ("lavila", args.lavila_ckpt),
    ]:
        if not Path(ckpt_path).exists():
            print(f"ERROR: {ckpt_name} checkpoint not found at {ckpt_path}")
            print("  Run: bash setup_vidrecap.sh  then download Drive checkpoints")
            sys.exit(1)

    load_vidrecap_modules(args.vidrecap_repo)

    print(f"Loading models on {args.device}...")
    lavila_encoder = load_lavila_encoder(args.lavila_ckpt, args.device)
    clip_model, _ = load_model(args.clip_ckpt, args.device)
    segment_model, _ = load_model(args.segment_ckpt, args.device)
    video_model, _ = load_model(args.video_ckpt, args.device)
    print("Models loaded.")

    video_files = sorted(Path(args.video_dir).glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {args.video_dir}")
        sys.exit(1)

    print(f"\nRunning VideoRecap on {len(video_files)} video(s)...")

    all_results = []
    for vpath in video_files:
        result = run_single_video(
            str(vpath), clip_model, segment_model, video_model,
            lavila_encoder, args.device,
        )
        out_path = Path(args.output_dir) / f"{result['video_id']}_result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

    # Save summary
    summary_path = Path(args.output_dir) / "vidrecap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("VideoRecap Inference Complete")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n[{r['video_id']}]")
        print(f"  Clips: {r['n_clips']}  Segments: {r['n_segments']}")
        print(f"  Summary: {r['video_caption'][:200]}")
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
