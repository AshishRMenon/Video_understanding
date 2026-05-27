"""
Frame extractor for Ego4D hour-long videos.
Extracts at 1 FPS (not 10 FPS) to keep frame count manageable for hour-long videos.
A 60-min video at 1 FPS = 3600 frames vs 36000 frames at 10 FPS.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(video_path)
        ],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            return float(stream.get("duration", 0))
    return 0.0


def extract_frames(video_path: str, out_dir: str, fps: float = 1.0, short_side: int = 256):
    """Extract frames from video at given FPS, resize short side."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"scale='if(gt(iw,ih),-1,{short_side})':'if(gt(iw,ih),{short_side},-1)',fps={fps}",
        "-q:v", "2",
        "-y",
        os.path.join(out_dir, "frame%06d.jpg")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return 0
    frames = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data/ego4d/videos",
                        help="Directory containing Ego4D video files")
    parser.add_argument("--out_dir", default="data/ego4d/frames",
                        help="Output directory for extracted frames")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1.0 for hour-long videos)")
    parser.add_argument("--short_side", type=int, default=256,
                        help="Resize short side to this value")
    parser.add_argument("--min_duration", type=float, default=1800.0,
                        help="Only process videos longer than this (seconds). Default: 1800 = 30min")
    parser.add_argument("--video_list", default=None,
                        help="Optional: path to a text file with specific video filenames to process")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)

    # Collect video files
    if args.video_list:
        with open(args.video_list) as f:
            video_files = [video_dir / line.strip() for line in f if line.strip()]
    else:
        video_files = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.mkv"))

    print(f"Found {len(video_files)} video files")

    summary = []
    for vpath in tqdm(video_files, desc="Extracting frames"):
        vid_id = vpath.stem
        duration = get_video_duration(str(vpath))

        if duration < args.min_duration:
            print(f"  SKIP {vid_id}: duration {duration:.0f}s < {args.min_duration:.0f}s")
            continue

        frame_out = out_dir / vid_id
        if frame_out.exists() and len(list(frame_out.glob("*.jpg"))) > 0:
            n_frames = len(list(frame_out.glob("*.jpg")))
            print(f"  SKIP {vid_id}: already extracted ({n_frames} frames)")
        else:
            print(f"  Processing {vid_id}: {duration:.0f}s ({duration/60:.1f}min)")
            n_frames = extract_frames(str(vpath), str(frame_out), args.fps, args.short_side)
            print(f"    Extracted {n_frames} frames")

        summary.append({
            "video_id": vid_id,
            "duration_sec": round(duration, 1),
            "duration_min": round(duration / 60, 1),
            "n_frames_extracted": len(list(frame_out.glob("*.jpg"))),
            "fps": args.fps,
            "frame_dir": str(frame_out)
        })

    # Save summary
    summary_path = out_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Processed {len(summary)} videos")
    for s in summary:
        print(f"  {s['video_id']}: {s['duration_min']:.1f}min → {s['n_frames_extracted']} frames @ {s['fps']}fps")


if __name__ == "__main__":
    main()
