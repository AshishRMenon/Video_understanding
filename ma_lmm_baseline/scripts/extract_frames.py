"""
Extract frames from videos at fixed FPS, matching the official MA-LMM
preprocessing recipe in MA-LMM/data/extract_frames.py (fps=10, scale=-1:256).

Layout expected:
    data/videos/{10min,30min,60min}/<video_id>.mp4

Layout produced:
    data/frames/{10min,30min,60min}/<video_id>/frame000001.jpg ...
"""

import argparse
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def extract_one(video_path: Path, out_dir: Path, fps: int) -> tuple[Path, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", "scale=-1:256",
        "-pix_fmt", "yuvj422p",
        "-q:v", "1",
        "-r", str(fps),
        str(out_dir / "frame%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    n_frames = len(list(out_dir.glob("frame*.jpg")))
    return video_path, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", default="data/videos",
                    help="Root with subdirs per duration variant (10min/, 30min/, 60min/)")
    ap.add_argument("--frame_root", default="data/frames",
                    help="Output root mirroring video_root structure")
    ap.add_argument("--fps", type=int, default=10,
                    help="Sampling FPS. Official MA-LMM preprocess uses 10.")
    ap.add_argument("--variants", nargs="+", default=["10min", "30min", "60min"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    for variant in args.variants:
        vdir = Path(args.video_root) / variant
        fdir = Path(args.frame_root) / variant
        if not vdir.exists():
            print(f"[skip] {vdir} not found")
            continue
        for vp in sorted(vdir.iterdir()):
            if vp.suffix.lower() not in {".mp4", ".mkv", ".mov", ".avi", ".webm"}:
                continue
            out_dir = fdir / vp.stem
            if out_dir.exists() and any(out_dir.glob("frame*.jpg")):
                print(f"[skip] {out_dir} already populated")
                continue
            jobs.append((vp, out_dir))

    if not jobs:
        print("Nothing to extract.")
        return

    print(f"Extracting frames for {len(jobs)} videos at {args.fps} fps "
          f"with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, vp, od, args.fps) for vp, od in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            vp, n = fut.result()
            print(f"  {vp.name}: {n} frames")


if __name__ == "__main__":
    main()
