"""
Build a minimal MA-LMM-style annotation JSON per duration variant.
One JSON entry per video, recording the frame count actually on disk
(this matters because the official README notes ffmpeg-induced length drift).
"""

import argparse
import json
from pathlib import Path


def build(frame_dir: Path) -> list[dict]:
    entries = []
    for video_dir in sorted(p for p in frame_dir.iterdir() if p.is_dir()):
        n_frames = len(list(video_dir.glob("frame*.jpg")))
        if n_frames == 0:
            continue
        entries.append({
            "video_id": video_dir.name,
            "n_frames": n_frames,
            "caption": "",  # no GT — we are running zero-shot inference
        })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_root", default="data/frames")
    ap.add_argument("--ann_root", default="data/annotations")
    ap.add_argument("--variants", nargs="+", default=["10min", "30min", "60min"])
    args = ap.parse_args()

    Path(args.ann_root).mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        fdir = Path(args.frame_root) / variant
        if not fdir.exists():
            print(f"[skip] {fdir} not found")
            continue
        entries = build(fdir)
        out = Path(args.ann_root) / f"ego4d_{variant}.json"
        out.write_text(json.dumps(entries, indent=2))
        print(f"Wrote {len(entries)} entries to {out}")


if __name__ == "__main__":
    main()
