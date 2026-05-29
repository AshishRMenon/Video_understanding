"""
Download 5 long Ego4D videos + their narration annotations.

Steps:
  1. Download ego4d metadata JSON
  2. Pick 5 videos with varied long durations
  3. Download those 5 videos via ego4d CLI
  4. Download annotations and save a filtered JSON for just those 5 videos

Usage:
    python3 scripts/download_ego4d_5videos.py \
        --output_dir data/ego4d \
        --min_duration 3600

Credentials must be in ~/.aws/credentials (standard AWS format).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list, desc: str = ""):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"  ERROR: command failed (exit {result.returncode})")
        sys.exit(1)


def download_metadata(output_dir: Path) -> Path:
    meta_path = output_dir / "v2" / "ego4d.json"
    if meta_path.exists():
        print(f"  [skip] metadata already at {meta_path}")
        return meta_path

    print("  Downloading Ego4D metadata (~67MB)...")
    run(["ego4d",
         "--output_directory", str(output_dir),
         "--datasets", "annotations",
         "--version", "v2"], desc="metadata")

    # ego4d CLI puts it under output_dir/v2/ego4d.json
    if not meta_path.exists():
        # Search for it
        candidates = list(output_dir.rglob("ego4d.json"))
        if not candidates:
            print("ERROR: ego4d.json not found after download")
            sys.exit(1)
        meta_path = candidates[0]
    return meta_path


def select_videos(meta_path: Path, min_duration: int, n: int = 5) -> list:
    with open(meta_path) as f:
        d = json.load(f)

    long_videos = [
        {"uid": v["video_uid"],
         "duration_sec": v.get("duration_sec", 0),
         "duration_min": round(v.get("duration_sec", 0) / 60, 1)}
        for v in d["videos"]
        if v.get("duration_sec", 0) >= min_duration
    ]
    long_videos.sort(key=lambda x: -x["duration_sec"])

    if len(long_videos) < n:
        print(f"  WARNING: only {len(long_videos)} videos >= {min_duration}s, taking all")
        selected = long_videos
    else:
        # Pick diverse durations: spread across the sorted list
        step = len(long_videos) // n
        selected = [long_videos[i * step] for i in range(n)]

    print(f"\n  Selected {len(selected)} videos:")
    print(f"  {'UID':<40} {'Duration'}")
    print(f"  {'-'*55}")
    for v in selected:
        print(f"  {v['uid']}  {v['duration_min']} min")
    return selected


def download_videos(selected: list, video_dir: Path):
    video_dir.mkdir(parents=True, exist_ok=True)
    uids = [v["uid"] for v in selected]

    # Check which already exist
    missing = [uid for uid in uids if not (video_dir / f"{uid}.mp4").exists()]
    if not missing:
        print("  [skip] all videos already downloaded")
        return

    print(f"  Downloading {len(missing)} video(s)...")
    run(["ego4d",
         "--output_directory", str(video_dir.parent),
         "--datasets", "full_scale",
         "--video_uids"] + missing)


def download_and_filter_annotations(selected: list, output_dir: Path) -> Path:
    """
    Download full annotations package and extract narrations for our 5 videos.
    Saves a filtered JSON: {video_uid: {narrations: [...], duration_sec: ...}}
    """
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = ann_dir / "ego4d_5videos_annotations.json"
    if filtered_path.exists():
        print(f"  [skip] filtered annotations already at {filtered_path}")
        return filtered_path

    # Download full annotations (narrations, etc.)
    print("  Downloading full annotations package...")
    run(["ego4d",
         "--output_directory", str(output_dir),
         "--datasets", "annotations",
         "--version", "v2"])

    # Find narration annotation file
    narration_file = None
    for candidate in output_dir.rglob("narration.json"):
        narration_file = candidate
        break

    our_uids = {v["uid"] for v in selected}
    uid_to_info = {v["uid"]: v for v in selected}
    filtered = {}

    if narration_file and narration_file.exists():
        print(f"  Parsing narrations from {narration_file} ...")
        with open(narration_file) as f:
            narr_data = json.load(f)

        # Ego4D narration format: {"videos": [{video_uid, narration_pass_1: {narrations:[...]}, ...}]}
        videos_list = narr_data.get("videos", []) if isinstance(narr_data, dict) else narr_data
        for entry in videos_list:
            uid = entry.get("video_uid", "")
            if uid in our_uids:
                narrations = []
                for pass_key in ["narration_pass_1", "narration_pass_2"]:
                    for n in entry.get(pass_key, {}).get("narrations", []):
                        narrations.append({
                            "timestamp_sec": n.get("timestamp_sec"),
                            "narration_text": n.get("narration_text", ""),
                        })
                narrations.sort(key=lambda x: x.get("timestamp_sec") or 0)
                filtered[uid] = {
                    "duration_min": uid_to_info[uid]["duration_min"],
                    "duration_sec": uid_to_info[uid]["duration_sec"],
                    "narration_count": len(narrations),
                    "narrations": narrations,
                }
    else:
        print("  WARNING: narration.json not found — saving video metadata only")
        for v in selected:
            filtered[v["uid"]] = {"duration_min": v["duration_min"], "duration_sec": v["duration_sec"]}

    with open(filtered_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"  Saved filtered annotations → {filtered_path}")
    for uid, info in filtered.items():
        print(f"    {uid}  {info.get('narration_count', 0)} narrations")
    return filtered_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/ego4d",
                        help="Root output directory for videos and annotations")
    parser.add_argument("--min_duration", type=int, default=3600,
                        help="Minimum video duration in seconds (default: 3600 = 1hr)")
    parser.add_argument("--n_videos", type=int, default=5,
                        help="Number of videos to download (default: 5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n================================================")
    print(" Ego4D: Download 5 long videos + annotations")
    print("================================================\n")

    # Step 1: metadata
    print("[1/4] Fetching metadata...")
    meta_path = download_metadata(output_dir)
    print(f"  Metadata: {meta_path}")

    # Step 2: select
    print(f"\n[2/4] Selecting {args.n_videos} videos >= {args.min_duration}s ...")
    selected = select_videos(meta_path, args.min_duration, args.n_videos)

    # Save selection list
    uids_path = output_dir / "selected_video_uids.txt"
    with open(uids_path, "w") as f:
        for v in selected:
            f.write(f"{v['uid']}\n")
    print(f"\n  UIDs saved to {uids_path}")

    # Step 3: download videos
    print(f"\n[3/4] Downloading {len(selected)} videos...")
    video_dir = output_dir / "videos"
    download_videos(selected, video_dir)
    print(f"  Videos in: {video_dir}")

    # Step 4: annotations
    print("\n[4/4] Downloading + filtering annotations...")
    ann_path = download_and_filter_annotations(selected, output_dir)

    print("\n================================================")
    print(" Done!")
    print(f"  Videos     : {video_dir}")
    print(f"  Annotations: {ann_path}")
    print(f"  UIDs list  : {uids_path}")
    print("\n Next step:")
    print("   bash run_experiment.sh")
    print("================================================\n")


if __name__ == "__main__":
    main()
