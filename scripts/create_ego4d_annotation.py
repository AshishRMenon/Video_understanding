"""
Creates annotation JSON for MA-LMM inference on Ego4D hour-long videos.
Formats: compatible with MA-LMM's VideoCaptionEvalDataset.

For each video, creates one annotation entry. If Ego4D-HCap ground-truth
captions are available, they are included for metric computation.
"""

import os
import json
import argparse
from pathlib import Path


# Selected 5 Ego4D video IDs from Ego4D-HCap (from VideoRecap paper supplementary).
# Replace with actual IDs once you have Ego4D access.
# These are placeholders — update after running: ego4d --metadata
SELECTED_VIDEO_IDS = [
    # Format: (video_id, approximate_duration_min, character)
    ("PLACEHOLDER_V1", 60, "highly_eventful_cooking_social"),
    ("PLACEHOLDER_V2", 62, "sparse_walking_commute"),
    ("PLACEHOLDER_V3", 90, "mixed_sparse_and_eventful"),
    ("PLACEHOLDER_V4", 45, "repetitive_assembly_cleaning"),
    ("PLACEHOLDER_V5", 58, "dense_activity_throughout"),
]


def build_annotation(video_id: str, frame_dir: str, gt_caption: str = "", instance_id: int = 0):
    return {
        "video": video_id,           # used as key to locate frames
        "image_id": instance_id,     # COCO eval compat
        "instance_id": instance_id,
        "caption": gt_caption,       # empty string if no GT available
        "video_id": video_id,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", default="data/ego4d/frames",
                        help="Root dir with subdirs named by video_id")
    parser.add_argument("--ego4d_hcap_path", default=None,
                        help="Optional: path to Ego4D-HCap annotation JSON (from VideoRecap repo)")
    parser.add_argument("--out_path", default="data/ego4d/annotations/ego4d_eval.json",
                        help="Output annotation JSON path")
    parser.add_argument("--video_ids", nargs="+", default=None,
                        help="Specific video IDs to include. If None, uses all dirs in frame_dir")
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load Ego4D-HCap GT captions if available
    gt_captions = {}
    if args.ego4d_hcap_path and os.path.exists(args.ego4d_hcap_path):
        with open(args.ego4d_hcap_path) as f:
            hcap = json.load(f)
        # Ego4D-HCap format: list of {video_uid, summary, ...}
        for entry in hcap:
            vid = entry.get("video_uid", entry.get("video_id", ""))
            # Use the highest-level (video-level) caption
            caption = entry.get("summary", entry.get("caption", ""))
            gt_captions[vid] = caption
        print(f"Loaded {len(gt_captions)} GT captions from Ego4D-HCap")

    # Collect video IDs
    if args.video_ids:
        video_ids = args.video_ids
    else:
        video_ids = sorted([d.name for d in frame_dir.iterdir() if d.is_dir()])

    annotations = []
    for idx, vid_id in enumerate(video_ids):
        vid_frame_dir = frame_dir / vid_id
        if not vid_frame_dir.exists():
            print(f"  WARNING: No frame dir for {vid_id}, skipping")
            continue
        n_frames = len(list(vid_frame_dir.glob("*.jpg")))
        gt_cap = gt_captions.get(vid_id, "")
        ann = build_annotation(vid_id, str(vid_frame_dir), gt_cap, idx)
        ann["n_frames"] = n_frames
        annotations.append(ann)
        print(f"  {vid_id}: {n_frames} frames, GT: {'yes' if gt_cap else 'NO'}")

    with open(out_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"\nSaved {len(annotations)} annotations to {out_path}")


if __name__ == "__main__":
    main()
