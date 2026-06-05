"""
Step 3 — Level 2: Generate segment descriptions (~3-minute windows).

Reads clip captions from step 2 and CLS features from step 1.
Groups clips into SEGMENT_DURATION-second windows per video, then
runs the VideoRecap segment model on each window.

Output: pickle file of per-segment dicts with 'generated_text'.
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
from tqdm import tqdm

VIDRECAP_ROOT = "/workspace/MA_LMM_coding/VideoRecap"
sys.path.insert(0, VIDRECAP_ROOT)

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


def build_segment_metadata(clip_captions, segment_duration):
    """
    Group (vid, start, end, caption) tuples into segment-duration windows.

    Each segment entry dict has:
      'vid', 'start_sec', 'end_sec',
      'captions_pred': [(vid, start, end, caption), ...]
    The dataset loader reads captions_pred[i][3] as the caption text,
    which matches index 3 of (vid, start, end, caption).
    """
    by_vid = {}
    for (vid, start, end, cap) in clip_captions:
        by_vid.setdefault(vid, []).append((vid, start, end, cap))

    segments = []
    for vid, clips in sorted(by_vid.items()):
        clips = sorted(clips, key=lambda x: x[1])
        seg_start  = clips[0][1]
        seg_clips  = []
        for clip in clips:
            _, c_start, c_end, _ = clip
            if seg_clips and (c_start - seg_start) >= segment_duration:
                segments.append({
                    "vid":           vid,
                    "start_sec":     seg_start,
                    "end_sec":       seg_clips[-1][2],
                    "captions_pred": seg_clips,
                })
                seg_start = c_start
                seg_clips = []
            seg_clips.append(clip)
        if seg_clips:
            segments.append({
                "vid":           vid,
                "start_sec":     seg_start,
                "end_sec":       seg_clips[-1][2],
                "captions_pred": seg_clips,
            })
    return segments


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading clip captions: {args.clip_captions}")
    with open(args.clip_captions, "rb") as f:
        clip_captions = pickle.load(f)
    print(f"  {len(clip_captions)} clips")

    segment_metadata = build_segment_metadata(clip_captions, args.segment_duration)
    n_vids = len({s["vid"] for s in segment_metadata})
    print(f"  Built {len(segment_metadata)} segments across {n_vids} videos")

    print(f"\nLoading checkpoint: {args.segment_ckpt}")
    ckpt       = torch.load(args.segment_ckpt, map_location="cpu")
    old_args   = ckpt["args"]
    state_dict = OrderedDict({k.replace("module.", ""): v
                              for k, v in ckpt["state_dict"].items()})

    old_args.video_feature_path  = args.feature_dir
    old_args.video_feature_type  = "cls"
    old_args.video_sampling_type = "uniform"
    old_args.dataset             = "segment_description"
    old_args.text_feature_type   = "token"
    old_args.metadata            = segment_metadata
    old_args.chunk_len           = -1

    for attr, default in [("finetune_mapper", False), ("query_width", 768),
                          ("vision_model_type", "timesformer"),
                          ("cross_attn_freq", 1), ("use_lora", False),
                          ("share_mapper", False), ("freeze_lm_entire", False),
                          ("video_encoder_ckpt", "")]:
        if not hasattr(old_args, attr):
            setattr(old_args, attr, default)

    tokenizer = AutoTokenizer.from_pretrained(old_args.decoder_name)
    dataset   = VideoCaptionDataset(old_args, transform=None, is_training=False)
    collator  = CaptionDataCollator(
        tokenizer, max_gen_tokens=old_args.max_gen_tokens,
        add_bos=True, add_eos=True, pad_token_id=0,
    )
    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=collator,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    print(f"  Dataset: {len(dataset)} segments")

    print("Loading VideoRecap segment model...")
    model = VideoRecap(old_args)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}")

    t0 = time.time()
    print(f"\nGenerating segment descriptions ({len(dataset)} segments)...")
    with torch.no_grad():
        for samples in tqdm(loader):
            indices = samples["indices"]
            queries = model.map_features(samples)
            generated_ids, _ = model.generate(
                queries, tokenizer,
                do_sample=False,
                max_text_length=old_args.max_gen_tokens,
                num_return_sequences=1,
            )
            for j in range(generated_ids.shape[0]):
                desc = decode_one(generated_ids[j], tokenizer).strip()
                dataset.samples[indices[j].item()]["generated_text"] = desc

    pkl_path = os.path.join(args.output_dir, "segment_descriptions.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset.samples, f)
    print(f"\nSaved pickle → {pkl_path}  ({time.time()-t0:.1f}s)")

    json_path = os.path.join(args.output_dir, "segment_descriptions_summary.json")
    with open(json_path, "w") as f:
        json.dump([{"vid": s["vid"], "start_sec": s["start_sec"],
                    "end_sec": s["end_sec"],
                    "description": s.get("generated_text", "")}
                   for s in dataset.samples], f, indent=2)
    print(f"Saved JSON  → {json_path}")

    for s in dataset.samples[:4]:
        print(f"\n  [{s['vid']}] {s['start_sec']:.0f}-{s['end_sec']:.0f}s")
        print(f"    {s.get('generated_text', '')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VideoRecap Level-2: segment descriptions")

    # ── paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--feature_dir",       required=True,
                        help="Directory with per-video .npy CLS features (from step 1)")
    parser.add_argument("--clip_captions",     required=True,
                        help="Path to clip_captions.pkl (output of step 2)")
    parser.add_argument("--segment_ckpt",      required=True,
                        help="Path to videorecap_segment.pt checkpoint")
    parser.add_argument("--output_dir",        required=True,
                        help="Directory to save segment_descriptions.pkl and JSON")

    # ── data options ───────────────────────────────────────────────────────
    parser.add_argument("--segment_duration",  type=float, default=180.0,
                        help="Seconds per segment window (default: 180 = 3 min)")

    # ── compute options ────────────────────────────────────────────────────
    parser.add_argument("--batch_size",        type=int, default=8,
                        help="Segments per forward pass (default: 8)")
    parser.add_argument("--num_workers",       type=int, default=4,
                        help="DataLoader workers (default: 4)")

    main(parser.parse_args())
