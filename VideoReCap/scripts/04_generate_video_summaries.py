"""
Step 4 — Level 3: Generate video-level summaries.

Reads segment descriptions from step 3 and CLS features from step 1.
Aggregates all segments per video into one entry, then runs the
VideoRecap video summary model.

Output: video_summaries.json with one summary per video.
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


def build_video_metadata(segment_descs):
    """
    Aggregate per-segment dicts into one entry per video.

    Each entry:
      'vid', 'start_sec', 'end_sec',
      'segment_descriptions_pred': [desc1, desc2, ...],
      'segment_timestamps':        [(s, e), ...]
    """
    by_vid = {}
    for seg in segment_descs:
        vid  = seg["vid"]
        desc = seg.get("generated_text", "")
        if vid not in by_vid:
            by_vid[vid] = {
                "vid":                        vid,
                "start_sec":                  seg["start_sec"],
                "end_sec":                    seg["end_sec"],
                "segment_descriptions_pred":  [],
                "segment_timestamps":         [],
            }
        by_vid[vid]["end_sec"] = max(by_vid[vid]["end_sec"], seg["end_sec"])
        by_vid[vid]["segment_descriptions_pred"].append(desc)
        by_vid[vid]["segment_timestamps"].append((seg["start_sec"], seg["end_sec"]))
    return list(by_vid.values())


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading segment descriptions: {args.segment_descs}")
    with open(args.segment_descs, "rb") as f:
        segment_descs = pickle.load(f)
    n_vids = len({s["vid"] for s in segment_descs})
    print(f"  {len(segment_descs)} segments across {n_vids} videos")

    video_metadata = build_video_metadata(segment_descs)
    print(f"  Built {len(video_metadata)} video entries")

    print(f"\nLoading checkpoint: {args.video_ckpt}")
    ckpt       = torch.load(args.video_ckpt, map_location="cpu")
    old_args   = ckpt["args"]
    state_dict = OrderedDict({k.replace("module.", ""): v
                              for k, v in ckpt["state_dict"].items()})

    old_args.video_feature_path  = args.feature_dir
    old_args.video_feature_type  = "cls"
    old_args.video_sampling_type = "uniform"
    old_args.dataset             = "video_summary"
    old_args.text_feature_type   = "token"
    old_args.metadata            = video_metadata
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
    print(f"  Dataset: {len(dataset)} videos")

    print("Loading VideoRecap video summary model...")
    model = VideoRecap(old_args)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}")

    t0 = time.time()
    print(f"\nGenerating video summaries ({len(dataset)} videos)...")
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
                summary = decode_one(generated_ids[j], tokenizer).strip()
                dataset.samples[indices[j].item()]["video_summary"] = summary

    results = [{
        "vid":                   s["vid"],
        "start_sec":             s["start_sec"],
        "end_sec":               s["end_sec"],
        "n_segments":            len(s["segment_descriptions_pred"]),
        "segment_timestamps":    s.get("segment_timestamps", []),
        "segment_descriptions":  s["segment_descriptions_pred"],
        "video_summary":         s.get("video_summary", ""),
    } for s in dataset.samples]

    out_path = os.path.join(args.output_dir, "video_summaries.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}  ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 60)
    for r in results:
        print(f"\n[{r['vid']}]  {r['n_segments']} segments")
        print(f"  Summary: {r['video_summary']}")
        for ts, desc in zip(r["segment_timestamps"], r["segment_descriptions"]):
            print(f"    [{ts[0]:.0f}-{ts[1]:.0f}s] {desc[:100]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VideoRecap Level-3: video summaries")

    # ── paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--feature_dir",    required=True,
                        help="Directory with per-video .npy CLS features (from step 1)")
    parser.add_argument("--segment_descs",  required=True,
                        help="Path to segment_descriptions.pkl (output of step 3)")
    parser.add_argument("--video_ckpt",     required=True,
                        help="Path to videorecap_video.pt checkpoint")
    parser.add_argument("--output_dir",     required=True,
                        help="Directory to save video_summaries.json")

    # ── compute options ────────────────────────────────────────────────────
    parser.add_argument("--batch_size",     type=int, default=4,
                        help="Videos per forward pass (default: 4)")
    parser.add_argument("--num_workers",    type=int, default=2,
                        help="DataLoader workers (default: 2)")

    main(parser.parse_args())
