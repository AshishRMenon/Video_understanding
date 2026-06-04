"""
Two-stage video captioning that avoids the LLaMA-1 / LLaMA-2 mismatch in the
original infer.py (instruct_blip_vicuna7b_trimmed.pth was trained with a
LLaMA-1 Vicuna; the on-disk weights are v1.5 / LLaMA-2).

Stage 1 – InstructBLIP-FlanT5-XL (Salesforce/instructblip-flan-t5-xl)
          captions each of N_KEYFRAMES uniformly-sampled frames.
          FlanT5 carries no LLaMA licence and has no version-mismatch issue.

Stage 2 – Vicuna-7b-v1.5 (the weights already on disk) acts as a
          pure-text summariser: it is given the N frame descriptions and
          asked to write one cohesive paragraph about the full video.

Usage:
    python scripts/infer_v2.py \
        --frame_dir data/frames/10min \
        --ann_path  data/annotations/ego4d_10min.json \
        --output    outputs/captions_10min.json \
        --duration_label "10-minute"
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)

HF_CACHE = os.environ.get("HF_CACHE", "/workspace/.hf_cache")
INSTBLIP_MODEL = "Salesforce/instructblip-flan-t5-xl"
FRAME_PROMPT = (
    "Briefly describe what the person is doing and what objects or environment "
    "are visible in this first-person egocentric video frame."
)


def sample_frames(frame_dir: Path, n: int) -> list:
    files = sorted(frame_dir.glob("frame*.jpg"))
    if not files:
        raise FileNotFoundError(f"No frames in {frame_dir}")
    idx = np.linspace(0, len(files) - 1, n, dtype=int)
    return [files[i] for i in idx]


def caption_frames_batch(frame_paths: list, processor, model, device: str) -> list:
    captions = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        inputs = processor(
            images=img,
            text=FRAME_PROMPT,
            return_tensors="pt",
        ).to(device, torch.float16)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=80, num_beams=3)
        cap = processor.decode(ids[0], skip_special_tokens=True).strip()
        captions.append(cap)
    return captions


def synthesize_caption(
    frame_captions: list,
    duration_label: str,
    tokenizer,
    llm,
    device: str,
) -> str:
    joined = "\n".join(f"Frame {i+1}: {c}" for i, c in enumerate(frame_captions))
    prompt = (
        f"Below are brief descriptions of {len(frame_captions)} frames sampled "
        f"uniformly from a {duration_label} first-person egocentric video.\n"
        "Write ONE cohesive paragraph (4-6 sentences) describing the full video "
        "activity from start to finish. Be specific and descriptive.\n\n"
        f"{joined}\n\nVideo description:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            num_beams=3,
            repetition_penalty=1.3,
        )
    new_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--ann_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vicuna_path",
                    default="/workspace/MA_LMM_coding/MA-LMM/llm/vicuna-7b")
    ap.add_argument("--n_keyframes", type=int, default=16)
    ap.add_argument("--duration_label", default="10-minute")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.ann_path) as f:
        anns = json.load(f)

    # ── Stage 1: per-frame captions via InstructBLIP-FlanT5-XL ───────────
    print(f"\n[Stage 1] Loading {INSTBLIP_MODEL} (cache → {HF_CACHE}) …")
    processor = InstructBlipProcessor.from_pretrained(
        INSTBLIP_MODEL, cache_dir=HF_CACHE
    )
    blip = InstructBlipForConditionalGeneration.from_pretrained(
        INSTBLIP_MODEL,
        torch_dtype=torch.float16,
        cache_dir=HF_CACHE,
    ).to(args.device).eval()
    print("  Model loaded.")

    frame_captions: dict = {}
    for ann in tqdm(anns, desc="Frame captions"):
        vid = ann["video_id"]
        fdir = Path(args.frame_dir) / vid
        if not fdir.exists():
            print(f"  [skip] {fdir} missing")
            continue
        frames = sample_frames(fdir, args.n_keyframes)
        frame_captions[vid] = caption_frames_batch(frames, processor, blip, args.device)

    del blip
    torch.cuda.empty_cache()

    # ── Stage 2: video-level synthesis via Vicuna-7b ──────────────────────
    print(f"\n[Stage 2] Loading Vicuna-7b from {args.vicuna_path} …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.vicuna_path, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    vicuna = AutoModelForCausalLM.from_pretrained(
        args.vicuna_path, torch_dtype=torch.float16
    ).to(args.device).eval()
    print("  Model loaded.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = []
    for ann in tqdm(anns, desc=f"Synthesis ({Path(args.frame_dir).name})"):
        vid = ann["video_id"]
        if vid not in frame_captions:
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)
        t0 = time.time()

        caption = synthesize_caption(
            frame_captions[vid],
            args.duration_label,
            tokenizer,
            vicuna,
            args.device,
        )
        elapsed = time.time() - t0
        peak_gb = (
            torch.cuda.max_memory_allocated(args.device) / 1024**3
            if torch.cuda.is_available() else 0.0
        )

        rec = {
            "video_id": vid,
            "n_frames_available": ann.get("n_frames"),
            "n_keyframes_captioned": args.n_keyframes,
            "frame_captions": frame_captions[vid],
            "caption": caption,
            "inference_time_sec": round(elapsed, 2),
            "gpu_peak_memory_gb": round(peak_gb, 3),
        }
        results.append(rec)
        print(f"  {vid[:8]}…: {caption[:140]}")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results → {args.output}")


if __name__ == "__main__":
    main()
