"""
Experiment 1 — Hierarchical chunk-level captioning of a single long video.

Splits one video's frame directory into fixed-duration temporal chunks,
captions each chunk independently with MA-LMM, then uses the in-memory
Vicuna LLM to summarize the chunk captions into a single video-level
description.

Two canonical settings for a 60-min video:
  --chunk_duration_min 10  →  6 chunks, each 10 min  →  6 captions
  --chunk_duration_min 1   →  60 chunks, each 1 min  →  60 captions

Usage (single video, 10-min chunks):
    python scripts/infer_chunked.py \
        --frame_dir data/frames/60min/<video_id> \
        --output    outputs/exp1_chunked/10min_chunks/<video_id>.json \
        --chunk_duration_min 10 \
        --video_fps 10 \
        --num_frames_per_chunk 80 \
        --memory_bank_length 40

Use run_chunked.sh to process all videos in a duration split.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent.parent / "MA-LMM"
sys.path.insert(0, str(REPO_DIR))

_TRANSFORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
_TRANSFORM_STD  = [0.26862954, 0.26130258, 0.27577711]


def load_frames_from_list(files, num_frames, image_size=224):
    """Uniformly subsample `num_frames` from `files` and return [T, C, H, W]."""
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_TRANSFORM_MEAN, std=_TRANSFORM_STD),
    ])
    idx = np.linspace(0, len(files) - 1, num_frames, dtype=int)
    return torch.stack([transform(Image.open(files[i]).convert("RGB")) for i in idx])


def _llm_generate(model, prompt, device, max_new_tokens=300):
    """Single greedy-decode pass through Vicuna. Greedy (num_beams=1) avoids
    the ~1.5 GiB beam-search attention buffer that causes OOM on 22 GiB GPUs."""
    inputs = model.llm_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        out = model.llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            repetition_penalty=1.3,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return model.llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def summarize_chunk_captions(model, chunk_captions, chunk_duration_min,
                             group_size=15):
    """
    Text-only Vicuna pass: merge N chunk captions into one paragraph.
    Re-uses the LLM already loaded inside the MA-LMM model object.

    For large numbers of chunks the prompt overflows the 2048-token context and
    earlier segments are silently dropped.  When len(chunk_captions) > group_size
    we first summarize in groups of `group_size`, then merge those summaries —
    hierarchical summarization keeps each prompt well within the token budget.
    """
    device = next(model.parameters()).device

    def _make_prompt(captions, start_offset_chunks):
        n = len(captions)
        segments = "\n".join(
            f"  Segment {start_offset_chunks + i + 1} "
            f"({int((start_offset_chunks + i) * chunk_duration_min)}-"
            f"{int((start_offset_chunks + i + 1) * chunk_duration_min)} min): {cap}"
            for i, cap in enumerate(captions)
        )
        return (
            f"Below are descriptions of {n} sequential "
            f"{chunk_duration_min}-minute segments from a single video. "
            "Write one coherent paragraph that summarizes these segments.\n\n"
            f"{segments}\n\nComprehensive summary:"
        )

    if len(chunk_captions) <= group_size:
        prompt = _make_prompt(chunk_captions, 0)
        return _llm_generate(model, prompt, device)

    # Hierarchical: summarize groups, then merge the group summaries.
    group_summaries = []
    for g_start in range(0, len(chunk_captions), group_size):
        group = chunk_captions[g_start: g_start + group_size]
        prompt = _make_prompt(group, g_start)
        group_summaries.append(_llm_generate(model, prompt, device, max_new_tokens=200))

    merge_prompt = (
        f"Below are {len(group_summaries)} partial summaries of consecutive "
        "portions of a single video. Write one coherent paragraph that covers "
        "the entire video.\n\n"
        + "\n".join(f"  Part {i+1}: {s}" for i, s in enumerate(group_summaries))
        + "\n\nComprehensive video summary:"
    )
    return _llm_generate(model, merge_prompt, device)


def build_chunks(all_frames, chunk_duration_min, video_fps, min_frames_per_chunk):
    """Split frame list into temporal chunks; drop trailing chunks that are too short."""
    frames_per_chunk = max(1, int(chunk_duration_min * 60 * video_fps))
    chunks = []
    for start in range(0, len(all_frames), frames_per_chunk):
        chunk = all_frames[start: start + frames_per_chunk]
        if len(chunk) >= min_frames_per_chunk:
            chunks.append(chunk)
    return chunks, frames_per_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", required=True,
                    help="Frame dir for one video (contains frame*.jpg)")
    ap.add_argument("--output", required=True,
                    help="Output JSON path")
    ap.add_argument("--chunk_duration_min", type=float, default=10,
                    help="Duration of each temporal chunk in minutes (e.g., 10 or 1)")
    ap.add_argument("--video_fps", type=float, default=10,
                    help="FPS used during frame extraction (must match prepare_frames.sh)")
    ap.add_argument("--num_frames_per_chunk", type=int, default=80,
                    help="Frames uniformly sampled from each chunk for MA-LMM")
    ap.add_argument("--memory_bank_length", type=int, default=40)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--prompt",
                    default="Describe what happens in this video segment in detail.")
    ap.add_argument("--no_summarize", action="store_true",
                    help="Skip Vicuna summarization; output only chunk captions")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    frame_dir = Path(args.frame_dir)
    all_frames = sorted(frame_dir.glob("frame*.jpg"))
    if not all_frames:
        raise FileNotFoundError(f"No frame*.jpg files in {frame_dir}")

    # Minimum frames a chunk must have to be captioned (avoid 1-frame chunks).
    min_chunk_frames = max(1, args.num_frames_per_chunk // 4)
    chunks, frames_per_chunk = build_chunks(
        all_frames, args.chunk_duration_min, args.video_fps, min_chunk_frames
    )

    print(
        f"Video : {frame_dir.name}\n"
        f"Frames: {len(all_frames)} total  |  "
        f"{frames_per_chunk} frames/chunk  |  "
        f"{len(chunks)} chunks"
    )

    print(f"Loading MA-LMM on {args.device}…")
    from lavis.models import load_model_and_preprocess
    model, _, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct_malmm",
        model_type="vicuna7b",
        is_eval=True,
        device=args.device,
    )
    model.memory_bank_length = args.memory_bank_length
    model.use_memory_bank = args.memory_bank_length > 0
    model.num_frames = args.num_frames_per_chunk
    model.eval()

    chunk_results = []
    wall_t0 = time.time()

    for i, chunk_files in enumerate(tqdm(chunks, desc="chunks")):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)

        frames = (
            load_frames_from_list(chunk_files, args.num_frames_per_chunk)
            .permute(1, 0, 2, 3)  # [T,C,H,W] → [C,T,H,W]
            .unsqueeze(0)          # → [B=1,C,T,H,W]
            .to(args.device)
        )
        sample = {
            "image": frames,
            "prompt": args.prompt,
            "num_frames": args.num_frames_per_chunk,
            "is_video": True,
        }

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                sample,
                use_nucleus_sampling=False,
                num_beams=args.num_beams,
                max_length=args.max_len,
                min_length=args.min_len,
            )
        elapsed = time.time() - t0
        caption = out[0] if isinstance(out, list) else str(out)
        peak_gb = (
            torch.cuda.max_memory_allocated(args.device) / 1024 ** 3
            if torch.cuda.is_available() else 0.0
        )

        start_min = i * args.chunk_duration_min
        end_min   = start_min + args.chunk_duration_min
        chunk_results.append({
            "chunk_idx":          i,
            "start_min":          round(start_min, 2),
            "end_min":            round(end_min, 2),
            "n_frames_in_chunk":  len(chunk_files),
            "num_frames_sampled": args.num_frames_per_chunk,
            "caption":            caption,
            "inference_time_sec": round(elapsed, 2),
            "gpu_peak_memory_gb": round(peak_gb, 3),
        })
        print(
            f"  [{i+1}/{len(chunks)}] {start_min:.0f}–{end_min:.0f} min  "
            f"| {elapsed:.1f}s  {peak_gb:.2f}GB  "
            f"| {caption[:80]}…"
        )

    total_elapsed = time.time() - wall_t0

    summary = None
    if not args.no_summarize:
        print("\nSummarizing chunk captions with Vicuna…")
        t_sum = time.time()
        summary = summarize_chunk_captions(
            model,
            [r["caption"] for r in chunk_results],
            args.chunk_duration_min,
        )
        elapsed_sum = time.time() - t_sum
        print(f"  Summary ({elapsed_sum:.1f}s): {summary[:200]}…")

    result = {
        "video_id":                    frame_dir.name,
        "chunk_duration_min":          args.chunk_duration_min,
        "n_chunks":                    len(chunks),
        "num_frames_per_chunk":        args.num_frames_per_chunk,
        "memory_bank_length":          args.memory_bank_length,
        "total_inference_time_sec":    round(total_elapsed, 2),
        "summary":                     summary,
        "chunks":                      chunk_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
