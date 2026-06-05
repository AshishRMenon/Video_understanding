"""
Step 5 — Analyse and compare VideoRecap vs MA-LMM outputs.

Prints a side-by-side comparison and saves comparison_malmm_vs_vidrecap.json.
"""

import argparse
import json
import os
import re
import pickle
from pathlib import Path
from collections import Counter


def type_token_ratio(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def is_degenerate(text, threshold=0.8):
    if not text:
        return True
    counts   = Counter(text)
    top_frac = counts.most_common(1)[0][1] / len(text)
    return top_frac > threshold


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def load_malmm(malmm_dir):
    results = {}
    for f in sorted(Path(malmm_dir).glob("*_result.json")):
        d   = json.load(open(f))
        vid = d["video_id"].replace("_10min", "")
        results[vid] = d
    return results


def load_vidrecap_summaries(path):
    if not os.path.exists(path):
        return {}
    data = json.load(open(path))
    return {e["vid"].replace("_10min", ""): e for e in data}


def load_vidrecap_segments(path):
    if not os.path.exists(path):
        return {}
    segs   = pickle.load(open(path, "rb"))
    by_vid = {}
    for s in segs:
        by_vid.setdefault(s["vid"].replace("_10min", ""), []).append(s)
    return by_vid


def load_vidrecap_clips(path):
    if not os.path.exists(path):
        return {}
    clips  = pickle.load(open(path, "rb"))
    by_vid = {}
    for (vid, s, e, cap) in clips:
        by_vid.setdefault(vid.replace("_10min", ""), []).append((s, e, cap))
    return by_vid


def safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main(args):
    malmm        = load_malmm(args.malmm_dir)
    vr_summaries = load_vidrecap_summaries(args.vidrecap_summaries)
    vr_segments  = load_vidrecap_segments(args.vidrecap_segments)
    vr_clips     = load_vidrecap_clips(args.vidrecap_clips)

    all_vids = sorted(set(list(malmm.keys()) + list(vr_summaries.keys())))

    print("\n" + "=" * 70)
    print(" Per-Video Comparison: MA-LMM  vs  VideoRecap")
    print("=" * 70)

    rows = []
    for base in all_vids:
        m  = malmm.get(base, {})
        vr = vr_summaries.get(base, {})

        malmm_cap = m.get("caption", "")
        vr_sum    = vr.get("video_summary", "")

        row = dict(
            vid         = base,
            malmm_cap   = malmm_cap,
            malmm_words = word_count(malmm_cap),
            malmm_ttr   = type_token_ratio(malmm_cap),
            malmm_degen = is_degenerate(malmm_cap),
            vr_sum      = vr_sum,
            vr_words    = word_count(vr_sum),
            vr_ttr      = type_token_ratio(vr_sum),
            vr_degen    = is_degenerate(vr_sum),
            vr_n_segs   = vr.get("n_segments", 0),
        )
        rows.append(row)

        print(f"\n{'─'*70}")
        print(f"Video: {base}")
        print(f"\n  MA-LMM  (words={row['malmm_words']}, "
              f"TTR={row['malmm_ttr']:.3f}, degenerate={row['malmm_degen']}):")
        print(f"    {malmm_cap[:200]}{'...' if len(malmm_cap)>200 else ''}")

        if vr_sum:
            print(f"\n  VideoRecap  (words={row['vr_words']}, "
                  f"TTR={row['vr_ttr']:.3f}, "
                  f"segments={row['vr_n_segs']}, degenerate={row['vr_degen']}):")
            print(f"    {vr_sum[:200]}{'...' if len(vr_sum)>200 else ''}")
            for seg in vr_segments.get(base, []):
                desc = seg.get("generated_text", "")
                print(f"    [{seg['start_sec']:.0f}-{seg['end_sec']:.0f}s] {desc[:110]}")
        else:
            print("\n  VideoRecap: (not yet generated — run steps 01–04 first)")

    print("\n" + "=" * 70)
    print(" Aggregate Statistics")
    print("=" * 70)
    hdr = f"{'Metric':<32} {'MA-LMM':>12} {'VideoRecap':>12}"
    print(hdr)
    print("─" * len(hdr))

    stats = [
        ("Avg words per caption",
         safe_mean([r["malmm_words"] for r in rows]),
         safe_mean([r["vr_words"]    for r in rows if r["vr_words"] > 0])),
        ("Avg type-token ratio",
         safe_mean([r["malmm_ttr"]   for r in rows]),
         safe_mean([r["vr_ttr"]      for r in rows if r["vr_ttr"] > 0])),
        ("# degenerate captions",
         sum(r["malmm_degen"] for r in rows),
         sum(r["vr_degen"]    for r in rows if r["vr_words"] > 0)),
    ]
    for metric, a, b in stats:
        if isinstance(a, float):
            print(f"{metric:<32} {a:>12.3f} {b:>12.3f}")
        else:
            print(f"{metric:<32} {a:>12} {b:>12}")

    if vr_clips:
        print("\n" + "=" * 70)
        print(" VideoRecap Clip Sample (first 4 per video)")
        print("=" * 70)
        for vid, clips in sorted(vr_clips.items()):
            print(f"\n  [{vid}]  {len(clips)} total clips")
            for (s, e, cap) in clips[:4]:
                print(f"    [{s:.0f}-{e:.0f}s] {cap[:100]}")

    # Save JSON comparison
    os.makedirs(args.output_dir, exist_ok=True)
    comparison = [{
        "vid": base,
        "malmm": {
            "caption":    malmm.get(base, {}).get("caption", ""),
            "word_count": word_count(malmm.get(base, {}).get("caption", "")),
            "ttr":        type_token_ratio(malmm.get(base, {}).get("caption", "")),
            "degenerate": is_degenerate(malmm.get(base, {}).get("caption", "")),
            "gpu_peak_memory_gb": malmm.get(base, {}).get("gpu_peak_memory_gb"),
            "inference_time_sec": malmm.get(base, {}).get("inference_time_sec"),
        },
        "videorecap": {
            "video_summary":         vr_summaries.get(base, {}).get("video_summary", ""),
            "word_count":            word_count(vr_summaries.get(base, {}).get("video_summary", "")),
            "ttr":                   type_token_ratio(vr_summaries.get(base, {}).get("video_summary", "")),
            "degenerate":            is_degenerate(vr_summaries.get(base, {}).get("video_summary", "")),
            "n_segments":            vr_summaries.get(base, {}).get("n_segments", 0),
            "segment_descriptions":  vr_summaries.get(base, {}).get("segment_descriptions", []),
        },
    } for base in all_vids]

    out = os.path.join(args.output_dir, "comparison_malmm_vs_vidrecap.json")
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare VideoRecap vs MA-LMM outputs")

    # ── paths ──────────────────────────────────────────────────────────────
    parser.add_argument("--malmm_dir",           required=True,
                        help="Directory with MA-LMM *_result.json files")
    parser.add_argument("--vidrecap_summaries",  required=True,
                        help="Path to video_summaries.json (step 4 output)")
    parser.add_argument("--output_dir",          required=True,
                        help="Directory to save comparison JSON")

    # ── optional extra files for richer output ─────────────────────────────
    parser.add_argument("--vidrecap_segments",   default="",
                        help="(optional) segment_descriptions.pkl for segment-level output")
    parser.add_argument("--vidrecap_clips",      default="",
                        help="(optional) clip_captions.pkl for clip-level sample output")

    main(parser.parse_args())
