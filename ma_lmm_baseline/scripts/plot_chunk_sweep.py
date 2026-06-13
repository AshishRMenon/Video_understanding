"""
Experiment 4 — Combined metrics graph: caption quality vs. chunk size (Graph 2).

Mirrors plot_sweep.py but X-axis is chunk size (2 / 5 / 10 min) instead of
frame budget.  Reads captions_{N}min.json + qa_{N}min.json produced by
run_chunk_sweep.sh and computes ROUGE-L, BERTScore, Macro QA, Micro QA.

Usage:
    python scripts/plot_chunk_sweep.py \\
        --out_dir     outputs/exp4_chunk_sweep \\
        --chunk_sizes 2 5 10 \\
        --plot_dir    outputs/exp4_chunk_sweep/plots \\
        --narration   /workspace/ego4d_meta/v2/annotations/narration.json \\
        --ann_path    data/annotations/ego4d_30min.json
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import shared style + helpers from plot_sweep
sys.path.insert(0, str(Path(__file__).parent))
from plot_sweep import (
    METRIC_STYLE,
    load_ground_truth,
    compute_rouge_l,
    compute_bertscore,
    plot_combined_metrics,
)


def load_qa(qa_path: str) -> dict:
    """Read {macro_qa_score, micro_qa_score} from a qa_eval.py single-config output."""
    if not Path(qa_path).exists():
        return {}
    with open(qa_path) as f:
        d = json.load(f)
    return {
        "macro_qa_score": d.get("macro_qa_score"),
        "micro_qa_score": d.get("micro_qa_score"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",     required=True,
                    help="Directory containing captions_Nmin.json and qa_Nmin.json files")
    ap.add_argument("--chunk_sizes", nargs="+", type=int, default=[2, 5, 10],
                    help="Chunk sizes in minutes (must match files in out_dir)")
    ap.add_argument("--plot_dir",    default=None,
                    help="Output directory for plots (default: {out_dir}/plots)")
    ap.add_argument("--narration",   default=None,
                    help="Ego4D narration.json (enables ROUGE-L and BERTScore)")
    ap.add_argument("--ann_path",    default=None,
                    help="Annotation JSON (e.g. data/annotations/ego4d_30min.json)")
    ap.add_argument("--no_bertscore", action="store_true",
                    help="Skip BERTScore computation")
    args = ap.parse_args()

    plot_dir = args.plot_dir or f"{args.out_dir}/plots"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Ground truth for ROUGE-L / BERTScore
    gt = {}
    if args.narration and args.ann_path:
        gt = load_ground_truth(args.narration, args.ann_path)
        print(f"Ground-truth summaries loaded for {len(gt)} video(s).")

    # ---- Collect per-chunk-size metrics ------------------------------------
    x_labels  = []
    rouge_vals = []
    bert_vals  = []
    macro_vals = []
    micro_vals = []

    for cs in sorted(args.chunk_sizes):
        label    = f"{cs} min"
        cap_path = Path(args.out_dir) / f"captions_{cs}min.json"
        qa_path  = Path(args.out_dir) / f"qa_{cs}min.json"

        if not cap_path.exists():
            print(f"[skip] {cap_path} not found — run run_chunk_sweep.sh first.")
            continue

        with open(cap_path) as f:
            records = json.load(f)
        preds = [r.get("caption", "") for r in records]
        refs  = [gt.get(r["video_id"], "") for r in records] if gt else []

        x_labels.append(label)

        # ROUGE-L
        if refs:
            rouge_vals.append(compute_rouge_l(preds, refs))
        else:
            rouge_vals.append(None)

        # BERTScore
        if refs and not args.no_bertscore:
            print(f"  BERTScore for {label}…", end=" ", flush=True)
            b = compute_bertscore(preds, refs)
            bert_vals.append(b)
            print(f"{b:.4f}" if b is not None else "N/A")
        else:
            bert_vals.append(None)

        # QA scores
        qa = load_qa(str(qa_path))
        macro_vals.append(qa.get("macro_qa_score"))
        micro_vals.append(qa.get("micro_qa_score"))

        print(f"  {label}: ROUGE-L={rouge_vals[-1]:.4f if rouge_vals[-1] else 'N/A'}  "
              f"Macro QA={macro_vals[-1] or 'N/A'}  Micro QA={micro_vals[-1] or 'N/A'}")

    if not x_labels:
        print("No data found. Run run_chunk_sweep.sh first.")
        return

    # ---- Build metrics dict -----------------------------------------------
    metrics = {}
    if any(v is not None for v in rouge_vals):
        metrics["ROUGE-L"]   = rouge_vals
    if any(v is not None for v in bert_vals):
        metrics["BERTScore"] = bert_vals
    if any(v is not None for v in macro_vals):
        metrics["Macro QA"]  = macro_vals
    if any(v is not None for v in micro_vals):
        metrics["Micro QA"]  = micro_vals

    if not metrics:
        print("No metrics to plot. Provide --narration or run qa_eval.py.")
        return

    # ---- Graph 2: Combined metrics ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_combined_metrics(
        ax, x_labels, metrics,
        xlabel="Chunk size (minutes)",
        title="Caption Quality vs. Chunk Size  (30-min videos)\n"
              "ROUGE-L · BERTScore · Macro QA · Micro QA",
    )
    fig.tight_layout()
    out = f"{plot_dir}/chunk_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ---- Summary table ----------------------------------------------------
    print()
    print("=" * 68)
    print(f"{'Chunk':>8}  {'ROUGE-L':>9}  {'BERTScore':>10}  "
          f"{'Macro QA':>9}  {'Micro QA':>9}")
    print("-" * 68)
    for i, lbl in enumerate(x_labels):
        r  = f"{rouge_vals[i]:.4f}"  if rouge_vals[i]  is not None else "N/A"
        b  = f"{bert_vals[i]:.4f}"   if bert_vals[i]   is not None else "N/A"
        ma = f"{macro_vals[i]:.4f}"  if macro_vals[i]  is not None else "N/A"
        mi = f"{micro_vals[i]:.4f}"  if micro_vals[i]  is not None else "N/A"
        print(f"  {lbl:>6}  {r:>9}  {b:>10}  {ma:>9}  {mi:>9}")
    print("=" * 68)


if __name__ == "__main__":
    main()
