"""
Experiment 2 — Visualize the frame-budget vs quality / compute trade-off.

Reads all JSON files produced by run_frame_sweep.sh and generates four plots:

  1. inference_time.png  — wall-clock time vs num_frames, one line per MBL
  2. gpu_memory.png      — GPU peak memory vs num_frames, one line per MBL
  3. caption_quality.png — ROUGE-L vs num_frames (requires ground-truth narrations)
  4. quality_vs_compute.png — Pareto scatter: ROUGE-L vs inference time (per config)

Also prints a summary table to stdout for copy-paste into a paper appendix.

Usage (compute metrics only, no ground truth):
    python scripts/plot_sweep.py \
        --sweep_dir outputs/exp2_frame_sweep \
        --plot_dir  outputs/exp2_frame_sweep/plots

Usage (with caption quality; requires Ego4D narration.json):
    python scripts/plot_sweep.py \
        --sweep_dir  outputs/exp2_frame_sweep \
        --plot_dir   outputs/exp2_frame_sweep/plots \
        --narration  /path/to/ego4d/v2/annotations/narration.json \
        --ann_path   data/annotations/ego4d_10min.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PALETTE = plt.cm.tab10.colors


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweep_results(sweep_dir):
    """
    Return list of (num_frames, mbl, records) tuples.
    Filenames are expected to be captions_f{NF}_m{MBL}.json.
    """
    results = []
    for fpath in sorted(Path(sweep_dir).glob("captions_f*_m*.json")):
        stem = fpath.stem  # e.g. captions_f200_m80
        parts = stem.split("_")
        try:
            nf  = int(parts[1][1:])
            mbl = int(parts[2][1:])
        except (IndexError, ValueError):
            print(f"[skip] unrecognised filename: {fpath.name}")
            continue
        with open(fpath) as f:
            records = json.load(f)
        results.append((nf, mbl, records))
    return sorted(results, key=lambda x: (x[0], x[1]))


def _mean(records, key):
    vals = [r[key] for r in records if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_rouge(predictions, references):
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = defaultdict(list)
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        s = scorer.score(ref, pred)
        for k, v in s.items():
            scores[k].append(v.fmeasure)
    return {k: float(np.mean(v)) for k, v in scores.items()}


def load_ground_truth(narration_path, ann_path):
    """Build video_id → concatenated narration string from Ego4D narration.json."""
    with open(ann_path) as f:
        anns = json.load(f)
    with open(narration_path) as f:
        narr_data = json.load(f)

    gt = {}
    for ann in anns:
        vid = ann["video_id"]
        max_sec = ann.get("n_frames", 0) / 10  # assumes 10 fps extraction
        vid_narr = narr_data.get(vid, {})
        narrations = vid_narr.get("narration_pass_1", {}).get("narrations", [])
        texts = [
            n["narration_text"].lstrip("#").strip()
            for n in narrations
            if n.get("timestamp_sec", 9999) < max_sec
        ]
        if texts:
            gt[vid] = " ".join(texts)
    return gt


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _line_plot(ax, data_by_mbl, xlabel, ylabel, title):
    """data_by_mbl: {mbl: [(nf, val), ...]}"""
    for i, mbl in enumerate(sorted(data_by_mbl)):
        pts = sorted(data_by_mbl[mbl])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys,
            marker="o",
            color=PALETTE[i % len(PALETTE)],
            label=f"MBL={mbl}",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="memory_bank_length")
    ax.grid(True, alpha=0.3)


def _scatter_quality_compute(ax, records_meta, qual_map):
    """
    Pareto scatter: x = avg inference time, y = ROUGE-L.
    Each point is one (nf, mbl) config; label shows the config.
    """
    for nf, mbl, t_avg, q in records_meta:
        if t_avg is None or q is None:
            continue
        ax.scatter(t_avg, q, s=80, zorder=3)
        ax.annotate(
            f"F{nf}/M{mbl}",
            (t_avg, q),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )
    ax.set_xlabel("Avg. inference time (sec)")
    ax.set_ylabel("ROUGE-L (F1)")
    ax.set_title("Quality vs Compute — Pareto frontier")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir",  default="outputs/exp2_frame_sweep",
                    help="Directory containing captions_f*_m*.json files")
    ap.add_argument("--plot_dir",   default="outputs/exp2_frame_sweep/plots",
                    help="Directory to write PNG figures")
    ap.add_argument("--narration",  default=None,
                    help="Ego4D narration.json path (enables quality metrics)")
    ap.add_argument("--ann_path",   default=None,
                    help="Annotation JSON used for the sweep (e.g. ego4d_10min.json)")
    args = ap.parse_args()

    configs = load_sweep_results(args.sweep_dir)
    if not configs:
        print(f"No sweep results found in {args.sweep_dir}. Run run_frame_sweep.sh first.")
        return

    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    # Ground truth (optional)
    gt = {}
    if args.narration and args.ann_path:
        gt = load_ground_truth(args.narration, args.ann_path)
        print(f"Loaded ground-truth for {len(gt)} video(s).")

    # ---- Aggregate compute metrics ----------------------------------------
    time_by_mbl: dict = defaultdict(list)
    mem_by_mbl:  dict = defaultdict(list)
    qual_by_mbl: dict = defaultdict(list)
    pareto_pts = []

    for nf, mbl, recs in configs:
        t_avg = _mean(recs, "inference_time_sec")
        m_avg = _mean(recs, "gpu_peak_memory_gb")

        if t_avg is not None:
            time_by_mbl[mbl].append((nf, t_avg))
        if m_avg is not None:
            mem_by_mbl[mbl].append((nf, m_avg))

        rouge_l = None
        if gt:
            preds  = [r["caption"] for r in recs]
            refs   = [gt.get(r["video_id"], "") for r in recs]
            rouge  = compute_rouge(preds, refs)
            rouge_l = rouge.get("rougeL")
            if rouge_l is not None:
                qual_by_mbl[mbl].append((nf, rouge_l))

        pareto_pts.append((nf, mbl, t_avg, rouge_l))

    # ---- Plot 1: Inference time -------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    _line_plot(ax, time_by_mbl,
               xlabel="num_frames",
               ylabel="Avg. inference time (sec)",
               title="Inference Time vs. Number of Input Frames")
    fig.tight_layout()
    fig.savefig(f"{args.plot_dir}/inference_time.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {args.plot_dir}/inference_time.png")

    # ---- Plot 2: GPU peak memory -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    _line_plot(ax, mem_by_mbl,
               xlabel="num_frames",
               ylabel="GPU peak memory (GB)",
               title="GPU Memory vs. Number of Input Frames")
    fig.tight_layout()
    fig.savefig(f"{args.plot_dir}/gpu_memory.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {args.plot_dir}/gpu_memory.png")

    # ---- Plot 3: Caption quality (ROUGE-L) --------------------------------
    if qual_by_mbl:
        fig, ax = plt.subplots(figsize=(7, 4))
        _line_plot(ax, qual_by_mbl,
                   xlabel="num_frames",
                   ylabel="ROUGE-L (F1)",
                   title="Caption Quality vs. Number of Input Frames")
        fig.tight_layout()
        fig.savefig(f"{args.plot_dir}/caption_quality.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {args.plot_dir}/caption_quality.png")

        # ---- Plot 4: Pareto scatter (quality vs compute) ------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        _scatter_quality_compute(ax, pareto_pts, {})
        fig.tight_layout()
        fig.savefig(f"{args.plot_dir}/quality_vs_compute.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {args.plot_dir}/quality_vs_compute.png")
    else:
        print("Quality plots skipped — no ground truth provided.\n"
              "  Pass --narration and --ann_path to enable them.")

    # ---- Summary table ---------------------------------------------------
    print()
    print("=" * 72)
    print(f"{'num_frames':>12}  {'MBL':>6}  {'time(s)':>10}  "
          f"{'mem(GB)':>9}  {'ROUGE-L':>8}")
    print("-" * 72)
    for nf, mbl, t_avg, rouge_l in sorted(pareto_pts, key=lambda x: (x[0], x[1])):
        t_str = f"{t_avg:.1f}"   if t_avg   is not None else "N/A"
        q_str = f"{rouge_l:.4f}" if rouge_l is not None else "N/A"
        m_vals = [v for (nf2, v) in mem_by_mbl.get(mbl, []) if nf2 == nf]
        m_str  = f"{m_vals[0]:.2f}" if m_vals else "N/A"
        print(f"{nf:>12}  {mbl:>6}  {t_str:>10}  {m_str:>9}  {q_str:>8}")
    print("=" * 72)


if __name__ == "__main__":
    main()
