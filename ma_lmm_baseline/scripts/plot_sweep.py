"""
Experiment 2 — Combined metrics graph: caption quality vs. (num_frames, memory_bank_length).

Produces ONE graph with four metric lines on the same axes:
  • ROUGE-L    — lexical overlap with ground-truth summaries  (requires --narration)
  • BERTScore  — semantic similarity with ground-truth         (requires --narration)
  • Macro QA   — main-theme coverage score                    (requires qa_scores.json)
  • Micro QA   — fine-grained detail coverage score           (requires qa_scores.json)

X-axis: config label "(num_frames, MBL)"   e.g. "(100, 40)" → "(1000, 300)"
Y-axis: score (0–1)

Usage:
    # All metrics:
    python scripts/plot_sweep.py \\
        --sweep_dir outputs/exp2_frame_sweep \\
        --plot_dir  outputs/exp2_frame_sweep/plots \\
        --narration /workspace/ego4d_meta/v2/annotations/narration.json \\
        --ann_path  data/annotations/ego4d_10min.json

    # qa_scores.json is auto-detected from {sweep_dir}/qa_scores.json
    # Run qa_eval.py first to generate it.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Consistent colours + markers across both graphs
METRIC_STYLE = {
    "ROUGE-L":   {"color": "#2196F3", "marker": "o", "ls": "-"},
    "BERTScore": {"color": "#4CAF50", "marker": "s", "ls": "-"},
    "Macro QA":  {"color": "#FF9800", "marker": "^", "ls": "-"},
    "Micro QA":  {"color": "#E91E63", "marker": "D", "ls": "-"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweep_results(sweep_dir):
    results = []
    for fpath in sorted(Path(sweep_dir).glob("captions_f*_m*.json")):
        parts = fpath.stem.split("_")
        try:
            nf  = int(parts[1][1:])
            mbl = int(parts[2][1:])
        except (IndexError, ValueError):
            continue
        with open(fpath) as f:
            records = json.load(f)
        results.append((nf, mbl, records))
    return sorted(results, key=lambda x: (x[0], x[1]))


def load_ground_truth(narration_path, ann_path):
    """
    {video_id: reference_text} from Ego4D segment *summaries*.
    Summaries are high-level (5-min blocks) — a much better ROUGE / BERTScore
    reference for paragraph-length captions than second-by-second narrations.
    """
    with open(ann_path) as f:
        anns = json.load(f)
    with open(narration_path) as f:
        narr_data = json.load(f)

    gt = {}
    for ann in anns:
        vid     = ann["video_id"]
        max_sec = ann.get("n_frames", 0) / 10
        summaries = (narr_data.get(vid, {})
                     .get("narration_pass_1", {})
                     .get("summaries", []))
        texts = [
            re.sub(r"#\w+\s*", "", s["summary_text"]).strip()
            for s in summaries
            if s.get("start_sec", 9999) < max_sec and s.get("summary_text", "").strip()
        ]
        if texts:
            gt[vid] = " ".join(texts)
    return gt


def load_qa_scores(qa_scores_path):
    with open(qa_scores_path) as f:
        data = json.load(f)
    mapping = {}
    for cfg in data.get("configs", []):
        key = (cfg["num_frames"], cfg["memory_bank_length"])
        mapping[key] = {
            "macro_qa_score": cfg.get("macro_qa_score"),
            "micro_qa_score": cfg.get("micro_qa_score"),
        }
    return mapping


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_rouge_l(predictions, references):
    try:
        from rouge_score import rouge_scorer as rs
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rouge-score"])
        from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        if pred and ref:
            scores.append(scorer.score(ref, pred)["rougeL"].fmeasure)
    return float(np.mean(scores)) if scores else None


def compute_bertscore(predictions, references):
    try:
        from bert_score import score as bs
    except ImportError:
        print("  Installing bert-score (first run downloads roberta-large ~500 MB)…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bert-score"])
        from bert_score import score as bs
    valid = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not valid:
        return None
    preds, refs = zip(*valid)
    _, _, F = bs(list(preds), list(refs), lang="en",
                 model_type="roberta-large", verbose=False)
    return float(F.mean())


# ---------------------------------------------------------------------------
# Combined metrics plot  (shared by both sweep and chunk-sweep)
# ---------------------------------------------------------------------------

def plot_combined_metrics(ax, x_labels, metrics_dict, xlabel, title):
    """
    x_labels    : list of N strings for the X-axis ticks
    metrics_dict: {metric_name: [v0, v1, …, vN-1]}  (None = no data for that point)
    """
    x = np.arange(len(x_labels))
    for name, vals in metrics_dict.items():
        style = METRIC_STYLE.get(name, {"color": "grey", "marker": "o", "ls": "-"})
        x_plot = [x[i] for i, v in enumerate(vals) if v is not None]
        y_plot = [v        for v in vals           if v is not None]
        if not x_plot:
            continue
        ax.plot(x_plot, y_plot,
                color=style["color"], marker=style["marker"],
                linestyle=style["ls"], linewidth=2, markersize=8,
                label=name)
        for xi, yi in zip(x_plot, y_plot):
            ax.annotate(f"{yi:.3f}", (xi, yi),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color=style["color"])

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score  (0 – 1)")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default="outputs/exp2_frame_sweep")
    ap.add_argument("--plot_dir",  default="outputs/exp2_frame_sweep/plots")
    ap.add_argument("--narration", default=None,
                    help="Ego4D narration.json  (enables ROUGE-L and BERTScore)")
    ap.add_argument("--ann_path",  default=None,
                    help="Annotation JSON  (e.g. data/annotations/ego4d_10min.json)")
    ap.add_argument("--qa_scores", default=None,
                    help="qa_scores.json from qa_eval.py  (auto-detected if omitted)")
    ap.add_argument("--no_bertscore", action="store_true",
                    help="Skip BERTScore (saves time; omits the model download)")
    args = ap.parse_args()

    # Auto-detect qa_scores.json
    if args.qa_scores is None:
        default_qa = Path(args.sweep_dir) / "qa_scores.json"
        if default_qa.exists():
            args.qa_scores = str(default_qa)
            print(f"Auto-detected: {args.qa_scores}")

    configs = load_sweep_results(args.sweep_dir)
    if not configs:
        print(f"No results in {args.sweep_dir}. Run run_frame_sweep.sh first.")
        return

    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    # Ground truth for ROUGE-L / BERTScore
    gt = {}
    if args.narration and args.ann_path:
        gt = load_ground_truth(args.narration, args.ann_path)
        print(f"Ground-truth summaries loaded for {len(gt)} video(s).")

    # QA scores
    qa_map = {}
    if args.qa_scores:
        qa_map = load_qa_scores(args.qa_scores)
        print(f"QA scores loaded for {len(qa_map)} config(s).")

    # ---- Collect per-config metrics ----------------------------------------
    x_labels   = []
    rouge_vals  = []
    bert_vals   = []
    macro_vals  = []
    micro_vals  = []

    for nf, mbl, recs in configs:
        label = f"({nf}, {mbl})"
        x_labels.append(label)

        # ROUGE-L
        if gt:
            preds = [r.get("caption", "") for r in recs]
            refs  = [gt.get(r["video_id"], "") for r in recs]
            rouge_vals.append(compute_rouge_l(preds, refs))
        else:
            rouge_vals.append(None)

        # BERTScore
        if gt and not args.no_bertscore:
            preds = [r.get("caption", "") for r in recs]
            refs  = [gt.get(r["video_id"], "") for r in recs]
            print(f"  BERTScore for {label}…", end=" ", flush=True)
            b = compute_bertscore(preds, refs)
            bert_vals.append(b)
            print(f"{b:.4f}" if b is not None else "N/A")
        else:
            bert_vals.append(None)

        # QA metrics
        qa_entry = qa_map.get((nf, mbl), {})
        macro_vals.append(qa_entry.get("macro_qa_score"))
        micro_vals.append(qa_entry.get("micro_qa_score"))

    # ---- Build metrics dict (only include metrics that have at least one value)
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
        print("No metrics available to plot. Provide --narration / run qa_eval.py.")
        return

    # ---- Graph 1: Combined metrics -----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_combined_metrics(
        ax, x_labels, metrics,
        xlabel="(num_frames,  memory_bank_length)",
        title="Caption Quality vs. Frame Budget\n"
              "ROUGE-L · BERTScore · Macro QA · Micro QA",
    )
    fig.tight_layout()
    out = f"{args.plot_dir}/metrics_vs_frames.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ---- Summary table -----------------------------------------------------
    print()
    print("=" * 72)
    print(f"{'Config':<16}  {'ROUGE-L':>9}  {'BERTScore':>10}  "
          f"{'Macro QA':>9}  {'Micro QA':>9}")
    print("-" * 72)
    for i, lbl in enumerate(x_labels):
        r  = f"{rouge_vals[i]:.4f}"  if rouge_vals[i]  is not None else "N/A"
        b  = f"{bert_vals[i]:.4f}"   if bert_vals[i]   is not None else "N/A"
        ma = f"{macro_vals[i]:.4f}"  if macro_vals[i]  is not None else "N/A"
        mi = f"{micro_vals[i]:.4f}"  if micro_vals[i]  is not None else "N/A"
        print(f"  {lbl:<14}  {r:>9}  {b:>10}  {ma:>9}  {mi:>9}")
    print("=" * 72)


if __name__ == "__main__":
    main()
