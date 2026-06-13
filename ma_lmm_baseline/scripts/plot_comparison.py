"""
Experiment 3 — Plot chunked vs. full-video QA score comparison (macro + micro).

Reads chunked_qa.json and full_qa.json produced by run_comparison.sh and
generates:
  1. qa_comparison.png  — grouped bars: macro QA and micro QA for each approach,
                          per-video and overall
  2. Prints a side-by-side summary table to stdout

Usage:
    python scripts/plot_comparison.py \
        --chunked_qa outputs/exp3_comparison/chunked_qa.json \
        --full_qa    outputs/exp3_comparison/full_qa.json \
        --plot_dir   outputs/exp3_comparison/plots \
        --chunk_mins 2 \
        --best_nf    200 \
        --best_mbl   160
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Colour scheme: blue family for chunked, orange family for full-video
C_MACRO = "#4C72B0"   # chunked macro
C_MICRO = "#9DB8D9"   # chunked micro (lighter)
F_MACRO = "#DD8452"   # full-video macro
F_MICRO = "#F5BC8A"   # full-video micro (lighter)


def load_qa(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _score(d, key):
    v = d.get(key)
    return v if v is not None else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunked_qa", required=True)
    ap.add_argument("--full_qa",    required=True)
    ap.add_argument("--plot_dir",   default="outputs/exp3_comparison/plots")
    ap.add_argument("--chunk_mins", default="2")
    ap.add_argument("--best_nf",    default="200")
    ap.add_argument("--best_mbl",   default="160")
    args = ap.parse_args()

    chunked = load_qa(args.chunked_qa)
    full    = load_qa(args.full_qa)

    label_c = f"Chunked ({args.chunk_mins}-min)"
    label_f = f"Full-video (F{args.best_nf}/M{args.best_mbl})"

    # Index per-video scores
    chunked_per = {r["video_id"]: r for r in chunked.get("per_video", [])}
    full_per    = {r["video_id"]: r for r in full.get("per_video", [])}
    common_vids = sorted(set(chunked_per) & set(full_per))

    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    # ---- Build data arrays -----------------------------------------------
    # Groups: AVERAGE first, then per-video
    group_labels = ["AVERAGE"] + [v[:8] + "…" for v in common_vids]
    n_groups = len(group_labels)

    def scores_for(qa_data, per_data, vids, key):
        overall = [_score(qa_data, key)]
        per     = [_score(per_data.get(v, {}), key.replace("qa_score", "score")
                          .replace("macro_qa_score", "macro_score")
                          .replace("micro_qa_score", "micro_score"))
                   for v in vids]
        return overall + per

    c_macro = scores_for(chunked, chunked_per, common_vids, "macro_qa_score")
    c_micro = scores_for(chunked, chunked_per, common_vids, "micro_qa_score")
    f_macro = scores_for(full,    full_per,    common_vids, "macro_qa_score")
    f_micro = scores_for(full,    full_per,    common_vids, "micro_qa_score")

    # ---- Grouped bar chart -----------------------------------------------
    x = np.arange(n_groups)
    w = 0.2   # bar width; 4 bars per group

    fig, ax = plt.subplots(figsize=(max(9, 2.2 * n_groups), 5))

    b1 = ax.bar(x - 1.5*w, c_macro, w, label=f"{label_c} — Macro", color=C_MACRO, alpha=0.9)
    b2 = ax.bar(x - 0.5*w, c_micro, w, label=f"{label_c} — Micro", color=C_MICRO, alpha=0.9)
    b3 = ax.bar(x + 0.5*w, f_macro, w, label=f"{label_f} — Macro", color=F_MACRO, alpha=0.9)
    b4 = ax.bar(x + 1.5*w, f_micro, w, label=f"{label_f} — Micro", color=F_MICRO, alpha=0.9)

    for bar in list(b1) + list(b2) + list(b3) + list(b4):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    # Shade the AVERAGE column
    ax.axvspan(-0.5, 0.5, color="lightgrey", alpha=0.25, zorder=0)
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("QA Score (fraction of questions answered)")
    ax.set_title(
        f"Macro vs. Micro QA: {label_c}  vs.  {label_f}\n"
        f"(model: {chunked.get('model','?')}, "
        f"{chunked.get('n_macro','?')} macro + {chunked.get('n_micro','?')} micro questions/video)"
    )
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = f"{args.plot_dir}/qa_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ---- Summary table ---------------------------------------------------
    col = 30
    print()
    print("=" * 88)
    print(f"{'VIDEO':<{col}}  {'MACRO':^21}  {'MICRO':^21}  {'OVERALL':^10}")
    print(f"{'':30}  {'chunk':>9}  {'full':>9}  {'chunk':>9}  {'full':>9}  {'chunk':>4}  {'full':>4}")
    print("-" * 88)

    for vid in common_vids:
        cr = chunked_per.get(vid, {})
        fr = full_per.get(vid, {})
        cm = _score(cr, "macro_score");  fm = _score(fr, "macro_score")
        ci = _score(cr, "micro_score");  fi = _score(fr, "micro_score")
        co = _score(cr, "score");        fo = _score(fr, "score")
        print(f"  {vid[:col-2]:<{col-2}}  "
              f"{cm:>9.4f}  {fm:>9.4f}  "
              f"{ci:>9.4f}  {fi:>9.4f}  "
              f"{co:>4.2f}  {fo:>4.2f}")

    print("-" * 88)
    cm_avg = _score(chunked, "macro_qa_score"); fm_avg = _score(full, "macro_qa_score")
    ci_avg = _score(chunked, "micro_qa_score"); fi_avg = _score(full, "micro_qa_score")
    co_avg = _score(chunked, "qa_score");       fo_avg = _score(full, "qa_score")
    print(f"  {'AVERAGE':<{col-2}}  "
          f"{cm_avg:>9.4f}  {fm_avg:>9.4f}  "
          f"{ci_avg:>9.4f}  {fi_avg:>9.4f}  "
          f"{co_avg:>4.2f}  {fo_avg:>4.2f}")
    print("=" * 88)

    # Winner summary
    print()
    for metric, c_val, f_val, name in [
        ("Macro QA", cm_avg, fm_avg, "macro"),
        ("Micro QA", ci_avg, fi_avg, "micro"),
        ("Overall",  co_avg, fo_avg, "overall"),
    ]:
        delta = f_val - c_val
        winner = label_f if delta > 0.01 else (label_c if delta < -0.01 else "tie")
        print(f"  {metric:<12}: {label_c} {c_val:.4f}  vs  {label_f} {f_val:.4f}"
              f"  →  {winner} wins by {abs(delta):.4f}" if winner != "tie"
              else f"  {metric:<12}: {label_c} {c_val:.4f}  vs  {label_f} {f_val:.4f}"
              f"  →  tie")


if __name__ == "__main__":
    main()
