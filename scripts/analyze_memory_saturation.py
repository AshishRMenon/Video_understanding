"""
Analyzes MA-LMM memory saturation from inference results.

Generates:
  1. Memory diversity score per video (lower = more saturated)
  2. Caption temporal coverage analysis (does caption mention early events?)
  3. Comparison across videos of different lengths/densities
  4. Plots: diversity vs video length, diversity vs eventfulness

Usage:
    python scripts/analyze_memory_saturation.py \
        --results_dir outputs/exp01_memory_saturation \
        --out_dir outputs/exp01_memory_saturation/analysis
"""

import os
import json
import argparse
from pathlib import Path


def load_results(results_dir: str) -> list:
    results = []
    for fpath in sorted(Path(results_dir).glob("*_result.json")):
        with open(fpath) as f:
            results.append(json.load(f))
    return results


def temporal_coverage_score(caption: str, early_keywords: list, late_keywords: list) -> dict:
    """
    Rough proxy: how many early-frame keywords appear in the caption?
    (In practice, manually annotate these after reviewing the video.)
    """
    cap_lower = caption.lower()
    early_hits = sum(1 for kw in early_keywords if kw.lower() in cap_lower)
    late_hits = sum(1 for kw in late_keywords if kw.lower() in cap_lower)
    return {
        "early_keyword_hits": early_hits,
        "late_keyword_hits": late_hits,
        "temporal_bias": "late" if late_hits > early_hits else ("early" if early_hits > late_hits else "balanced"),
    }


def print_saturation_report(results: list):
    print("\n" + "=" * 80)
    print("MEMORY SATURATION ANALYSIS REPORT")
    print("=" * 80)

    print(f"\n{'Video ID':<30} {'Avail Frames':<14} {'Sampled':<10} {'Diversity':<12} {'Temporal Bias'}")
    print("-" * 80)

    for r in results:
        vid = r.get("video_id", "?")
        avail = r.get("n_frames_available", "?")
        sampled = r.get("num_frames_processed", "?")
        ms = r.get("memory_stats", {})
        diversity = ms.get("diversity_score", "?")
        # Simple temporal bias from caption length as proxy (if GT available)
        # For now just show diversity
        print(f"{vid:<30} {str(avail):<14} {str(sampled):<10} {str(diversity):<12}")

    print("\n--- CAPTIONS ---")
    for r in results:
        print(f"\n[{r.get('video_id')}]")
        print(f"  Generated: {r.get('caption', 'N/A')[:300]}")
        if r.get("gt_caption"):
            print(f"  GT:        {r['gt_caption'][:300]}")

    print("\n--- KEY OBSERVATIONS ---")
    diversities = [r.get("memory_stats", {}).get("diversity_score", None)
                   for r in results if r.get("memory_stats")]
    diversities = [d for d in diversities if d is not None]
    if diversities:
        print(f"  Average memory diversity: {sum(diversities)/len(diversities):.3f}")
        print(f"  Min diversity (most saturated): {min(diversities):.3f}")
        print(f"  Max diversity: {max(diversities):.3f}")
        if min(diversities) < 0.3:
            print("  >> HIGH SATURATION DETECTED: At least one video shows diversity < 0.3")
            print("  >> This supports the hypothesis that memory-based methods fail on long/sparse video")

    print("\n--- COVERAGE CHECK ---")
    print("  Manual inspection needed: Review captions for early-event coverage.")
    print("  For each video: does the generated caption mention events from the first 10 minutes?")
    print("  If not, that is direct evidence of memory saturation / recency bias.")


def save_latex_table(results: list, out_path: str):
    """Saves a LaTeX table for the paper."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{MA-LMM Memory Saturation on Ego4D Hour-Long Videos}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Video ID & Duration (min) & Frames & Sampled & Diversity & GPU (GB) \\\\",
        "\\midrule",
    ]
    for r in results:
        ms = r.get("memory_stats", {})
        lines.append(
            f"{r.get('video_id', '?')} & ? & {r.get('n_frames_available','?')} & "
            f"{r.get('num_frames_processed','?')} & {ms.get('diversity_score','?')} & "
            f"{r.get('gpu_peak_memory_gb','?')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/exp01_memory_saturation")
    parser.add_argument("--out_dir", default="outputs/exp01_memory_saturation/analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = load_results(args.results_dir)
    if not results:
        print("No result files found. Run run_inference_ego4d.py first.")
        return

    print(f"Loaded {len(results)} results")
    print_saturation_report(results)
    save_latex_table(results, os.path.join(args.out_dir, "saturation_table.tex"))

    # Save consolidated JSON
    with open(os.path.join(args.out_dir, "saturation_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nConsolidated analysis saved to {args.out_dir}/saturation_analysis.json")


if __name__ == "__main__":
    main()
