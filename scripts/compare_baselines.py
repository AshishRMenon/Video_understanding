"""
Side-by-side comparison of MA-LMM vs VideoRecap on the same videos.

Loads result JSONs from both models and produces:
  1. Console comparison table
  2. outputs/comparison/comparison_report.json
  3. outputs/comparison/comparison_table.tex  (for paper)

Usage:
    python3 scripts/compare_baselines.py \
        --malmm_dir outputs/exp01_memory_saturation \
        --vidrecap_dir outputs/vidrecap \
        --out_dir outputs/comparison \
        --narration_json data/ego4d/annotations/ego4d_5videos_annotations.json
"""

import argparse
import json
import os
from pathlib import Path


def load_results(result_dir: str, caption_key: str = "caption") -> dict:
    """Load all *_result.json files from a directory. Returns {video_id: result}."""
    results = {}
    for fpath in sorted(Path(result_dir).glob("*_result.json")):
        with open(fpath) as f:
            d = json.load(f)
        vid = d.get("video_id", fpath.stem.replace("_result", ""))
        d["caption"] = d.get(caption_key) or d.get("video_caption") or d.get("caption", "N/A")
        results[vid] = d
    return results


def load_narrations(narration_json: str) -> dict:
    """
    Load filtered narration JSON (from download_ego4d_5videos.py).
    Returns {video_id: [narration_text, ...]} — first 10 narrations as a proxy GT.
    """
    if not narration_json or not Path(narration_json).exists():
        return {}
    with open(narration_json) as f:
        data = json.load(f)
    gt = {}
    for uid, info in data.items():
        narrs = [n["narration_text"] for n in info.get("narrations", [])[:10]]
        gt[uid] = narrs
    return gt


def temporal_coverage(caption: str, narrations: list) -> dict:
    """
    Rough measure: how many narration keywords appear in the caption?
    Split narrations into early (first half) and late (second half).
    """
    if not narrations:
        return {"early_hits": "N/A", "late_hits": "N/A", "bias": "N/A"}

    mid = len(narrations) // 2
    early_narrs = narrations[:mid]
    late_narrs = narrations[mid:]

    cap_lower = caption.lower()

    def keyword_hits(narr_list):
        hits = 0
        for narr in narr_list:
            words = [w.strip(".,!?").lower() for w in narr.split() if len(w) > 4]
            hits += sum(1 for w in words if w in cap_lower)
        return hits

    early = keyword_hits(early_narrs)
    late = keyword_hits(late_narrs)
    bias = "late" if late > early else ("early" if early > late else "balanced")
    return {"early_hits": early, "late_hits": late, "bias": bias}


def print_comparison(malmm: dict, vidrecap: dict, gt: dict):
    all_vids = sorted(set(malmm) | set(vidrecap))

    print("\n" + "=" * 90)
    print("BASELINE COMPARISON: MA-LMM  vs  VideoRecap")
    print("=" * 90)

    for vid in all_vids:
        m = malmm.get(vid)
        v = vidrecap.get(vid)
        narrations = gt.get(vid, [])

        print(f"\n{'─'*90}")
        print(f"VIDEO: {vid}")
        if m:
            print(f"  Duration  : {m.get('n_frames_available','?')} frames available, "
                  f"{m.get('num_frames_processed','?')} processed")
        print(f"  Narrations: {len(narrations)} (ground-truth narrations available)")

        print(f"\n  ── MA-LMM ──────────────────────────────────────────────────────────")
        if m:
            cap = m.get("caption", "N/A")
            ms = m.get("memory_stats", {})
            diversity = ms.get("diversity_score", "N/A")
            gpu_gb = m.get("gpu_peak_memory_gb", "N/A")
            tc = temporal_coverage(cap, narrations)
            print(f"  Caption   : {cap[:300]}")
            print(f"  Diversity : {diversity}  (lower = more saturated)")
            print(f"  GPU Peak  : {gpu_gb} GB")
            print(f"  Temp bias : early_hits={tc['early_hits']}  late_hits={tc['late_hits']}  → {tc['bias']}")
        else:
            print("  [no results]")

        print(f"\n  ── VideoRecap ──────────────────────────────────────────────────────")
        if v:
            cap = v.get("caption", "N/A")
            n_clips = v.get("n_clips", "?")
            n_segs = v.get("n_segments", "?")
            tc = temporal_coverage(cap, narrations)
            print(f"  Caption   : {cap[:300]}")
            print(f"  Hierarchy : {n_clips} clips → {n_segs} segments → 1 video summary")
            print(f"  Temp bias : early_hits={tc['early_hits']}  late_hits={tc['late_hits']}  → {tc['bias']}")
            # Print segment captions too
            for i, seg in enumerate(v.get("segment_captions", [])[:5]):
                print(f"  Seg {i+1}     : [{seg['start']:.0f}s-{seg['end']:.0f}s] {seg['description'][:120]}")
        else:
            print("  [no results]")

    print(f"\n{'='*90}")

    # Summary stats
    print("\nSUMMARY")
    print(f"  Videos with MA-LMM results  : {len(malmm)}")
    print(f"  Videos with VidRecap results: {len(vidrecap)}")
    print(f"  Videos with both            : {len(set(malmm) & set(vidrecap))}")

    diversities = [r.get("memory_stats", {}).get("diversity_score")
                   for r in malmm.values() if r.get("memory_stats")]
    diversities = [d for d in diversities if d is not None]
    if diversities:
        print(f"\n  MA-LMM memory diversity scores:")
        print(f"    Mean : {sum(diversities)/len(diversities):.3f}")
        print(f"    Min  : {min(diversities):.3f}  ← most saturated")
        print(f"    Max  : {max(diversities):.3f}")
        if min(diversities) < 0.3:
            print("    >> HIGH SATURATION: supports our hypothesis")

    print()


def save_latex_table(malmm: dict, vidrecap: dict, out_path: str):
    all_vids = sorted(set(malmm) | set(vidrecap))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{MA-LMM vs.\ VideoRecap on Ego4D Hour-Long Videos. "
        r"Diversity score: 1=distinct memory slots, 0=fully saturated.}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Video & Duration & \multicolumn{2}{c}{MA-LMM} & \multicolumn{2}{c}{VideoRecap} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r" & (min) & Diversity$\uparrow$ & Bias & Segs & Bias \\",
        r"\midrule",
    ]
    for vid in all_vids:
        m = malmm.get(vid, {})
        v = vidrecap.get(vid, {})
        dur = "?"
        if m.get("n_frames_available"):
            dur = str(round(int(m["n_frames_available"]) / 60, 0))
        m_div = m.get("memory_stats", {}).get("diversity_score", "—")
        m_div = f"{m_div:.2f}" if isinstance(m_div, float) else str(m_div)
        v_segs = v.get("n_segments", "—")
        lines.append(
            f"{vid[:20]} & {dur} & {m_div} & — & {v_segs} & — \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--malmm_dir", default="outputs/exp01_memory_saturation")
    parser.add_argument("--vidrecap_dir", default="outputs/vidrecap")
    parser.add_argument("--out_dir", default="outputs/comparison")
    parser.add_argument("--narration_json",
                        default="data/ego4d/annotations/ego4d_5videos_annotations.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    malmm = load_results(args.malmm_dir)
    vidrecap = load_results(args.vidrecap_dir, caption_key="video_caption")
    gt = load_narrations(args.narration_json)

    if not malmm and not vidrecap:
        print("No results found. Run both inference scripts first.")
        return

    print_comparison(malmm, vidrecap, gt)

    # Save JSON report
    report = {}
    for vid in sorted(set(malmm) | set(vidrecap)):
        narrations = gt.get(vid, [])
        m = malmm.get(vid, {})
        v = vidrecap.get(vid, {})
        report[vid] = {
            "malmm_caption": m.get("caption", "N/A"),
            "malmm_diversity": m.get("memory_stats", {}).get("diversity_score"),
            "malmm_temporal_bias": temporal_coverage(m.get("caption", ""), narrations)["bias"],
            "vidrecap_caption": v.get("caption", "N/A"),
            "vidrecap_n_segments": v.get("n_segments"),
            "vidrecap_temporal_bias": temporal_coverage(v.get("caption", ""), narrations)["bias"],
            "vidrecap_segment_captions": v.get("segment_captions", []),
        }

    report_path = os.path.join(args.out_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Comparison report saved to {report_path}")

    save_latex_table(malmm, vidrecap, os.path.join(args.out_dir, "comparison_table.tex"))


if __name__ == "__main__":
    main()
