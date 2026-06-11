"""
Compare MALMM (infer.py) vs two-stage (infer_v2.py) against Ego4D ground truth.

Ground truth: narration_pass_1 segment summaries from ego4d narration.json,
filtered to the video's actual duration and concatenated into one reference.

Metrics: ROUGE-1/2/L, BERTScore (roberta-large).

Usage:
    python scripts/evaluate.py \
        --malmm    outputs_latest_2/captions_10min_malmm.json \
        --twostage outputs_v2_100frames/captions_10min.json \
        --narration /workspace/ego4d_meta/v2/annotations/narration.json \
        --ann_path  data/annotations/ego4d_10min.json
"""

import argparse
import json
import re
import subprocess
import sys


def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Installing rouge-score...")
    install("rouge-score")
    from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score
except ImportError:
    print("Installing bert-score...")
    install("bert-score")
    from bert_score import score as bert_score


def clean_summary(text: str) -> str:
    text = re.sub(r"^#Summary\s*", "", text.strip())
    text = re.sub(r"#unsure", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_ground_truth(narration_data: dict, video_id: str, n_frames: int, fps: int = 10) -> str:
    duration_sec = n_frames / fps
    entry = narration_data.get(video_id, {})
    summaries = entry.get("narration_pass_1", {}).get("summaries", [])
    relevant = [
        clean_summary(s["summary_text"])
        for s in summaries
        if s.get("start_sec", 0) < duration_sec
    ]
    return " ".join(relevant) if relevant else ""


def compute_rouge(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    per_video = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        per_video.append({k: round(v.fmeasure * 100, 2) for k, v in scores.items()})
        for k, v in scores.items():
            agg[k].append(v.fmeasure * 100)
    avg = {k: round(sum(v) / len(v), 2) for k, v in agg.items()}
    return {"per_video": per_video, "avg": avg}


def compute_bertscore(predictions: list, references: list) -> dict:
    print("  Computing BERTScore (roberta-large) — first run downloads the model...")
    P, R, F = bert_score(predictions, references, lang="en", model_type="roberta-large", verbose=False)
    per_video = [round(f.item() * 100, 2) for f in F]
    avg = round(sum(per_video) / len(per_video), 2)
    return {"per_video": per_video, "avg": avg}


def print_table(video_ids, references, malmm_caps, v2_caps, malmm_rouge, v2_rouge, malmm_bs, v2_bs):
    col = 36
    print("\n" + "=" * 110)
    print(f"{'VIDEO':<38} {'METHOD':<12} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'BERTScore':>10}    CAPTION (first 80 chars)")
    print("=" * 110)
    for i, vid in enumerate(video_ids):
        short = vid[:8] + "..."
        ref_preview = references[i][:60] + "..."

        # MALMM row
        m_r = malmm_rouge["per_video"][i]
        m_b = malmm_bs["per_video"][i]
        m_cap = malmm_caps[i][:80]
        print(f"  {short:<36} {'MALMM':<12} {m_r['rouge1']:>6.1f} {m_r['rouge2']:>6.1f} {m_r['rougeL']:>6.1f} {m_b:>10.1f}    {m_cap}")

        # Two-stage row
        v_r = v2_rouge["per_video"][i]
        v_b = v2_bs["per_video"][i]
        v_cap = v2_caps[i][:80]
        print(f"  {'':36} {'Two-stage':<12} {v_r['rouge1']:>6.1f} {v_r['rouge2']:>6.1f} {v_r['rougeL']:>6.1f} {v_b:>10.1f}    {v_cap}")

        print(f"  {'GT ref':<50} {ref_preview}")
        print("-" * 110)

    # Averages
    m_avg = malmm_rouge["avg"]
    v_avg = v2_rouge["avg"]
    print(f"\n{'AVERAGE':<50} {'MALMM':<12} {m_avg['rouge1']:>6.1f} {m_avg['rouge2']:>6.1f} {m_avg['rougeL']:>6.1f} {malmm_bs['avg']:>10.1f}")
    print(f"{'':50} {'Two-stage':<12} {v_avg['rouge1']:>6.1f} {v_avg['rouge2']:>6.1f} {v_avg['rougeL']:>6.1f} {v2_bs['avg']:>10.1f}")
    print("=" * 110)

    # Delta
    print(f"\n{'DELTA (MALMM - Two-stage)':<50} {'':12} "
          f"{m_avg['rouge1']-v_avg['rouge1']:>+6.1f} "
          f"{m_avg['rouge2']-v_avg['rouge2']:>+6.1f} "
          f"{m_avg['rougeL']-v_avg['rougeL']:>+6.1f} "
          f"{malmm_bs['avg']-v2_bs['avg']:>+10.1f}")
    print("  (positive = MALMM wins, negative = Two-stage wins)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malmm",     required=True, help="infer.py output JSON")
    ap.add_argument("--twostage",  required=True, help="infer_v2.py output JSON")
    ap.add_argument("--narration", required=True, help="ego4d narration.json")
    ap.add_argument("--ann_path",  required=True, help="ego4d_10min.json (has n_frames)")
    args = ap.parse_args()

    with open(args.malmm) as f:
        malmm_data = {r["video_id"]: r for r in json.load(f)}
    with open(args.twostage) as f:
        v2_data = {r["video_id"]: r for r in json.load(f)}
    with open(args.narration) as f:
        narration = json.load(f)
    with open(args.ann_path) as f:
        anns = json.load(f)

    # Align on videos present in both outputs
    video_ids, references, malmm_caps, v2_caps = [], [], [], []
    for ann in anns:
        vid = ann["video_id"]
        if vid not in malmm_data or vid not in v2_data:
            print(f"[skip] {vid} missing from one output")
            continue
        gt = build_ground_truth(narration, vid, ann["n_frames"])
        if not gt:
            print(f"[skip] {vid} has no ground-truth summaries in range")
            continue
        video_ids.append(vid)
        references.append(gt)
        malmm_caps.append(malmm_data[vid]["caption"])
        v2_caps.append(v2_data[vid]["caption"])

    print(f"\nEvaluating {len(video_ids)} videos...")
    print("\n[1/4] ROUGE for MALMM...")
    malmm_rouge = compute_rouge(malmm_caps, references)
    print("[2/4] ROUGE for Two-stage...")
    v2_rouge = compute_rouge(v2_caps, references)
    print("[3/4] BERTScore for MALMM...")
    malmm_bs = compute_bertscore(malmm_caps, references)
    print("[4/4] BERTScore for Two-stage...")
    v2_bs = compute_bertscore(v2_caps, references)

    print_table(video_ids, references, malmm_caps, v2_caps, malmm_rouge, v2_rouge, malmm_bs, v2_bs)

    # Save results
    results = {
        "videos": [
            {
                "video_id": vid,
                "ground_truth": references[i],
                "malmm": {
                    "caption": malmm_caps[i],
                    "rouge": malmm_rouge["per_video"][i],
                    "bertscore": malmm_bs["per_video"][i],
                },
                "twostage": {
                    "caption": v2_caps[i],
                    "rouge": v2_rouge["per_video"][i],
                    "bertscore": v2_bs["per_video"][i],
                },
            }
            for i, vid in enumerate(video_ids)
        ],
        "averages": {
            "malmm":    {"rouge": malmm_rouge["avg"], "bertscore": malmm_bs["avg"]},
            "twostage": {"rouge": v2_rouge["avg"],    "bertscore": v2_bs["avg"]},
        },
    }
    out_path = "outputs/eval_comparison.json"
    import os; os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {out_path}\n")


if __name__ == "__main__":
    main()
