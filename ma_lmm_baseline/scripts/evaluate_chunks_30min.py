"""
Evaluate chunk-sweep outputs against Ego4D 30-min ground truth.

Ground truth: narration_pass_1 summaries from narration.json whose
start_sec falls within the video's 30-min duration (n_frames/10 sec).

Metrics: ROUGE-1/2/L, BERTScore (roberta-large).

Usage:
    python scripts/evaluate_chunks_30min.py \
        --narration /workspace/ego4d_meta/v2/annotations/narration.json \
        --ann_path  data/annotations/ego4d_30min.json
"""

import argparse
import json
import os
import re
import subprocess
import sys


def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


try:
    from rouge_score import rouge_scorer
except ImportError:
    install("rouge-score")
    from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score
except ImportError:
    install("bert-score")
    from bert_score import score as bert_score


CHUNK_SIZES = ["2min", "5min", "10min"]
CAPTIONS_DIR = "outputs/exp4_chunk_sweep"


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
    print("  Computing BERTScore (roberta-large)...")
    P, R, F = bert_score(predictions, references, lang="en", model_type="roberta-large", verbose=False)
    per_video = [round(f.item() * 100, 2) for f in F]
    avg = round(sum(per_video) / len(per_video), 2)
    return {"per_video": per_video, "avg": avg}


def evaluate_chunk_size(chunk_size: str, video_ids: list, references: list, narration: dict) -> dict:
    caps_path = os.path.join(CAPTIONS_DIR, f"captions_{chunk_size}.json")
    with open(caps_path) as f:
        caps_by_vid = {r["video_id"]: r["caption"] for r in json.load(f)}

    predictions = []
    valid_ids = []
    valid_refs = []
    for vid, ref in zip(video_ids, references):
        if vid not in caps_by_vid:
            print(f"  [skip] {vid} not in {caps_path}")
            continue
        predictions.append(caps_by_vid[vid])
        valid_ids.append(vid)
        valid_refs.append(ref)

    print(f"\n[{chunk_size}] Evaluating {len(valid_ids)} videos...")
    print(f"  ROUGE...")
    rouge = compute_rouge(predictions, valid_refs)
    bs = compute_bertscore(predictions, valid_refs)

    return {
        "chunk_size": chunk_size,
        "video_ids": valid_ids,
        "predictions": predictions,
        "references": valid_refs,
        "rouge": rouge,
        "bertscore": bs,
    }


def print_results(results: list):
    print("\n" + "=" * 80)
    print(f"{'CHUNK SIZE':<12} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BERTScore':>10}")
    print("=" * 80)
    for r in results:
        avg_r = r["rouge"]["avg"]
        avg_b = r["bertscore"]["avg"]
        print(f"  {r['chunk_size']:<10} {avg_r['rouge1']:>8.2f} {avg_r['rouge2']:>8.2f} {avg_r['rougeL']:>8.2f} {avg_b:>10.2f}")
    print("=" * 80)

    # Per-video breakdown for each chunk size
    for r in results:
        print(f"\n--- Per-video: {r['chunk_size']} ---")
        print(f"  {'VIDEO':<42} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'BERT':>7}")
        for i, vid in enumerate(r["video_ids"]):
            rv = r["rouge"]["per_video"][i]
            bv = r["bertscore"]["per_video"][i]
            print(f"  {vid[:40]:<42} {rv['rouge1']:>6.2f} {rv['rouge2']:>6.2f} {rv['rougeL']:>6.2f} {bv:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--narration", required=True)
    ap.add_argument("--ann_path",  required=True, help="ego4d_30min.json")
    args = ap.parse_args()

    with open(args.narration) as f:
        narration = json.load(f)
    with open(args.ann_path) as f:
        anns = json.load(f)

    # Build ground truths from 30min annotation
    video_ids, references = [], []
    for ann in anns:
        vid = ann["video_id"]
        gt = build_ground_truth(narration, vid, ann["n_frames"])
        if not gt:
            print(f"[skip] {vid} — no narration summaries in range")
            continue
        video_ids.append(vid)
        references.append(gt)

    print(f"\nGround truth built for {len(video_ids)} videos (duration = n_frames/10 sec each)")

    # Show GT summary counts per video for transparency
    print("\nGround truth summary counts per video:")
    for vid, ref in zip(video_ids, references):
        n_sents = ref.count(". ") + 1
        ann = next(a for a in anns if a["video_id"] == vid)
        print(f"  {vid[:40]}  n_frames={ann['n_frames']}  duration={ann['n_frames']/10:.0f}s  ~{n_sents} sentences")

    # Evaluate each chunk size
    all_results = []
    for chunk_size in CHUNK_SIZES:
        result = evaluate_chunk_size(chunk_size, video_ids, references, narration)
        all_results.append(result)

    print_results(all_results)

    # Save
    out = {
        "ground_truth_source": "ego4d_30min.json (n_frames/10 sec duration filter)",
        "results": [
            {
                "chunk_size": r["chunk_size"],
                "avg_rouge": r["rouge"]["avg"],
                "avg_bertscore": r["bertscore"]["avg"],
                "per_video": [
                    {
                        "video_id": r["video_ids"][i],
                        "rouge": r["rouge"]["per_video"][i],
                        "bertscore": r["bertscore"]["per_video"][i],
                        "prediction_preview": r["predictions"][i][:150],
                        "reference_preview": r["references"][i][:150],
                    }
                    for i in range(len(r["video_ids"]))
                ],
            }
            for r in all_results
        ],
    }
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/eval_chunk_sweep_30min.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved to {out_path}\n")


if __name__ == "__main__":
    main()
