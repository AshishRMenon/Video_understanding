"""
QA-based caption quality evaluation (macro + micro) for Experiment 2 frame sweep.

Approach (QAGS-style, two-level):
  For each video two independent question sets are generated from ground truth:

    MACRO  — from Ego4D segment-level summaries (5-min blocks).
             Tests whether the caption captures the main activity, setting,
             and key objects.  Answerable from any decent paragraph-length
             caption.

    MICRO  — from Ego4D second-by-second narrations (uniformly sampled).
             Tests whether the caption captures specific actions and details.
             Only answered by more detailed / longer captions.

  Questions are generated once per video and cached; answer-checking runs
  once per (video, config) pair.

  Three scores are reported per config:
    macro_qa_score  — fraction of macro questions answered (avg over videos)
    micro_qa_score  — fraction of micro questions answered (avg over videos)
    qa_score        — average of macro + micro (overall coverage)

  The macro/micro gap reveals how much detail depth an approach captures —
  a useful baseline for future model comparisons.

Output:
  Sweep mode  → {sweep_dir}/qa_scores.json   (auto-read by plot_sweep.py)
  Single mode → {captions_json stem}_qa.json

Usage:
    # On existing sweep outputs:
    python scripts/qa_eval.py \\
        --sweep_dir  outputs/exp2_frame_sweep \\
        --narration  /workspace/ego4d_meta/v2/annotations/narration.json \\
        --ann_path   data/annotations/ego4d_10min.json

    # On a single captions file (e.g. from run_comparison.sh):
    python scripts/qa_eval.py \\
        --captions_json outputs/captions_10min.json \\
        --narration  /workspace/ego4d_meta/v2/annotations/narration.json \\
        --ann_path   data/annotations/ego4d_10min.json

    # Use a stronger model:
    python scripts/qa_eval.py ... --model claude-sonnet-4-6

Prerequisites:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------

try:
    import anthropic
except ImportError:
    print("Installing anthropic SDK...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "anthropic"])
    import anthropic


# ---------------------------------------------------------------------------
# Ego4D notation cleaning
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Replace Ego4D annotation placeholders with natural language."""
    text = re.sub(r"#Summary\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"#unsure", "", text, flags=re.IGNORECASE)
    text = re.sub(r"#[A-Za-z]\b", "", text)      # strip any #X marker
    text = re.sub(r"\bC\b", "the person", text)   # camera wearer
    text = re.sub(r"\b[OX]\b", "another person", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Narration loading — two levels
# ---------------------------------------------------------------------------

def load_macro_narrations(narration_path: str, ann_path: str) -> dict[str, str]:
    """
    {video_id: text} from segment-level *summaries* (5-min blocks).
    These are short, high-level descriptions — ideal for generating questions
    that paragraph-length captions can plausibly answer.
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
            _clean(s["summary_text"])
            for s in summaries
            if s.get("start_sec", 9999) < max_sec and s.get("summary_text", "").strip()
        ]
        if texts:
            gt[vid] = " ".join(texts)
    return gt


def load_micro_narrations(narration_path: str, ann_path: str,
                           max_narrations: int = 80) -> dict[str, str]:
    """
    {video_id: text} from second-by-second *narrations*, uniformly sampled.
    Dense enough to cover fine-grained actions; capped so questions stay
    specific but not hyper-local (e.g. not "did the person blink?").
    """
    with open(ann_path) as f:
        anns = json.load(f)
    with open(narration_path) as f:
        narr_data = json.load(f)

    gt = {}
    for ann in anns:
        vid     = ann["video_id"]
        max_sec = ann.get("n_frames", 0) / 10
        narrations = (narr_data.get(vid, {})
                      .get("narration_pass_1", {})
                      .get("narrations", []))
        in_range = [
            n for n in narrations
            if n.get("timestamp_sec", 9999) < max_sec
            and n.get("narration_text", "").strip()
        ]
        if not in_range:
            continue
        if len(in_range) > max_narrations:
            step = len(in_range) / max_narrations
            in_range = [in_range[int(i * step)] for i in range(max_narrations)]
        texts = [_clean(n["narration_text"]) for n in in_range]
        texts = [t for t in texts if t]
        if texts:
            gt[vid] = " ".join(texts)
    return gt


# ---------------------------------------------------------------------------
# Sweep result loading
# ---------------------------------------------------------------------------

def load_sweep_results(sweep_dir: str) -> list[tuple[int, int, list]]:
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


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_GEN_SYSTEM = "You are a video understanding researcher designing factual evaluation questions."

_MACRO_PROMPT = """\
You are given high-level segment summaries of a video clip.

Generate exactly {n} yes/no questions to test whether a short paragraph-length \
video caption (200-400 words) captures the MAIN THEMES of this video.

Rules:
- Questions must be answerable YES or NO from a paragraph-length video description.
- Focus on HIGH-LEVEL content: dominant activity, setting, key objects, main people.
- Good: "Does the video show a person weaving fabric on a loom?"
- Bad:  "Does the person pass the shuttle from left hand to right hand?"
- Use natural language — write "the person" not any placeholder initials.
- Vary aspects: activity type, location/setting, objects used, people present, \
overall goal.
- Output one question per line, numbered 1 to {n}. No extra text.

Summaries:
{narration}

Generate exactly {n} macro yes/no questions:"""

_MICRO_PROMPT = """\
You are given fine-grained second-by-second narrations of a video clip.

Generate exactly {n} yes/no questions to test whether a video caption captures \
SPECIFIC DETAILS of this video — things a rich, detailed caption would mention \
but a superficial caption would miss.

Rules:
- Questions must be answerable YES or NO from a detailed video description.
- Focus on SPECIFIC details: particular objects handled, sequences of actions, \
  how something is done, specific interactions between people or objects.
- Good: "Does the person use a weaving pin to separate threads on the loom?"
- Bad:  "Does the person weave?" (too broad — save that for macro)
- Also bad: "Does the person blink?" (too trivial — must be captionable)
- Use natural language — write "the person" not any placeholder initials.
- Output one question per line, numbered 1 to {n}. No extra text.

Narrations:
{narration}

Generate exactly {n} micro yes/no questions:"""

_CHECK_SYSTEM = "You are a strict evaluator of video caption quality."

_CHECK_PROMPT = """\
A predicted video caption is given below. For each question answer YES if the \
caption contains enough information to reasonably answer it, or NO if the \
information is absent or too vague.

Caption:
{caption}

Questions:
{questions}

Respond with exactly one line per question — "1. YES" or "1. NO". Nothing else."""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call(client, model, system, user):
    msg = client.messages.create(
        model=model,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


def _parse_questions(raw: str, n: int) -> list[str]:
    questions = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned:
            questions.append(cleaned)
    return questions[:n]


def _parse_answers(raw: str, n: int) -> list[bool]:
    answers = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        answers.append("YES" in line.upper())
    while len(answers) < n:
        answers.append(False)
    return answers[:n]


def gen_questions(client, model, narration, n, prompt_template):
    raw = _call(client, model, _GEN_SYSTEM,
                prompt_template.format(n=n, narration=narration))
    return _parse_questions(raw, n)


def check_answers(client, model, caption, questions) -> list[bool]:
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    raw = _call(client, model, _CHECK_SYSTEM,
                _CHECK_PROMPT.format(caption=caption, questions=numbered))
    return _parse_answers(raw, len(questions))


def _avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else None


# ---------------------------------------------------------------------------
# Core evaluation — shared by single-config and sweep modes
# ---------------------------------------------------------------------------

def evaluate_records(client, model, records, cache, gt_macro, gt_micro):
    """
    Evaluate a list of {video_id, caption} records against cached questions.
    Returns (per_video_results, macro_score, micro_score, overall_score).
    """
    per_video = []
    macro_scores, micro_scores = [], []

    for rec in records:
        vid     = rec["video_id"]
        caption = rec.get("caption", "")
        entry   = cache.get(vid, {})
        macro_qs = entry.get("macro", [])
        micro_qs = entry.get("micro", [])

        if not caption:
            print(f"    [skip] {vid} — empty caption")
            continue
        if not macro_qs and not micro_qs:
            print(f"    [skip] {vid} — no questions in cache")
            continue

        macro_ans = check_answers(client, model, caption, macro_qs) if macro_qs else []
        micro_ans = check_answers(client, model, caption, micro_qs) if micro_qs else []

        m_score  = sum(macro_ans) / len(macro_ans) if macro_ans else None
        mi_score = sum(micro_ans) / len(micro_ans) if micro_ans else None
        overall  = _avg([s for s in [m_score, mi_score] if s is not None])

        if m_score  is not None: macro_scores.append(m_score)
        if mi_score is not None: micro_scores.append(mi_score)

        per_video.append({
            "video_id":         vid,
            "macro_questions":  macro_qs,
            "macro_answers":    macro_ans,
            "macro_score":      round(m_score,  4) if m_score  is not None else None,
            "micro_questions":  micro_qs,
            "micro_answers":    micro_ans,
            "micro_score":      round(mi_score, 4) if mi_score is not None else None,
            "score":            round(overall,  4) if overall  is not None else None,
        })

        macro_str = f"{sum(macro_ans)}/{len(macro_ans)}" if macro_ans else "—"
        micro_str = f"{sum(micro_ans)}/{len(micro_ans)}" if micro_ans else "—"
        print(f"    {vid[:12]}…  macro {macro_str}  micro {micro_str}  "
              f"overall {overall:.0%}" if overall is not None else
              f"    {vid[:12]}…  macro {macro_str}  micro {micro_str}")

    return per_video, _avg(macro_scores), _avg(micro_scores), _avg(macro_scores + micro_scores)


def ensure_questions(client, model, video_ids, cache, gt_macro, gt_micro,
                     n_macro, n_micro, cache_path):
    """Generate and cache questions for any videos not yet in cache."""
    needs = [v for v in video_ids
             if v not in cache and (v in gt_macro or v in gt_micro)]
    if not needs:
        return

    print(f"Generating questions for {len(needs)} video(s)…")
    for i, vid in enumerate(needs, 1):
        print(f"  [{i}/{len(needs)}] {vid[:12]}…", end=" ", flush=True)
        entry = {}
        try:
            if vid in gt_macro and n_macro > 0:
                entry["macro"] = gen_questions(
                    client, model, gt_macro[vid], n_macro, _MACRO_PROMPT)
            if vid in gt_micro and n_micro > 0:
                entry["micro"] = gen_questions(
                    client, model, gt_micro[vid], n_micro, _MICRO_PROMPT)
            cache[vid] = entry
            print(f"→ {len(entry.get('macro',[]))} macro  "
                  f"{len(entry.get('micro',[]))} micro")
        except Exception as e:
            print(f"ERROR: {e}")

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  Cache saved to {cache_path}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sweep_dir",
                     help="Directory of captions_f*_m*.json (multi-config sweep)")
    grp.add_argument("--captions_json",
                     help="Single [{video_id, caption}] JSON (one-config eval)")
    ap.add_argument("--narration",   required=True, help="Ego4D narration.json path")
    ap.add_argument("--ann_path",    required=True, help="Annotation JSON (ego4d_10min.json)")
    ap.add_argument("--output",      default=None,  help="Output JSON path (auto-named if omitted)")
    ap.add_argument("--model",       default="claude-haiku-4-5-20251001",
                    help="Anthropic model. Use claude-sonnet-4-6 for higher accuracy.")
    ap.add_argument("--n_macro",     type=int, default=8,
                    help="Macro questions per video (from segment summaries)")
    ap.add_argument("--n_micro",     type=int, default=8,
                    help="Micro questions per video (from fine-grained narrations)")
    ap.add_argument("--cache",       default=None,
                    help="Question cache JSON path (shared across runs)")
    args = ap.parse_args()

    if args.sweep_dir:
        out_path   = args.output or f"{args.sweep_dir}/qa_scores.json"
        cache_path = args.cache  or f"{args.sweep_dir}/qa_cache.json"
    else:
        base       = args.captions_json.replace(".json", "")
        out_path   = args.output or f"{base}_qa.json"
        cache_path = args.cache  or str(Path(args.captions_json).parent / "qa_cache.json")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Model          : {args.model}")
    print(f"Questions/video: {args.n_macro} macro + {args.n_micro} micro "
          f"= {args.n_macro + args.n_micro} total")
    print(f"Input          : {args.sweep_dir or args.captions_json}")
    print(f"Output         : {out_path}")
    print()

    # Load both narration levels
    print("Loading ground-truth narrations…")
    gt_macro = load_macro_narrations(args.narration, args.ann_path)
    gt_micro = load_micro_narrations(args.narration, args.ann_path)
    print(f"  Macro (summaries)   : {len(gt_macro)} videos")
    print(f"  Micro (narrations)  : {len(gt_micro)} videos")

    # Load cache
    cache: dict = {}
    if Path(cache_path).exists():
        with open(cache_path) as f:
            raw_cache = json.load(f)
        # Migrate old flat-list format → new {macro, micro} dict format
        for vid, val in raw_cache.items():
            cache[vid] = val if isinstance(val, dict) else {}
        print(f"  Cache loaded: {len(cache)} video(s) from {cache_path}")

    # ----------------------------------------------------------------
    # Single-config mode
    # ----------------------------------------------------------------
    if args.captions_json:
        with open(args.captions_json) as f:
            records = json.load(f)
        print(f"  {len(records)} caption records loaded.\n")

        all_vids = [r["video_id"] for r in records]
        ensure_questions(client, args.model, all_vids, cache,
                         gt_macro, gt_micro, args.n_macro, args.n_micro, cache_path)

        print("Evaluating captions…")
        per_video, macro_score, micro_score, qa_score = evaluate_records(
            client, args.model, records, cache, gt_macro, gt_micro)

        result = {
            "model":           args.model,
            "n_macro":         args.n_macro,
            "n_micro":         args.n_micro,
            "macro_qa_score":  macro_score,
            "micro_qa_score":  micro_score,
            "qa_score":        qa_score,
            "per_video":       per_video,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n  Macro QA : {macro_score:.2%}" if macro_score is not None else "\n  Macro QA : N/A")
        print(f"  Micro QA : {micro_score:.2%}" if micro_score is not None else "  Micro QA : N/A")
        print(f"  Overall  : {qa_score:.2%}"    if qa_score    is not None else "  Overall  : N/A")
        print(f"\nSaved to {out_path}")
        return

    # ----------------------------------------------------------------
    # Sweep mode
    # ----------------------------------------------------------------
    configs = load_sweep_results(args.sweep_dir)
    if not configs:
        print(f"No sweep results in {args.sweep_dir}. Run run_frame_sweep.sh first.")
        sys.exit(1)
    print(f"  {len(configs)} sweep configurations found.\n")

    all_vids = sorted({r["video_id"] for _, _, recs in configs for r in recs})
    ensure_questions(client, args.model, all_vids, cache,
                     gt_macro, gt_micro, args.n_macro, args.n_micro, cache_path)

    results_configs = []
    for cfg_idx, (nf, mbl, recs) in enumerate(configs, 1):
        print(f"[{cfg_idx}/{len(configs)}] num_frames={nf}  mbl={mbl}")
        per_video, macro_score, micro_score, qa_score = evaluate_records(
            client, args.model, recs, cache, gt_macro, gt_micro)

        results_configs.append({
            "num_frames":         nf,
            "memory_bank_length": mbl,
            "macro_qa_score":     macro_score,
            "micro_qa_score":     micro_score,
            "qa_score":           qa_score,
            "per_video":          per_video,
        })
        print(f"  → macro {macro_score:.2%}  micro {micro_score:.2%}  "
              f"overall {qa_score:.2%}\n"
              if all(s is not None for s in [macro_score, micro_score, qa_score])
              else "  → incomplete scores\n")

    output = {
        "model":    args.model,
        "n_macro":  args.n_macro,
        "n_micro":  args.n_micro,
        "configs":  results_configs,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}\n")

    # Summary table
    print("=" * 68)
    print(f"{'frames':>8}  {'MBL':>6}  {'Macro QA':>10}  {'Micro QA':>10}  {'Overall':>9}")
    print("-" * 68)
    for cfg in results_configs:
        ma = f"{cfg['macro_qa_score']:.4f}" if cfg['macro_qa_score'] is not None else "N/A"
        mi = f"{cfg['micro_qa_score']:.4f}" if cfg['micro_qa_score'] is not None else "N/A"
        ov = f"{cfg['qa_score']:.4f}"       if cfg['qa_score']       is not None else "N/A"
        print(f"{cfg['num_frames']:>8}  {cfg['memory_bank_length']:>6}  "
              f"{ma:>10}  {mi:>10}  {ov:>9}")
    print("=" * 68)


if __name__ == "__main__":
    main()
