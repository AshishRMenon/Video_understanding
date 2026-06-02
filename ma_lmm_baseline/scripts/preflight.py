"""
Preflight check for a RunPod (or similar) host that may already have
some of the assets we need under /workspace/.

Scans for:
  1) Vicuna-7b checkpoint (tokenizer.model + llama config.json)
  2) Video files; flags long-form (>50 min) source candidates
  3) Pre-existing 10/30/60-min variant directories
  4) An already-cloned MA-LMM repo
  5) The malmm_baseline conda env

Writes a human-readable summary to stdout AND a structured JSON report
to ./preflight_report.json so a follow-up agent can parse what's
already present and decide which setup steps to skip.

Skipped during the walk (cache/install bloat that slows the scan):
  .cache, .git, __pycache__, node_modules, site-packages, .conda, envs

Usage:
    python scripts/preflight.py --workspace /workspace
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
LONG_VIDEO_MIN_SEC = 50 * 60  # 50 min — buffer below the user's >60 min sources
PRUNE_DIRS = {".cache", ".git", "__pycache__", "node_modules",
              "site-packages", ".conda", "envs", "wandb"}


def walk(root: Path):
    """os.walk variant that prunes cache/install dirs and yields Paths."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        dp = Path(dirpath)
        for d in dirnames:
            yield dp / d, True
        for f in filenames:
            yield dp / f, False


def find_vicuna(root: Path) -> list[Path]:
    hits: list[Path] = []
    for p, is_dir in walk(root):
        if is_dir or p.name != "tokenizer.model":
            continue
        parent = p.parent
        cfg = parent / "config.json"
        if not cfg.exists():
            continue
        try:
            data = json.loads(cfg.read_text())
        except Exception:
            continue
        if data.get("model_type") == "llama" or "vicuna" in str(parent).lower():
            hits.append(parent)
    return sorted(set(hits))


def ffprobe_duration(path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def find_videos(root: Path) -> list[dict]:
    candidates = [p for p, is_dir in walk(root)
                  if (not is_dir) and p.suffix.lower() in VIDEO_EXTS]
    print(f"      ffprobe-ing {len(candidates)} candidate video files...")
    out = []
    for i, p in enumerate(candidates, 1):
        if i % 10 == 0 or i == len(candidates):
            print(f"        progress: {i}/{len(candidates)}")
        dur = ffprobe_duration(p)
        if dur is None:
            continue
        out.append({
            "path": str(p),
            "duration_sec": round(dur, 1),
            "duration_min": round(dur / 60, 2),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
        })
    return out


def find_variant_dirs(root: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {"10min": [], "30min": [], "60min": []}
    for p, is_dir in walk(root):
        if not is_dir:
            continue
        norm = p.name.lower().replace("_", "").replace("-", "").replace(" ", "")
        for key in ("10min", "30min", "60min"):
            if key in norm:
                has_video = any(
                    c.is_file() and c.suffix.lower() in VIDEO_EXTS
                    for c in p.iterdir()
                )
                if has_video:
                    found[key].append(str(p))
    return found


def find_malmm_clones(root: Path) -> list[Path]:
    cands: list[Path] = []
    for p, is_dir in walk(root):
        if not is_dir or p.name != "blip2_models":
            continue
        # MA-LMM/lavis/models/blip2_models  -> repo root is 3 dirs up
        repo_root = p.parent.parent.parent
        if (repo_root / "lavis").is_dir() and (repo_root / "train.py").exists():
            cands.append(repo_root)
    return sorted(set(cands))


def conda_env_present(name: str) -> bool:
    try:
        r = subprocess.run(["conda", "env", "list"],
                           capture_output=True, text=True, timeout=15)
        for line in r.stdout.splitlines():
            if line.startswith(name + " ") or line.endswith("/" + name):
                return True
        return False
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/workspace",
                    help="Root to scan (default: /workspace)")
    ap.add_argument("--report", default="preflight_report.json",
                    help="Where to write the JSON report")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    print(f"Scanning {ws} ...\n")

    if not ws.exists():
        print(f"! {ws} does not exist on this host. Nothing to scan.")
        return

    print("[1/5] Vicuna-7b weights (tokenizer.model + llama config.json)...")
    vicuna = find_vicuna(ws)
    for v in vicuna:
        print(f"      FOUND: {v}")
    if not vicuna:
        print("      none found")

    print("\n[2/5] Video files...")
    videos = find_videos(ws)
    long_videos = [v for v in videos if v["duration_sec"] >= LONG_VIDEO_MIN_SEC]
    print(f"      total video files: {len(videos)}")
    print(f"      long-form (>50 min): {len(long_videos)}")
    for v in long_videos:
        print(f"        {v['duration_min']:>6} min  {v['path']}")

    print("\n[3/5] Pre-existing 10/30/60-min variant directories...")
    variants = find_variant_dirs(ws)
    any_variant = False
    for k, paths in variants.items():
        for p in paths:
            print(f"      {k}: {p}")
            any_variant = True
    if not any_variant:
        print("      none found")

    print("\n[4/5] Existing MA-LMM repo clones...")
    malmm = find_malmm_clones(ws)
    for m in malmm:
        print(f"      FOUND: {m}")
    if not malmm:
        print("      none found")

    print("\n[5/5] Conda env 'malmm_baseline'...")
    has_env = conda_env_present("malmm_baseline")
    print(f"      {'present' if has_env else 'absent'}")

    # Structured report
    report = {
        "workspace": str(ws),
        "vicuna_candidates": [str(v) for v in vicuna],
        "videos_total": len(videos),
        "long_videos": long_videos,
        "variant_dirs": variants,
        "malmm_clones": [str(m) for m in malmm],
        "has_malmm_baseline_env": has_env,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\nStructured report -> {args.report}")

    # Actionable summary
    print("\n=== SUMMARY & NEXT STEPS ===")
    if vicuna:
        v = vicuna[0]
        print(f"  [OK] Vicuna found: {v}")
        print(f"       Wire it in:  mkdir -p MA-LMM/llm && ln -s {v} MA-LMM/llm/vicuna-7b")
    else:
        print("  [--] Vicuna missing -> setup.sh will download (~13 GB)")

    if len(long_videos) >= 5:
        print(f"  [OK] {len(long_videos)} long-form source videos located (need 5)")
    else:
        print(f"  [--] Only {len(long_videos)} long-form videos found (need 5)")

    have_all = all(variants[k] for k in ("10min", "30min", "60min"))
    if have_all:
        print("  [OK] 10/30/60-min variant directories already exist")
        for k in ("10min", "30min", "60min"):
            print(f"       ln -s {variants[k][0]} data/videos/{k}")
    else:
        missing = [k for k in ("10min", "30min", "60min") if not variants[k]]
        print(f"  [--] Missing variant dirs: {missing}")
        print("       Cut them from the long sources, e.g.:")
        print("         ffmpeg -ss 0 -t 600  -c copy <src> data/videos/10min/<id>.mp4")
        print("         ffmpeg -ss 0 -t 1800 -c copy <src> data/videos/30min/<id>.mp4")
        print("         ffmpeg -ss 0 -t 3600 -c copy <src> data/videos/60min/<id>.mp4")

    if malmm:
        print(f"  [OK] MA-LMM clone exists: {malmm[0]}")
        print(f"       Wire it in:  ln -s {malmm[0]} ./MA-LMM")
    else:
        print("  [--] MA-LMM not cloned -> setup.sh will clone it")

    if has_env:
        print("  [OK] conda env malmm_baseline already present")
    else:
        print("  [--] conda env malmm_baseline absent -> setup.sh will create it")


if __name__ == "__main__":
    main()
