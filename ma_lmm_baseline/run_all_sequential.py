#!/usr/bin/env python3
"""
Orchestrator: runs run_chunked.sh for four experiment configurations in sequence.

Order:
  1. 10-min chunks  on 60-min videos
  2.  5-min chunks  on 60-min videos
  3.  2-min chunks  on 60-min videos
  4.  1-min chunk   on 30-min videos

Each configuration is retried up to MAX_RETRIES times.  A retry happens when
run_chunked.sh exits non-zero (e.g. one video crashed) but at least one new
output JSON appeared — meaning progress was made and the next attempt will
skip already-finished videos.  Retrying stops when all videos are done OR no
progress was made on the last attempt (stuck failure).

Logs per-configuration:  logs/run_all/<label>.log
Master log:              logs/run_all/master.log
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configurations to run, in order
# ---------------------------------------------------------------------------
CONFIGS = [
    {"chunk_mins": "10", "variant": "60min"},
    {"chunk_mins": "5",  "variant": "60min"},
    {"chunk_mins": "2",  "variant": "60min"},
    {"chunk_mins": "1",  "variant": "30min"},
]

BASE_DIR   = Path(__file__).resolve().parent
LOG_DIR    = BASE_DIR / "logs" / "run_all"
MAX_RETRIES = 8   # per configuration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, log_fh=None):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    if log_fh:
        print(line, file=log_fh, flush=True)


def count_status(chunk_mins, variant):
    """Return (remaining, total) videos for this configuration."""
    frame_root = BASE_DIR / "data" / "frames" / variant
    out_dir    = BASE_DIR / "outputs" / "exp1_chunked" / f"{chunk_mins}min_chunks"
    if not frame_root.exists():
        return 0, 0
    all_vids = [d for d in sorted(frame_root.iterdir()) if d.is_dir()]
    done     = sum(1 for v in all_vids if (out_dir / f"{v.name}.json").exists())
    return len(all_vids) - done, len(all_vids)


def run_chunked_sh(chunk_mins, variant, cfg_log_path):
    """
    Invoke run_chunked.sh with the given env overrides.
    Appends stdout+stderr to cfg_log_path.
    Returns the process return code.
    """
    env = os.environ.copy()
    env["CHUNK_MINS"] = chunk_mins
    env["VARIANT"]    = variant

    with open(cfg_log_path, "a") as fh:
        fh.write(f"\n{'='*70}\n")
        fh.write(f"ATTEMPT at {ts()}  CHUNK_MINS={chunk_mins}  VARIANT={variant}\n")
        fh.write(f"{'='*70}\n")
        fh.flush()

        proc = subprocess.run(
            ["bash", "run_chunked.sh"],
            cwd=BASE_DIR,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


def tail_log(path, n=40):
    """Return last n lines of a file as a string."""
    try:
        lines = Path(path).read_text().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(could not read log)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    master_log_path = LOG_DIR / "master.log"

    with open(master_log_path, "a") as master:
        log("=" * 70, master)
        log("run_all_sequential.py  START", master)
        cfg_labels = [c["chunk_mins"] + "min/" + c["variant"] for c in CONFIGS]
        log("Configurations: " + str(cfg_labels), master)
        log("=" * 70, master)

        overall_ok = True

        for cfg in CONFIGS:
            chunk_mins = cfg["chunk_mins"]
            variant    = cfg["variant"]
            label      = f"{chunk_mins}min_chunks_{variant}"
            cfg_log    = LOG_DIR / f"{label}.log"

            log("", master)
            log(f"{'='*70}", master)
            log(f"CONFIG: {chunk_mins}-min chunks on {variant} videos", master)
            log(f"Log   : {cfg_log}", master)
            log(f"{'='*70}", master)

            remaining, total = count_status(chunk_mins, variant)
            if total == 0:
                log(f"  Frame directory missing for {variant}. Skipping.", master)
                continue

            if remaining == 0:
                log(f"  All {total} videos already complete. Skipping.", master)
                continue

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                remaining, total = count_status(chunk_mins, variant)
                if remaining == 0:
                    log(f"  All {total} videos done.", master)
                    success = True
                    break

                log(f"  Attempt {attempt}/{MAX_RETRIES}: {remaining}/{total} videos remaining…", master)

                rc = run_chunked_sh(chunk_mins, variant, cfg_log)

                remaining_after, _ = count_status(chunk_mins, variant)
                made_progress      = remaining_after < remaining

                if remaining_after == 0:
                    log(f"  ✓ All {total} videos complete after attempt {attempt}.", master)
                    success = True
                    break

                if rc == 0:
                    # Script exited 0 but some videos still missing — shouldn't
                    # happen normally; treat as done for this attempt.
                    log(f"  run_chunked.sh exited 0 but {remaining_after} videos still missing. Re-checking…", master)
                    continue

                # Non-zero exit — check progress.
                if made_progress:
                    log(f"  run_chunked.sh exited {rc} but made progress "
                        f"({remaining - remaining_after} new). Retrying…", master)
                    log(f"  --- last 40 lines of log ---", master)
                    log(tail_log(cfg_log), master)
                else:
                    log(f"  run_chunked.sh exited {rc} with NO progress. Pausing 30s then retrying…", master)
                    log(f"  --- last 40 lines of log ---", master)
                    log(tail_log(cfg_log), master)
                    # Short pause before retry so GPU can settle.
                    time.sleep(30)

            if not success:
                remaining_final, _ = count_status(chunk_mins, variant)
                if remaining_final > 0:
                    log(f"  FAILED: {remaining_final}/{total} videos incomplete after {MAX_RETRIES} attempts.", master)
                    log(f"  Check {cfg_log} for details.", master)
                    overall_ok = False
                else:
                    log(f"  All {total} videos done (completed during retries).", master)

        log("", master)
        log("=" * 70, master)
        log(f"run_all_sequential.py  {'DONE — all OK' if overall_ok else 'DONE — some configs FAILED'}", master)
        log("=" * 70, master)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
