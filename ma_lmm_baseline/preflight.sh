#!/usr/bin/env bash
# Scan the host (default /workspace/) for assets that may already be
# present — Vicuna weights, Ego4D videos, variant dirs, MA-LMM clone,
# conda env. Writes preflight_report.json and prints next-step
# suggestions so we can wire existing assets in via symlinks instead
# of re-downloading.
set -euo pipefail
cd "$(dirname "$0")"

WORKSPACE="${WORKSPACE:-/workspace}"
python scripts/preflight.py --workspace "${WORKSPACE}" --report preflight_report.json
