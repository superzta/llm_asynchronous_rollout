#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

mkdir -p results

python3 -m src.run_sync_baseline \
  --dataset data/tiny_coding_train.jsonl \
  --epochs 1 \
  --backend dummy \
  --results-jsonl results/sync_results.jsonl \
  --summary-json results/sync_summary.json
