#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

mkdir -p results

python3 -m src.run_async_baseline \
  --dataset data/tiny_coding_train.jsonl \
  --epochs 2 \
  --seed 7 \
  --backend tiny_policy \
  --lr 0.1 \
  --update-batch-size 4 \
  --staleness-k 1 \
  --max-new-tokens 64 \
  --results-jsonl results/async_train_results.jsonl \
  --summary-json results/async_train_summary.json
