#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

mkdir -p results

python3 -m src.run_async_areal_style \
  --dataset data/tiny_coding_train.jsonl \
  --backend tiny_policy \
  --num-rollout-workers 1 \
  --num-trainer-workers 1 \
  --rollout-devices cuda:0 \
  --trainer-devices cuda:1 \
  --epochs 2 \
  --seed 7 \
  --lr 0.1 \
  --update-batch-size 4 \
  --staleness-k 1 \
  --queue-maxsize 128 \
  --interrupt-check-interval-sec 0.02 \
  --generation-chunk-size 4 \
  --results-jsonl results/async_areal_results.jsonl \
  --summary-json results/async_areal_summary.json
