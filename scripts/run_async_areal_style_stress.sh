#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

mkdir -p results

python3 -m src.run_async_areal_style \
  --dataset data/tiny_coding_train.jsonl \
  --backend tiny_policy \
  --num-rollout-workers 2 \
  --num-trainer-workers 1 \
  --rollout-devices cpu,cpu \
  --trainer-devices cpu \
  --epochs 6 \
  --seed 7 \
  --lr 0.1 \
  --update-batch-size 2 \
  --staleness-k 1 \
  --queue-maxsize 256 \
  --producer-delay-sec 0.0 \
  --learner-delay-sec 0.03 \
  --interrupt-check-interval-sec 0.005 \
  --generation-chunk-size 8 \
  --results-jsonl results/async_areal_stress_results.jsonl \
  --summary-json results/async_areal_stress_summary.json
