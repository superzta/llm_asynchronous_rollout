#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

mkdir -p results/experiments

python3 -m src.run_experiment_grid \
  --output-root results/experiments \
  --modes sync_train,async_train,async_areal_style \
  --staleness-k-values 0,1,2,4 \
  --seeds 0,1,2 \
  --update-batch-sizes 4 \
  --queue-maxsizes 64 \
  --epochs-values 2 \
  --backend-values tiny_policy \
  --lr-values 0.1 \
  --num-rollout-workers-values 1 \
  --num-trainer-workers-values 1 \
  --rollout-devices cuda:0 \
  --trainer-devices cuda:1 \
  --interrupt-check-interval-sec 0.02 \
  --generation-chunk-size 4 \
  --max-new-tokens 64 \
  "$@"
