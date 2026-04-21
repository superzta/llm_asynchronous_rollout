#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

mkdir -p results/experiments

# Pressure-oriented 3-way comparison:
# - sync_train for reference
# - async_train under bounded staleness
# - async_areal_style with overlap/interrupt pressure
python3 -m src.run_experiment_grid \
  --output-root results/experiments \
  --modes sync_train,async_train,async_areal_style \
  --staleness-k-values 0,1,2,4 \
  --seeds 0,1,2 \
  --update-batch-sizes 2 \
  --queue-maxsizes 256 \
  --epochs-values 6 \
  --backend-values tiny_policy \
  --lr-values 0.1 \
  --num-rollout-workers-values 2 \
  --num-trainer-workers-values 1 \
  --rollout-devices cuda:0,cuda:0 \
  --trainer-devices cuda:1 \
  --producer-delay-sec 0.0 \
  --learner-delay-sec 0.04 \
  --interrupt-check-interval-sec 0.03 \
  --generation-chunk-size 2 \
  --rollout-chunk-delay-sec 0.01 \
  --replay-dispatch-delay-sec 0.01 \
  --controller-consume-delay-sec 0.005 \
  --max-interrupt-retries 2 \
  --max-new-tokens 64 \
  "$@"
