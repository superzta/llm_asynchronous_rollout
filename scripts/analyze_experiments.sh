#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/analyze_experiments.sh <experiment_dir>"
  echo "Example: bash scripts/analyze_experiments.sh results/experiments/20260329_120000"
  exit 1
fi

python3 -m src.analyze_experiments --experiment-dir "$1"
