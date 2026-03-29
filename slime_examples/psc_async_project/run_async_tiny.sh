#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/../.. >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

echo "Running standalone minimal async baseline (fallback path)."
bash scripts/run_async.sh
