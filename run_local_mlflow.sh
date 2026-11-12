#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"
./run_local.sh
set -a
source "$ROOT/local.env"
set +a
"$ROOT/.venv-mlflow/bin/python" "$ROOT/scripts/log_mlflow.py" --results "$ROOT/data/results" --latest --device "${GC_DEVICE:-unspecified}"
