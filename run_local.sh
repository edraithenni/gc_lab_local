#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
set -a
source "$ROOT/local.env"
set +a
unset RAY_ADDRESS
export DATA_ROOT="${DATA_ROOT:-$ROOT/data/cifar10}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/data/results}"
mkdir -p "$RESULTS_DIR"
python app/train_ray.py
