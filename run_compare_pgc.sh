
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$PWD"

[[ -f local.env ]] || { echo "local.env not found"; exit 1; }
[[ -x .venv/bin/python ]] || { echo ".venv is missing"; exit 1; }
[[ -x .venv-mlflow/bin/python ]] || { echo ".venv-mlflow is missing"; exit 1; }
[[ -f scripts/log_mlflow.py ]] || { echo "MLflow importer is missing"; exit 1; }
[[ -f app/coding/pipelined_fr.py ]] || { echo "PGC-FR codec is missing"; exit 1; }

export PATH="$ROOT/.venv/bin:$PATH"

if ! curl -fsS http://127.0.0.1:5000/ >/dev/null 2>&1; then
    mkdir -p "$ROOT/data/mlflow"
    (
        cd "$ROOT/data/mlflow"
        nohup "$ROOT/.venv-mlflow/bin/mlflow" server \
            --host 127.0.0.1 --port 5000 \
            > mlflow-server.log 2>&1 < /dev/null &
    )

    for ((i=0; i<60; i++)); do
        if curl -fsS http://127.0.0.1:5000/ >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    curl -fsS http://127.0.0.1:5000/ >/dev/null || {
        echo "MLflow failed to start. See data/mlflow/mlflow-server.log"
        exit 1
    }
fi

BACKUP="$(mktemp "${TMPDIR:-/tmp}/gc-local-env.XXXXXX")"
cp local.env "$BACKUP"

restore_env() {
    cp "$BACKUP" "$ROOT/local.env"
    rm -f "$BACKUP"
}
trap restore_env EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

EPOCHS="${EPOCHS:-100}"
SERIES="compare-all-$(date -u +%Y%m%d-%H%M%S)-$$"

for SCHEME in classic cce_gc_fr cce_gc_cr pgc_fr pgc_cr; do
    RESULTS_DIR="$ROOT/data/results/$SERIES/$SCHEME"
    mkdir -p "$RESULTS_DIR"

    awk '
        /^[[:space:]]*(GC_SCHEME|NUM_WORKERS|NUM_PARTITIONS|GC_STRAGGLERS|CCE_M|SEED|GC_SEED|TRAIN_LIMIT|TEST_LIMIT|BATCH_SIZE|EPOCHS|LR|MOMENTUM|WEIGHT_DECAY|EMA_DECAY|GC_DEVICE|GC_EVAL_DEVICE|OMP_NUM_THREADS|AGGREGATOR_THREADS|SLOW_WORKER|STRAGGLER_DELAY|GC_PROFILE_STEPS|RESULTS_DIR)[[:space:]]*=/ { next }
        { print }
    ' "$BACKUP" > local.env

    cat >> local.env <<EOF
GC_SCHEME=$SCHEME
NUM_WORKERS=6
NUM_PARTITIONS=6
GC_STRAGGLERS=1
CCE_M=2
SEED=1234
GC_SEED=1234
TRAIN_LIMIT=12000
TEST_LIMIT=10000
BATCH_SIZE=40
EPOCHS=$EPOCHS
LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.0005
EMA_DECAY=0.99
GC_DEVICE=mps
GC_EVAL_DEVICE=mps
OMP_NUM_THREADS=1
AGGREGATOR_THREADS=1
SLOW_WORKER=-1
STRAGGLER_DELAY=0
GC_PROFILE_STEPS=0
RESULTS_DIR="$RESULTS_DIR"
EOF

    echo "===== START $SCHEME | 6 workers | $EPOCHS epochs ====="
    ./run_local.sh

    echo "===== IMPORT $SCHEME INTO MLFLOW ====="
    "$ROOT/.venv-mlflow/bin/python" "$ROOT/scripts/log_mlflow.py" \
        --results "$RESULTS_DIR" \
        --tracking-uri http://127.0.0.1:5000 \
        --experiment "Gradient Coding - CIFAR10" \
        --device mps

    echo "===== DONE $SCHEME ====="
done

echo "All 5 runs finished. Results: data/results/$SERIES"
echo "MLflow UI: http://127.0.0.1:5000"
