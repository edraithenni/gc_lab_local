# MLflow for gc_lab_local

This add-on does **not** change the Ray training code. It reads existing run folders in `data/results/` and logs metrics to MLflow. It also adds PNG plots as MLflow artifacts. `model_final.pt` is left in the original result folder.

## Install (Mac, from gc_lab_local root)

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-mlflow
.venv-mlflow/bin/python -m pip install --upgrade pip
.venv-mlflow/bin/python -m pip install mlflow matplotlib
```

Copy `scripts/log_mlflow.py` to the local project's `scripts/` and `run_local_mlflow.sh` to the project root.

## Terminal 1: MLflow server

```bash
mkdir -p data/mlflow
(cd data/mlflow && ../../.venv-mlflow/bin/mlflow server --host 127.0.0.1 --port 5000)
```

Open http://127.0.0.1:5000. Leave the server terminal running.

## Terminal 2: Import existing runs

```bash
.venv-mlflow/bin/python scripts/log_mlflow.py --results data/results --latest --device mps
```

Omit `--latest` to import all completed run folders. Re-imports are skipped automatically by source folder name.

For future runs, run `./run_local_mlflow.sh` while the MLflow server is running. It starts training and then imports the newest run. Training files are retained even if the subsequent MLflow import fails.

## In MLflow UI

Choose experiment `Gradient Coding - CIFAR10`. Open the run to view scalar time series under Metrics and PNGs under Artifacts → `plots`. Compare several selected runs from the Experiment page.

The reported `train_loss` is a diagnostic metric and is **not** directly comparable between the exact GC and PGC schemes. `accepted_payload_mib` counts accepted payloads, not actual total network traffic.
