import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient


def read_jsonl(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def candidates(path, latest=False):
    if (path / "config.json").is_file():
        return [path]
    found = [p for p in path.iterdir() if p.is_dir() and (p / "config.json").is_file()]
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:1] if latest else found


def make_plots(folder, epochs, steps, n):
    if epochs:
        xs = [int(e["epoch"]) for e in epochs]
        elapsed = []
        current = 0.0
        for e in epochs:
            current += float(e["epoch_time_s"])
            elapsed.append(current / 60.0)
        plots = [
            ("accuracy_by_epoch.png", xs, [float(e["test_accuracy"]) for e in epochs], "Epoch", "Test accuracy", "Test accuracy vs epoch"),
            ("test_loss_by_epoch.png", xs, [float(e["test_loss"]) for e in epochs], "Epoch", "Test loss", "Test loss vs epoch"),
            ("accuracy_by_time.png", elapsed, [float(e["test_accuracy"]) for e in epochs], "Cumulative epoch time (min)", "Test accuracy", "Test accuracy vs training time"),
            ("epoch_duration.png", xs, [float(e["epoch_time_s"]) for e in epochs], "Epoch", "Seconds", "Epoch duration"),
        ]
        for name, x, y, xl, yl, title in plots:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(x, y)
            ax.set(xlabel=xl, ylabel=yl, title=title)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(folder / name, dpi=150)
            plt.close(fig)
    if steps:
        xs = list(range(1, len(steps) + 1))
        waits = [float(s["wait_time_s"]) for s in steps]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(xs, waits, linewidth=0.8)
        ax.set(xlabel="SGD update", ylabel="Wait time (s)", title="Waiting for sufficient responses")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(folder / "step_wait.png", dpi=150)
        plt.close(fig)
        counts = Counter(int(worker) for row in steps for worker in row.get("responders", []))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([str(i) for i in range(n)], [counts[i] for i in range(n)])
        ax.set(xlabel="Worker ID", ylabel="Selected steps", title="Worker responses used")
        fig.tight_layout()
        fig.savefig(folder / "worker_selection.png", dpi=150)
        plt.close(fig)


def add_metric(metrics, key, value, step, timestamp):
    if value is not None:
        metrics.append(Metric(key=key, value=float(value), timestamp=timestamp, step=int(step)))


def import_run(directory, client, experiment_id, device):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    epochs = read_jsonl(directory / "epoch_metrics.jsonl")
    steps = read_jsonl(directory / "step_metrics.jsonl")
    if not epochs and not steps:
        print(f"SKIP {directory.name}: no metrics")
        return
    source_name = directory.name
    existing = client.search_runs(
        [experiment_id], filter_string=f"tags.source_run = '{source_name}'", max_results=1
    )
    if existing:
        print(f"SKIP {source_name}: already imported ({existing[0].info.run_id})")
        return
    scheme = str(config.get("scheme", config.get("gc_scheme", "unknown")))
    n = int(config.get("num_workers", config.get("workers", 3)))
    steps_per_epoch = int(config.get("steps_per_epoch", 1))
    with mlflow.start_run(run_name=f"{scheme} | {source_name}") as run:
        mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))})
        mlflow.set_tags({
            "source_run": source_name,
            "scheme": scheme,
            "dataset": "CIFAR-10",
            "model": "ResNet-18",
            "hardware": device or str(config.get("device", "unspecified")),
            "origin": "local Ray JSONL import",
        })
        stamp = int(time.time() * 1000)
        metric_rows = []
        elapsed = 0.0
        for row in epochs:
            epoch = int(row["epoch"])
            elapsed += float(row["epoch_time_s"])
            add_metric(metric_rows, "quality/test_accuracy", row.get("test_accuracy"), epoch, stamp)
            add_metric(metric_rows, "quality/test_loss", row.get("test_loss"), epoch, stamp)
            add_metric(metric_rows, "quality/train_loss_reported", row.get("train_loss"), epoch, stamp)
            add_metric(metric_rows, "quality/train_eval_loss", row.get("train_eval_loss"), epoch, stamp)
            add_metric(metric_rows, "time/epoch_s", row.get("epoch_time_s"), epoch, stamp)
            add_metric(metric_rows, "time/cumulative_epoch_s", elapsed, epoch, stamp)
        selected = Counter()
        for row in steps:
            position = (int(row["epoch"]) - 1) * steps_per_epoch + int(row["step"])
            selected.update(int(w) for w in row.get("responders", []))
            timings = row.get("worker_timings", [])
            add_metric(metric_rows, "time/step_wait_s", row.get("wait_time_s"), position, stamp)
            add_metric(metric_rows, "resilience/stale_results", row.get("stale_results_seen", 0), position, stamp)
            add_metric(metric_rows, "resilience/selected_workers", len(row.get("responders", [])), position, stamp)
            add_metric(metric_rows, "communication/accepted_payload_mib", sum(w.get("gradient_bytes", 0) for w in timings) / 1048576, position, stamp)
            if timings:
                add_metric(metric_rows, "time/mean_selected_compute_s", mean(w["compute_time_s"] for w in timings), position, stamp)
                add_metric(metric_rows, "time/mean_selected_response_s", mean(w["response_time_s"] for w in timings), position, stamp)
            for worker in range(n):
                add_metric(metric_rows, f"workers/w{worker}_selected", int(worker in row.get("responders", [])), position, stamp)
            for w in timings:
                add_metric(metric_rows, f"workers/w{w['worker_id']}_response_s", w.get("response_time_s"), position, stamp)
                add_metric(metric_rows, f"workers/w{w['worker_id']}_compute_s", w.get("compute_time_s"), position, stamp)
        if epochs:
            add_metric(metric_rows, "summary/best_test_accuracy", max(e["test_accuracy"] for e in epochs), 0, stamp)
            add_metric(metric_rows, "summary/final_test_accuracy", epochs[-1]["test_accuracy"], 0, stamp)
            add_metric(metric_rows, "summary/final_test_loss", epochs[-1]["test_loss"], 0, stamp)
            add_metric(metric_rows, "summary/total_epoch_s", elapsed, 0, stamp)
            add_metric(metric_rows, "summary/completed_epochs", len(epochs), 0, stamp)
        if steps:
            add_metric(metric_rows, "summary/total_steps", len(steps), 0, stamp)
            add_metric(metric_rows, "summary/mean_step_wait_s", mean(s["wait_time_s"] for s in steps), 0, stamp)
            for worker in range(n):
                add_metric(metric_rows, f"summary/w{worker}_selected_count", selected[worker], 0, stamp)
        for i in range(0, len(metric_rows), 500):
            client.log_batch(run.info.run_id, metrics=metric_rows[i:i + 500])
        for filename in ("config.json", "epoch_metrics.jsonl", "step_metrics.jsonl"):
            if (directory / filename).is_file():
                mlflow.log_artifact(str(directory / filename), artifact_path="raw")
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            make_plots(temp, epochs, steps, n)
            mlflow.log_artifacts(str(temp), artifact_path="plots")
        print(f"IMPORTED {source_name}: {len(epochs)} epochs, {len(steps)} steps, {len(metric_rows)} metric points")


def main():
    parser = argparse.ArgumentParser(description="Import existing local Ray experiments into MLflow")
    parser.add_argument("--results", type=Path, default=Path("data/results"))
    parser.add_argument("--latest", action="store_true", help="Import only the latest experiment")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="Gradient Coding - CIFAR10")
    parser.add_argument("--device", default=None, help="Optional run label, e.g. mps or cpu")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.set_experiment(args.experiment)
    client = MlflowClient()
    found = candidates(args.results, args.latest)
    if not found:
        raise SystemExit(f"No result folders found in {args.results}")
    for directory in found:
        import_run(directory, client, exp.experiment_id, args.device)


if __name__ == "__main__":
    main()
