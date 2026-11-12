import json
import os
import time
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import asdict
from pathlib import Path

import ray
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from coding import make_scheme
from config import Config
from coordinator import ResponseCollector
from data import evaluation_loader
from model import assign_flat_gradient, build_model, flatten_parameters
from worker import GCWorker


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        predictions = model(inputs)
        total_loss += torch.nn.functional.cross_entropy(predictions, labels, reduction="sum").item()
        total_correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)
    model.train()
    return total_loss / total_samples, total_correct / total_samples


def write_jsonl(path, record):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


@ray.remote(num_cpus=1, resources={"gc_parameter_server": 1})
class ParameterServer:
    def run_training(self):
        config = Config.from_env()
        torch.set_num_threads(int(os.getenv("AGGREGATOR_THREADS", "1")))
        torch.manual_seed(config.seed)
        scheme = make_scheme(config)
        model = build_model()
        parameter_count = sum(p.numel() for p in model.parameters())
        ema_model = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay),
            use_buffers=True,
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.lr, momentum=config.momentum,
            weight_decay=config.weight_decay, nesterov=config.momentum > 0,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        test_loader = evaluation_loader(config, root=os.getenv("DATA_ROOT", "/data/cifar10"))
        eval_device = torch.device(os.getenv("GC_EVAL_DEVICE", os.getenv("GC_DEVICE", "cpu")))
        eval_model = build_model().to(eval_device)
        steps_per_epoch = config.train_limit // config.num_partitions // config.batch_size

        run_id = (
            f"{config.scheme}-"
            f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-"
            f"{uuid4().hex[:6]}"
        )
        results = Path(os.getenv("RESULTS_DIR", "/data/results")) / run_id
        results.mkdir(parents=True, exist_ok=True)
        print(f"results={results}", flush=True)
        for name in ("step_metrics.jsonl", "epoch_metrics.jsonl"):
            (results / name).write_text("", encoding="utf-8")
        settings = asdict(config)
        settings.update({
            "gc_scheme": config.scheme,
            "workers": config.num_workers,
            "stragglers_tolerated": config.stragglers,
            "parameter_count": parameter_count,
            "steps_per_epoch": steps_per_epoch,
        })
        (results / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

        workers = [GCWorker.remote(i, config) for i in range(config.num_workers)]
        collector = ResponseCollector(workers)
        weights = ray.put(flatten_parameters(model))
        collector.submit(range(config.num_workers), 0, 0, weights)
        print(
            f"scheme={config.scheme} workers={config.num_workers} "
            f"stragglers={config.stragglers}",
            flush=True,
        )

        for epoch in range(config.epochs):
            epoch_start = time.perf_counter()
            training_loss = 0.0

            for step in range(steps_per_epoch):
                responses, loss, timing = collector.collect(epoch, step, weights, scheme)
                gradient = scheme.decode(responses, parameter_count)
                optimizer.zero_grad(set_to_none=True)
                assign_flat_gradient(model, gradient)
                optimizer.step()
                ema_model.update_parameters(model)
                training_loss += loss
                write_jsonl(results / "step_metrics.jsonl", {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "gc_scheme": config.scheme,
                    "responders": sorted(responses),
                    **timing,
                })

                next_epoch = epoch + (step + 1 == steps_per_epoch)
                next_step = (step + 1) % steps_per_epoch
                if next_epoch < config.epochs:
                    weights = ray.put(flatten_parameters(model))
                    collector.submit(responses, next_epoch, next_step, weights)

            train_time_s = time.perf_counter() - epoch_start
            eval_started = time.perf_counter()
            eval_model.load_state_dict(ema_model.module.state_dict())
            test_loss, test_accuracy = evaluate(eval_model, test_loader, eval_device)
            if eval_device.type == "mps":
                torch.mps.synchronize()
            eval_time_s = time.perf_counter() - eval_started
            record = {
                "epoch": epoch + 1,
                "train_loss": training_loss / steps_per_epoch,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "train_time_s": train_time_s,
                "eval_time_s": eval_time_s,
                "epoch_time_s": time.perf_counter() - epoch_start,
            }
            write_jsonl(results / "epoch_metrics.jsonl", record)
            print(json.dumps(record), flush=True)
            scheduler.step()

        torch.save(ema_model.module.state_dict(), results / "model_final.pt")
