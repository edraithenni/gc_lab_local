import os
import time

import ray
import torch
from torch.nn import functional as F

from coding import make_scheme
from data import TrainingBatches
from model import build_model, flatten_gradients, load_flat_parameters


@ray.remote(num_cpus=1, resources={"gc_worker": 1}, max_restarts=0)
class GCWorker:
    def __init__(self, worker_id, config):
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
        self.worker_id = worker_id
        self.config = config
        device_name = os.getenv("GC_DEVICE", "cpu")
        if device_name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; check native arm64 PyTorch")
        self.device = torch.device(device_name)
        self.model = build_model().to(self.device)
        self.batches = TrainingBatches(config, root=os.getenv("DATA_ROOT", "/data/cifar10"))
        self.scheme = make_scheme(config)
        self.gradient_cache = {}
        self.loss_cache = {}
        self.last_step = None
        self.profile_steps = os.getenv("GC_PROFILE_STEPS", "0") == "1"
        if self.profile_steps:
            print(f"worker={worker_id} device={next(self.model.parameters()).device} pid={os.getpid()}", flush=True)

    def compute_step(self, epoch, step, weights):
        started = time.perf_counter()
        if self.worker_id == self.config.slow_worker:
            time.sleep(self.config.straggler_delay)

        load_flat_parameters(self.model, weights)
        weight_time_s = time.perf_counter() - started
        data_time_s = 0.0
        backward_time_s = 0.0
        global_step = epoch * self.batches.steps_per_epoch + step
        partitions = self.scheme.partitions_to_compute(
            self.worker_id, global_step, self.gradient_cache, self.last_step
        )

        for partition in partitions:
            batch_started = time.perf_counter()
            inputs, labels = self.batches.get(partition, epoch, step)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            data_time_s += time.perf_counter() - batch_started
            compute_started = time.perf_counter()
            self.model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(inputs), labels)
            loss.backward()
            self.gradient_cache[partition] = flatten_gradients(self.model)
            self.loss_cache[partition] = loss.item()
            backward_time_s += time.perf_counter() - compute_started

        self.last_step = global_step
        encode_started = time.perf_counter()
        payload = self.scheme.encode(self.worker_id, self.gradient_cache)
        payload_cpu = payload.detach().cpu()
        encode_time_s = time.perf_counter() - encode_started
        loss = sum(self.loss_cache.values()) / len(self.loss_cache)
        elapsed = time.perf_counter() - started
        if self.profile_steps and epoch == 0 and step in (0, 1):
            print(
                f"worker={self.worker_id} step={step} fresh={len(partitions)} "
                f"weights={weight_time_s:.3f}s data={data_time_s:.3f}s "
                f"backward={backward_time_s:.3f}s encode_transfer={encode_time_s:.3f}s "
                f"total={elapsed:.3f}s",
                flush=True,
            )
        return payload_cpu, loss, elapsed
