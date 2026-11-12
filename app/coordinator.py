import time

import ray


class ResponseCollector:
    def __init__(self, workers):
        self.workers = workers
        self.pending = {}
        self.failed = {}

    def submit(self, worker_ids, epoch, step, weights):
        for worker_id in worker_ids:
            ref = self.workers[worker_id].compute_step.remote(epoch, step, weights)
            self.pending[ref] = (worker_id, epoch, step, time.perf_counter())

    def collect(self, epoch, step, weights, scheme):
        responses = {}
        losses = []
        timings = []
        stale = 0
        started = time.perf_counter()

        while not scheme.can_decode(responses):
            if not self.pending:
                raise RuntimeError(f"Insufficient worker responses: {self.failed}")

            ready, _ = ray.wait(list(self.pending), num_returns=1)
            ref = ready[0]
            worker_id, task_epoch, task_step, sent_at = self.pending.pop(ref)

            try:
                gradient, loss, compute_seconds = ray.get(ref)
            except ray.exceptions.RayError as error:
                self.failed[worker_id] = str(error)
                continue

            if (task_epoch, task_step) != (epoch, step):
                stale += 1
                self.submit([worker_id], epoch, step, weights)
                continue

            responses[worker_id] = gradient
            losses.append(loss)
            timings.append({
                "worker_id": worker_id,
                "compute_time_s": compute_seconds,
                "response_time_s": time.perf_counter() - sent_at,
                "gradient_bytes": gradient.numel() * gradient.element_size(),
            })

        return responses, sum(losses) / len(losses), {
            "wait_time_s": time.perf_counter() - started,
            "stale_results_seen": stale,
            "worker_timings": timings,
        }
