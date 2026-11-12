"""Su and Li (2026): Pipelined Gradient Coding with cyclic repetition."""


class PipelinedGC:
    def __init__(self, *, num_workers, num_partitions, stragglers):
        # Each worker caches c=s+1 cyclically adjacent partition gradients.
        if num_workers != num_partitions or not 0 <= stragglers < num_workers:
            raise ValueError("PGC-CR requires 0 <= s < n = k")
        self.n, self.s = num_workers, stragglers
        self.c = self.s + 1
        self.required = self.n - self.s

    def assigned_partitions(self, worker_id):
        return [(worker_id + offset) % self.n for offset in range(self.c)]

    def partitions_to_compute(self, worker_id, step, cache, last_step):
        # Refill at warm-up or after missing steps, otherwise rotate one partition.
        if not cache or last_step is None or step != last_step + 1:
            return self.assigned_partitions(worker_id)
        return [(worker_id + (step - 1) % self.c) % self.n]

    def encode(self, worker_id, gradients):
        # All cached gradients enter with coefficient +1, including stale values.
        return sum(
            gradients[partition]
            for partition in self.assigned_partitions(worker_id)
        )

    def can_decode(self, workers):
        return len(set(workers)) >= self.required

    def decode(self, responses, gradient_size):
        # PGC is approximate: E[multiplicity]=(n-s)c/n and n*mu=(n-s)c.
        selected = sorted(responses)[:self.required]
        return sum(responses[worker] for worker in selected) / (
            self.required * self.c
        )
