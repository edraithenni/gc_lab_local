"""Tandon, Lei, Dimakis, Karampatziakis (ICML 2017): exact gradient coding."""

from itertools import combinations

import torch

from .base import StaticScheme


class ClassicGC(StaticScheme):
    def __init__(self, *, num_workers, num_partitions, stragglers, seed):
        if num_workers != num_partitions or not 0 <= stragglers < num_workers:
            raise ValueError("Classic GC requires 0 <= s < n = k")
        self.n, self.s = num_workers, stragglers
        self.B = self._coding_matrix(seed)
        self.decoders = self._decoders()

    def _coding_matrix(self, seed):
        # The n=3, s=1 matrix is exactly the construction from Fig. 1.
        if self.s == 0:
            return torch.eye(self.n)
        if self.n == 3 and self.s == 1:
            return torch.tensor(
                [[0.5, 1, 0], [0, 1, -1], [0.5, 0, 1]],
                dtype=torch.float32,
            )

        # Algorithm 2: H*1=0; each coding row has cyclic support of size s+1.
        generator = torch.Generator().manual_seed(seed)
        H = torch.randn(self.s, self.n, dtype=torch.float64, generator=generator)
        H[:, -1] = -H[:, :-1].sum(dim=1)
        B = torch.zeros(self.n, self.n, dtype=torch.float64)
        for worker in range(self.n):
            cols = [(worker + i) % self.n for i in range(self.s + 1)]
            B[worker, cols[0]] = 1
            B[worker, cols[1:]] = torch.linalg.solve(
                H[:, cols[1:]], -H[:, cols[0]]
            )
        return B.float()

    def _decoders(self):
        # Decode using B_S^T*a=1 for every survivor set of n-s workers.
        decoders = {}
        for subset in combinations(range(self.n), self.n - self.s):
            rows = self.B[list(subset)].double()
            ones = torch.ones(self.n, dtype=torch.float64)
            decoders[subset] = torch.linalg.lstsq(rows.T, ones).solution.float()
        return decoders

    def assigned_partitions(self, worker_id):
        return torch.nonzero(self.B[worker_id], as_tuple=False).flatten().tolist()

    def encode(self, worker_id, gradients):
        return sum(
            self.B[worker_id, shard].item() * gradient
            for shard, gradient in gradients.items()
        )

    def can_decode(self, workers):
        return len(set(workers)) >= self.n - self.s

    def decode(self, responses, gradient_size):
        selected = tuple(sorted(responses)[:self.n - self.s])
        coefficients = self.decoders[selected]
        result = torch.zeros_like(responses[selected[0]])
        for worker_id, coefficient in zip(selected, coefficients):
            result.add_(responses[worker_id], alpha=coefficient.item())
        return result / self.n
