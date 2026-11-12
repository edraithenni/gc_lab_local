"""Kadhe, Koyluoglu, Ramchandran (2020): communication-efficient FR-GC."""

from math import ceil

import torch

from .base import StaticScheme


class CCEGCFR(StaticScheme):
    def __init__(
        self, *, num_workers, num_partitions, stragglers,
        communication_factor, seed
    ):
        # The Gaussian MDS code has [N,K,delta]=[m+s,m,s+1].
        self.n = num_workers
        self.k = num_partitions
        self.s = stragglers
        self.K = communication_factor
        self.N = self.K + self.s
        if self.N > self.n or self.n % self.N or self.k * self.N % self.n or self.K < 1:
            raise ValueError("CommFR requires m+s to divide n and n to divide k*(m+s)")
        self.groups = self.n // self.N
        self.partitions_per_group = self.k // self.groups
        generator = torch.Generator().manual_seed(seed)
        self.G = torch.randn(self.K, self.N, dtype=torch.float64, generator=generator)

    def assigned_partitions(self, worker_id):
        start = (worker_id // self.N) * self.partitions_per_group
        return list(range(start, start + self.partitions_per_group))

    def encode(self, worker_id, gradients):
        # Eq. (4): concatenate K coordinate blocks into columns of X.
        vector = sum(gradients.values())
        rows = ceil(vector.numel() / self.K)
        padded = torch.nn.functional.pad(
            vector, (0, rows * self.K - vector.numel())
        )
        matrix = padded.view(self.K, rows).T.contiguous()
        column = self.G[:, worker_id % self.N].to(
            device=vector.device, dtype=vector.dtype
        )
        return (matrix @ column).contiguous()

    def can_decode(self, workers):
        counts = [0] * self.groups
        for worker_id in set(workers):
            counts[worker_id // self.N] += 1
        return all(count >= self.K for count in counts)

    def decode(self, responses, gradient_size):
        # Every group contributes K columns; solve Y=X G_sub and sum recovered X.
        first = next(iter(responses.values()))
        result = torch.zeros(
            gradient_size, dtype=first.dtype, device=first.device
        )
        for group in range(self.groups):
            selected = sorted(
                w for w in responses if w // self.N == group
            )[:self.K]
            Y = torch.stack([responses[w] for w in selected], dim=1).double()
            G_sub = self.G[:, [w % self.N for w in selected]]
            X = torch.linalg.solve(G_sub.T, Y.T).T
            vector = X.T.contiguous().reshape(-1)[:gradient_size]
            result.add_(vector.to(dtype=result.dtype, device=result.device))
        return result / self.k
