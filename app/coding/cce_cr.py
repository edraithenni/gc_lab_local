"""Ye and Abbe (ICML 2018): communication-computation efficient cyclic GC."""

from math import ceil

import torch

from .base import StaticScheme


class CCEGCCR(StaticScheme):
    def __init__(
        self, *, num_workers, num_partitions, stragglers, communication_factor
    ):
        # The optimal construction uses d=s+m assigned partitions per worker.
        self.n = num_workers
        self.k = num_partitions
        self.s = stragglers
        self.m = communication_factor
        self.d = self.s + self.m
        if self.n != self.k or self.m < 1 or not 0 <= self.s < self.n or self.d > self.n:
            raise ValueError("CCE-GC-CR requires 0 <= s < n = k and 1 <= m <= n-s")

        self.required = self.n - self.s
        values = [0.0] if self.n % 2 else []
        for i in range(self.n // 2):
            v = 1.0 + i / 2.0
            values.extend((-v, v))
        self.theta = torch.tensor(sorted(values), dtype=torch.float64)

        # V contains the selected workers' Vandermonde evaluation columns.
        powers = torch.arange(self.required, dtype=torch.float64)
        self.V = self.theta.unsqueeze(0).pow(powers.unsqueeze(1))
        self.polynomials = self._build_polynomials()
        self.coefficients = torch.zeros(self.n, self.n, self.m, dtype=torch.float64)
        for worker in range(self.n):
            for partition, family in enumerate(self.polynomials):
                for coordinate, polynomial in enumerate(family):
                    self.coefficients[worker, partition, coordinate] = self._evaluate(
                        polynomial, self.theta[worker].item()
                    )

    def assigned_partitions(self, worker_id):
        return [(worker_id + i) % self.n for i in range(self.d)]

    def encode(self, worker_id, gradients):
        # Coding across m gradient coordinates reduces each message to ceil(|g|/m).
        first = next(iter(gradients.values()))
        rows = ceil(first.numel() / self.m)
        result = torch.zeros(rows, dtype=first.dtype, device=first.device)
        for partition, gradient in gradients.items():
            padded = torch.nn.functional.pad(
                gradient, (0, rows * self.m - gradient.numel())
            )
            coefficients = self.coefficients[worker_id, partition].to(
                device=first.device, dtype=first.dtype
            )
            result.add_(padded.view(rows, self.m) @ coefficients)
        return result

    def can_decode(self, workers):
        return len(set(workers)) >= self.required

    def decode(self, responses, gradient_size):
        # Solve the Vandermonde system and extract the m systematic columns.
        selected = sorted(responses)[:self.required]
        first = responses[selected[0]]
        Y = torch.stack([responses[w] for w in selected], dim=1).double()
        V_sub = self.V[:, selected]
        rows = torch.linalg.solve(V_sub.T, Y.T).T
        blocks = rows[:, self.n - self.d:self.n - self.s]
        recovered = blocks.contiguous().reshape(-1)[:gradient_size]
        return recovered.to(dtype=first.dtype, device=first.device) / self.k

    def _build_polynomials(self):
        # Recurrence from Eq. (9); coefficients use ascending polynomial powers.
        if self.d == self.n:
            return [
                [
                    [float(power == coordinate) for power in range(self.m)]
                    for coordinate in range(self.m)
                ]
                for _ in range(self.n)
            ]

        missing = self.n - self.d
        families = []
        for partition in range(self.n):
            base = [1.0]
            for offset in range(1, missing + 1):
                root = self.theta[(partition + offset) % self.n].item()
                next_base = [0.0] * (len(base) + 1)
                for power, value in enumerate(base):
                    next_base[power] -= root * value
                    next_base[power + 1] += value
                base = next_base
            base.extend([0.0] * (self.required - len(base)))
            family = [base]
            for _ in range(1, self.m):
                previous = family[-1]
                cancel = previous[missing - 1]
                shifted = [0.0] + previous[:-1]
                family.append([
                    shifted[i] - cancel * base[i]
                    for i in range(self.required)
                ])
            families.append(family)
        return families

    @staticmethod
    def _evaluate(coefficients, x):
        value = 0.0
        for coefficient in reversed(coefficients):
            value = value * x + coefficient
        return value
