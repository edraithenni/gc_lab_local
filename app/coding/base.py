from typing import Mapping, Protocol

import torch


class CodingScheme(Protocol):
    def assigned_partitions(self, worker_id: int) -> list[int]: ...

    def partitions_to_compute(
        self,
        worker_id: int,
        step: int,
        cache: Mapping[int, torch.Tensor],
        last_step: int | None,
    ) -> list[int]: ...

    def encode(
        self, worker_id: int, gradients: Mapping[int, torch.Tensor]
    ) -> torch.Tensor: ...

    def can_decode(self, workers) -> bool: ...

    def decode(
        self, responses: Mapping[int, torch.Tensor], gradient_size: int
    ) -> torch.Tensor: ...


class StaticScheme:
    def partitions_to_compute(self, worker_id, step, cache, last_step):
        return self.assigned_partitions(worker_id)
