import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from coding.classic import ClassicGC
from coding.cce_fr import CCEGCFR
from coding.cce_cr import CCEGCCR
from coding.pipelined import PipelinedGC


def test_exact(scheme, n, size):
    torch.manual_seed(16)
    gradients = {i: torch.randn(size) for i in range(n)}
    messages = {
        worker: scheme.encode(worker, {p: gradients[p] for p in scheme.assigned_partitions(worker)})
        for worker in range(n)
    }
    for length in range(1, n + 1):
        for selected in itertools.combinations(range(n), length):
            if scheme.can_decode(selected):
                actual = scheme.decode({w: messages[w] for w in selected}, size)
                expected = torch.stack(list(gradients.values())).mean(0)
                assert torch.allclose(actual, expected, atol=2e-4, rtol=2e-4), (scheme, selected, (actual - expected).abs().max())


def test_pipelined():
    scheme = PipelinedGC(num_workers=3, num_partitions=3, stragglers=1)
    gradients = {i: torch.randn(11) for i in range(3)}
    messages = {w: scheme.encode(w, {p: gradients[p] for p in scheme.assigned_partitions(w)}) for w in range(3)}
    recovered = []
    for selected in itertools.combinations(range(3), 2):
        recovered.append(scheme.decode({w: messages[w] for w in selected}, 11))
    assert torch.allclose(torch.stack(recovered).mean(0), torch.stack(list(gradients.values())).mean(0), atol=1e-6)
    assert scheme.partitions_to_compute(0, 0, {}, None) == [0, 1]
    assert scheme.partitions_to_compute(0, 1, {0: gradients[0], 1: gradients[1]}, 0) == [0]
    assert scheme.partitions_to_compute(0, 2, {0: gradients[0], 1: gradients[1]}, 1) == [1]
    assert scheme.partitions_to_compute(0, 4, {0: gradients[0], 1: gradients[1]}, 1) == [0, 1]


if __name__ == "__main__":
    for size in [7, 8, 19, 101]:
        test_exact(ClassicGC(num_workers=3, num_partitions=3, stragglers=1, seed=1234), 3, size)
        test_exact(CCEGCFR(num_workers=3, num_partitions=3, stragglers=1, communication_factor=2, seed=1234), 3, size)
        test_exact(CCEGCCR(num_workers=3, num_partitions=3, stragglers=1, communication_factor=2), 3, size)
        test_exact(ClassicGC(num_workers=4, num_partitions=4, stragglers=1, seed=1234), 4, size)
        test_exact(CCEGCCR(num_workers=4, num_partitions=4, stragglers=1, communication_factor=2), 4, size)
        test_exact(CCEGCFR(num_workers=6, num_partitions=6, stragglers=1, communication_factor=2, seed=1234), 6, size)
    test_pipelined()
    print("All coding checks passed")
