from itertools import combinations
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from coding.pipelined_fr import PipelinedFR


def test_configuration():
    try:
        PipelinedFR(num_workers=3, num_partitions=3, stragglers=1)
        raise AssertionError("Expected an invalid FR configuration")
    except ValueError:
        pass


def test_recovery(n):
    scheme = PipelinedFR(num_workers=n, num_partitions=n, stragglers=1)
    torch.manual_seed(123)
    gradients = {p: torch.randn(17) for p in range(n)}
    messages = {
        w: scheme.encode(w, gradients)
        for w in range(n)
    }
    expected = torch.stack(list(gradients.values())).mean(0)

    for workers in combinations(range(n), n - 1):
        assert scheme.can_decode(workers)
        actual = scheme.decode({w: messages[w] for w in workers}, 17)
        assert torch.allclose(actual, expected, atol=1e-6), workers

    assert not scheme.can_decode(range(n - 2))
    assert scheme.partitions_to_compute(0, 0, {}, None) == [0, 1]
    assert scheme.partitions_to_compute(0, 1, gradients, 0) == [0]
    assert scheme.partitions_to_compute(0, 2, gradients, 1) == [1]
    assert scheme.partitions_to_compute(0, 4, gradients, 1) == [0, 1]


if __name__ == "__main__":
    test_configuration()
    test_recovery(4)
    test_recovery(6)
    print("PGC-FR checks passed")
