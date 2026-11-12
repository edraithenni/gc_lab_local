from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    scheme: str = "classic"
    num_workers: int = 3
    num_partitions: int = 3
    stragglers: int = 1
    communication_factor: int = 2
    seed: int = 1234
    gc_seed: int = 1234
    train_limit: int = 9000
    test_limit: int = 2000
    batch_size: int = 40
    epochs: int = 100
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    ema_decay: float = 0.99
    slow_worker: int = -1
    straggler_delay: float = 0.0

    @classmethod
    def from_env(cls):
        return cls(
            scheme=os.getenv("GC_SCHEME", "classic").strip().lower(),
            num_workers=int(os.getenv("NUM_WORKERS", "3")),
            num_partitions=int(os.getenv("NUM_PARTITIONS", os.getenv("NUM_WORKERS", "3"))),
            stragglers=int(os.getenv("GC_STRAGGLERS", "1")),
            communication_factor=int(os.getenv("CCE_M", "2")),
            seed=int(os.getenv("SEED", "1234")),
            gc_seed=int(os.getenv("GC_SEED", "1234")),
            train_limit=int(os.getenv("TRAIN_LIMIT", "9000")),
            test_limit=int(os.getenv("TEST_LIMIT", "2000")),
            batch_size=int(os.getenv("BATCH_SIZE", "40")),
            epochs=int(os.getenv("EPOCHS", "100")),
            lr=float(os.getenv("LR", "0.1")),
            momentum=float(os.getenv("MOMENTUM", "0.9")),
            weight_decay=float(os.getenv("WEIGHT_DECAY", "0.0005")),
            ema_decay=float(os.getenv("EMA_DECAY", "0.99")),
            slow_worker=int(os.getenv("SLOW_WORKER", "-1")),
            straggler_delay=float(os.getenv("STRAGGLER_DELAY", "0")),
        )
