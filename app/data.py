import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATA_ROOT = "/data/cifar10"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def dataset(train, root=DATA_ROOT):
    augmentation = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if train else []
    )
    transform = transforms.Compose(
        augmentation + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    return datasets.CIFAR10(
        root=root, train=train, download=False, transform=transform
    )


class TrainingBatches:
    def __init__(self, config, root=DATA_ROOT):
        self.config = config
        self.dataset = dataset(True, root)
        if config.train_limit % config.num_partitions:
            raise ValueError("TRAIN_LIMIT must be divisible by NUM_PARTITIONS")
        self.partition_size = config.train_limit // config.num_partitions
        if self.partition_size % config.batch_size:
            raise ValueError(
                "TRAIN_LIMIT / NUM_PARTITIONS must be divisible by BATCH_SIZE"
            )
        self.steps_per_epoch = self.partition_size // config.batch_size
        self.current_epoch = None
        self.orders = None

    def get(self, partition_id, epoch, step):
        if self.current_epoch != epoch:
            generator = torch.Generator().manual_seed(self.config.seed + epoch)
            self.orders = []
            for partition in range(self.config.num_partitions):
                permutation = torch.randperm(
                    self.partition_size, generator=generator
                ).tolist()
                start = partition * self.partition_size
                self.orders.append([start + i for i in permutation])
            self.current_epoch = epoch

        start = step * self.config.batch_size
        indices = self.orders[partition_id][
            start:start + self.config.batch_size
        ]
        global_step = epoch * self.steps_per_epoch + step
        aug_seed = (
            self.config.seed + 1_000_000
            + global_step * self.config.num_partitions + partition_id
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(aug_seed)
            loader = DataLoader(
                Subset(self.dataset, indices),
                batch_size=len(indices),
                num_workers=0,
            )
            return next(iter(loader))


def evaluation_loader(config, root=DATA_ROOT):
    data = dataset(False, root)
    subset = Subset(data, range(min(config.test_limit, len(data))))
    return DataLoader(subset, batch_size=256, num_workers=0)
