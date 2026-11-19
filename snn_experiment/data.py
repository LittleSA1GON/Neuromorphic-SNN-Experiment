"""Deterministic stratified partitions with separate augmentation pipelines."""
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def stratified_split(targets, validation_per_class=500, train_per_class=None, seed=42):
    """Return disjoint original CIFAR training indices, never test indices."""
    if validation_per_class < 1 or (train_per_class is not None and train_per_class < 1):
        raise ValueError("subset sizes must be positive")
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    training, validation = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(targets == label))
        if len(indices) <= validation_per_class:
            raise ValueError(f"class {label} has insufficient training samples")
        available = indices[validation_per_class:]
        count = len(available) if train_per_class is None else train_per_class
        if count > len(available):
            raise ValueError(f"class {label} has insufficient samples for train_per_class")
        training.extend(available[:count].tolist())
        validation.extend(indices[:validation_per_class].tolist())
    return sorted(training), sorted(validation)


def stratified_subset(targets, per_class, seed):
    if per_class is not None and per_class < 1:
        raise ValueError("test_per_class must be positive")
    rng = np.random.default_rng(seed)
    labels = np.asarray(targets)
    selected = []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = len(indices) if per_class is None else per_class
        if count > len(indices):
            raise ValueError(f"insufficient test samples for class {label}")
        selected.extend(indices[:count].tolist())
    return sorted(selected)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(config):
    if config.smoke:
        datasets = {}
        for offset, (name, size) in enumerate((("train", 40), ("validation", 20), ("test", 20))):
            generator = torch.Generator().manual_seed(config.seed + offset)
            datasets[name] = TensorDataset(torch.randn(size, 3, 32, 32, generator=generator),
                                           torch.arange(size) % 10)
        manifest = {"dataset": "synthetic_random_tensors", "scientific_result": False,
                    "counts": {key: len(value) for key, value in datasets.items()}}
    else:
        from torchvision import datasets as tv_datasets, transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
            transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        evaluation_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        augmented = tv_datasets.CIFAR10(config.data_dir, train=True, download=config.download, transform=train_transform)
        unaugmented = tv_datasets.CIFAR10(config.data_dir, train=True, download=False, transform=evaluation_transform)
        testing = tv_datasets.CIFAR10(config.data_dir, train=False, download=config.download, transform=evaluation_transform)
        train_indices, validation_indices = stratified_split(
            augmented.targets, config.validation_per_class, config.train_per_class, config.seed)
        test_indices = stratified_subset(testing.targets, config.test_per_class, config.seed)
        datasets = {"train": Subset(augmented, train_indices),
                    "validation": Subset(unaugmented, validation_indices), "test": Subset(testing, test_indices)}
        manifest = {"dataset": "CIFAR-10", "scientific_result": True,
                    "scope": "subset_pilot" if config.train_per_class or config.test_per_class else "full_split",
                    "counts": {key: len(value) for key, value in datasets.items()},
                    "training_source": "official training partition", "test_source": "official test partition",
                    "train_indices": train_indices, "validation_indices": validation_indices,
                    "test_indices": test_indices, "normalization_mean": MEAN, "normalization_std": STD}
    loaders = {key: DataLoader(value, batch_size=config.batch_size, shuffle=key == "train",
                              num_workers=config.workers, drop_last=False, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(config.seed + i))
               for i, (key, value) in enumerate(datasets.items())}
    encoded = repr({key: value for key, value in manifest.items() if key.endswith("indices")}).encode()
    manifest["split_sha256"] = hashlib.sha256(encoded).hexdigest()
    return loaders, manifest
