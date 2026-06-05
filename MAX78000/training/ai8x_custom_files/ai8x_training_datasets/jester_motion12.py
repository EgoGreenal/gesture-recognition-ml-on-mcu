###################################################################################################
# 20BN-Jester motion-channel cached dataset for ai8x-training
###################################################################################################
"""Cached 20BN-Jester motion-channel gesture dataset."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

import ai8x


LABELS = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things",
]


def _load_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload["data"], payload["labels"]


class CachedJesterSplit(Dataset):
    """Single cached split stored as uint8 motion-channel stacks."""

    def __init__(self, path: Path, args):
        self.data, self.labels = _load_cache(path)
        self.normalize = ai8x.normalize(args=args)

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, index):
        image = self.data[index].float().div(255.0)
        return self.normalize(image), self.labels[index]


class CachedJesterTrainWithValidation(Dataset):
    """Train dataset with explicit validation indices from the cached validation split."""

    def __init__(self, train_path: Path, val_path: Path, args):
        self.train_data, self.train_labels = _load_cache(train_path)
        self.val_data, self.val_labels = _load_cache(val_path)
        self.train_count = int(self.train_labels.numel())
        self.valid_indices = list(range(self.train_count, self.train_count + int(self.val_labels.numel())))
        self.normalize = ai8x.normalize(args=args)

    def __len__(self):
        return self.train_count + int(self.val_labels.numel())

    def __getitem__(self, index):
        if index < self.train_count:
            image = self.train_data[index]
            label = self.train_labels[index]
        else:
            val_index = index - self.train_count
            image = self.val_data[val_index]
            label = self.val_labels[val_index]

        image = image.float().div(255.0)
        return self.normalize(image), label


def jester_motion12_get_datasets(data, load_train=False, load_test=False):
    """Return cached 12-plane motion-channel Jester train/validation datasets."""
    data_dir, args = data
    cache_dir = Path(data_dir)

    train_dataset = None
    if load_train:
        train_dataset = CachedJesterTrainWithValidation(
            cache_dir / "train.pt",
            cache_dir / "val.pt",
            args,
        )

    test_dataset = None
    if load_test:
        test_dataset = CachedJesterSplit(cache_dir / "val.pt", args)
        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
            test_dataset.labels = test_dataset.labels[:1]

    return train_dataset, test_dataset


datasets = [
    {
        "name": "jester_motion12",
        "input": (12, 64, 64),
        "output": LABELS,
        "loader": jester_motion12_get_datasets,
    }
]
