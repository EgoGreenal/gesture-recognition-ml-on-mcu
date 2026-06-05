###################################################################################################
# 20BN-Jester motion-channel hard-class oversampled dataset for ai8x-training
###################################################################################################
"""Cached 20BN-Jester motion-channel dataset with hard-class oversampling."""

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

# These are the classes that dominated the current model's validation confusions.
HARD_CLASS_REPEAT = {
    6: 2,   # Sliding Two Fingers Left
    7: 2,   # Sliding Two Fingers Right
    10: 3,  # Pushing Two Fingers Away
    11: 3,  # Pulling Two Fingers In
    12: 3,  # Rolling Hand Forward
    13: 3,  # Rolling Hand Backward
    14: 3,  # Turning Hand Clockwise
    15: 3,  # Turning Hand Counterclockwise
}


def _load_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload["data"], payload["labels"]


class CachedJesterSplit(Dataset):
    """Single cached validation split stored as uint8 motion-channel stacks."""

    def __init__(self, path: Path, args):
        self.data, self.labels = _load_cache(path)
        self.normalize = ai8x.normalize(args=args)

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, index):
        image = self.data[index].float().div(255.0)
        return self.normalize(image), self.labels[index]


class CachedJesterHardTrainWithValidation(Dataset):
    """Train dataset with duplicated hard-class train indices plus untouched validation indices."""

    def __init__(self, train_path: Path, val_path: Path, args):
        self.train_data, self.train_labels = _load_cache(train_path)
        self.val_data, self.val_labels = _load_cache(val_path)
        self.train_indices = []
        for index, label in enumerate(self.train_labels.tolist()):
            self.train_indices.extend([index] * HARD_CLASS_REPEAT.get(int(label), 1))
        self.train_count = len(self.train_indices)
        self.valid_indices = list(range(self.train_count, self.train_count + int(self.val_labels.numel())))
        self.normalize = ai8x.normalize(args=args)

    def __len__(self):
        return self.train_count + int(self.val_labels.numel())

    def __getitem__(self, index):
        if index < self.train_count:
            train_index = self.train_indices[index]
            image = self.train_data[train_index]
            label = self.train_labels[train_index]
        else:
            val_index = index - self.train_count
            image = self.val_data[val_index]
            label = self.val_labels[val_index]

        image = image.float().div(255.0)
        return self.normalize(image), label


def jester_motion12_hard_get_datasets(data, load_train=False, load_test=False):
    """Return hard-class oversampled motion12 Jester train/validation datasets."""
    data_dir, args = data
    cache_dir = Path(data_dir)

    train_dataset = None
    if load_train:
        train_dataset = CachedJesterHardTrainWithValidation(
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
        "name": "jester_motion12_hard",
        "input": (12, 64, 64),
        "output": LABELS,
        "loader": jester_motion12_hard_get_datasets,
    }
]
