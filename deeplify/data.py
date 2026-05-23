from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

CLASS_NAMES = ("LP", "CR", "PO", "ND")


class RIAWELCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        image_size: int = 224,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.image_size = image_size
        self.seed = seed

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def setup(self, stage: str | None = None) -> None:
        split_dirs = {name: self.data_dir / name for name in ("train", "val", "test")}
        has_explicit_splits = all(path.exists() for path in split_dirs.values())

        if has_explicit_splits:
            if stage in (None, "fit"):
                self.train_dataset = datasets.ImageFolder(
                    root=split_dirs["train"], transform=self.train_transform
                )
                self.val_dataset = datasets.ImageFolder(
                    root=split_dirs["val"], transform=self.eval_transform
                )
            if stage in (None, "test"):
                self.test_dataset = datasets.ImageFolder(
                    root=split_dirs["test"], transform=self.eval_transform
                )
            return

        index_dataset = datasets.ImageFolder(root=self.data_dir)
        train_len = int(len(index_dataset) * self.train_split)
        val_len = int(len(index_dataset) * self.val_split)
        test_len = len(index_dataset) - train_len - val_len

        generator = torch.Generator().manual_seed(self.seed)
        train_indices, val_indices, test_indices = random_split(
            range(len(index_dataset)),
            lengths=[train_len, val_len, test_len],
            generator=generator,
        )

        train_base = datasets.ImageFolder(root=self.data_dir, transform=self.train_transform)
        eval_base = datasets.ImageFolder(root=self.data_dir, transform=self.eval_transform)

        if stage in (None, "fit"):
            self.train_dataset = Subset(train_base, train_indices.indices)
            self.val_dataset = Subset(eval_base, val_indices.indices)
        if stage in (None, "test"):
            self.test_dataset = Subset(eval_base, test_indices.indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
