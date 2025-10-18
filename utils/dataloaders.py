"""Lightning datamodule abstraction for MNIST and SUN datasets."""

import math

import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, SUN397


class GenericImageDataModule(pl.LightningDataModule):
    """DataModule that wraps MNIST and SUN datasets with configurable preprocessing."""

    def __init__(
        self,
        name: str = "mnist",
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.1,
        sun_resize: int = 64,
        sun_to_grayscale: bool = True,
        split_seed: int = 42,
    ):
        super().__init__()
        self.name = name.lower()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.sun_resize = sun_resize
        self.sun_to_grayscale = sun_to_grayscale
        self.split_seed = split_seed
        self.input_shape = self._infer_input_shape()
        self.transform = self._build_transform()
        self.train_ds = None
        self.val_ds = None

    def _infer_input_shape(self):
        """Return the tensor shape produced by the active dataset configuration."""
        if self.name == "mnist":
            return (1, 28, 28)
        if self.name == "sun":
            channels = 1 if self.sun_to_grayscale else 3
            return (channels, self.sun_resize, self.sun_resize)
        raise ValueError(f"Unknown dataset {self.name}")

    def _build_transform(self):
        """Assemble torchvision transforms specific to the chosen dataset."""
        if self.name == "mnist":
            return T.Compose([T.ToTensor()])
        if self.name == "sun":
            transforms = [T.Resize((self.sun_resize, self.sun_resize))]
            if self.sun_to_grayscale:
                transforms.append(T.Grayscale())
            transforms.append(T.ToTensor())
            return T.Compose(transforms)
        raise ValueError(f"Unknown dataset {self.name}")

    def prepare_data(self):
        """Download dataset assets when they are not yet present on disk."""
        if self.name == "mnist":
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)
        elif self.name == "sun":
            SUN397(self.data_dir, transform=self.transform, download=True)

    def setup(self, stage=None):
        """Create train/val splits and apply transforms ahead of dataloader construction."""
        if self.train_ds is not None and self.val_ds is not None:
            return

        if self.name == "mnist":
            self.train_ds = MNIST(self.data_dir, train=True, transform=self.transform)
            self.val_ds = MNIST(self.data_dir, train=False, transform=self.transform)
        elif self.name == "sun":
            full_ds = SUN397(self.data_dir, transform=self.transform, download=False)
            val_len = max(1, int(math.floor(len(full_ds) * self.val_split)))
            train_len = len(full_ds) - val_len
            generator = torch.Generator().manual_seed(self.split_seed)
            self.train_ds, self.val_ds = random_split(
                full_ds, [train_len, val_len], generator=generator
            )
        else:
            raise ValueError(f"Unknown dataset {self.name}")

    def train_dataloader(self):
        """Return the training dataloader for the configured dataset."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Return the validation dataloader for the configured dataset."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
