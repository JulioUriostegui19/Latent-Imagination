"""Lightning datamodule abstraction for MNIST and SUN datasets."""
# Mathematical operations for dataset splitting calculations
import math

# PyTorch libraries for data handling and image transformations
import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, SUN397


class GenericImageDataModule(pl.LightningDataModule):
    """DataModule that wraps MNIST and SUN datasets with configurable preprocessing."""
    # Generic data module that handles both MNIST and SUN397 datasets with appropriate preprocessing
    # Provides unified interface for different image datasets in VAE experiments

    def __init__(
        self,
        name: str = "mnist",  # Dataset name ('mnist' or 'sun')
        data_dir: str = "./data",  # Directory to store downloaded datasets
        batch_size: int = 128,  # Mini-batch size for training and validation
        num_workers: int = 4,  # Number of worker processes for data loading
        pin_memory: bool = True,  # Whether to pin memory for faster GPU transfer
        val_split: float = 0.1,  # Fraction of training data to use for validation (SUN only)
        sun_resize: int = 64,  # Target size for SUN dataset images (square resize)
        sun_to_grayscale: bool = True,  # Whether to convert SUN images to grayscale
        split_seed: int = 42,  # Random seed for reproducible train/val splits
    ):
        super().__init__()
        self.name = name.lower()  # Normalize dataset name to lowercase
        self.data_dir = data_dir  # Store data directory path
        self.batch_size = batch_size  # Store batch size for dataloaders
        self.num_workers = num_workers  # Store number of worker processes
        self.pin_memory = pin_memory  # Store memory pinning flag
        self.val_split = val_split  # Store validation split fraction
        self.sun_resize = sun_resize  # Store SUN resize parameter
        self.sun_to_grayscale = sun_to_grayscale  # Store SUN grayscale conversion flag
        self.split_seed = split_seed  # Store random seed for reproducibility
        self.input_shape = self._infer_input_shape()  # Determine tensor shape based on dataset
        self.transform = self._build_transform()  # Build appropriate image transforms
        self.train_ds = None  # Will store training dataset
        self.val_ds = None  # Will store validation dataset

    def _infer_input_shape(self):
        """Return the tensor shape produced by the active dataset configuration."""
        # Return appropriate tensor shape based on dataset configuration
        if self.name == "mnist":
            return (1, 28, 28)  # MNIST: 1 channel, 28x28 pixels
        if self.name == "sun":
            channels = 1 if self.sun_to_grayscale else 3  # SUN: 1 or 3 channels based on config
            return (channels, self.sun_resize, self.sun_resize)  # Square resize to specified size
        raise ValueError(f"Unknown dataset {self.name}")

    def _build_transform(self):
        """Assemble torchvision transforms specific to the chosen dataset."""
        # Build appropriate transform pipeline based on dataset
        if self.name == "mnist":
            # Normalize to [-1, 1] via mean=0.5, std=0.5 after ToTensor() yields [0,1]
            return T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        if self.name == "sun":
            transforms = [T.Resize((self.sun_resize, self.sun_resize))]  # SUN: resize to square
            if self.sun_to_grayscale:
                transforms.append(T.Grayscale())  # Optionally convert to grayscale
            transforms.append(T.ToTensor())  # Convert to tensor [0,1]
            # Normalize to [-1, 1]
            if self.sun_to_grayscale:
                transforms.append(T.Normalize((0.5,), (0.5,)))
            else:
                transforms.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            return T.Compose(transforms)
        raise ValueError(f"Unknown dataset {self.name}")

    def prepare_data(self):
        """Download dataset assets when they are not yet present on disk."""
        # Download datasets if not already present (called only on main process in distributed training)
        if self.name == "mnist":
            MNIST(self.data_dir, train=True, download=True)  # Download training set
            MNIST(self.data_dir, train=False, download=True)  # Download test set
        elif self.name == "sun":
            SUN397(self.data_dir, transform=self.transform, download=True)  # Download SUN dataset

    def setup(self, stage=None):
        """Create train/val splits and apply transforms ahead of dataloader construction."""
        # Skip if datasets already created (avoid recreating on multiple calls)
        if self.train_ds is not None and self.val_ds is not None:
            return

        # Create datasets based on selected dataset name
        if self.name == "mnist":
            # MNIST: use official train/test split
            self.train_ds = MNIST(self.data_dir, train=True, transform=self.transform)
            self.val_ds = MNIST(self.data_dir, train=False, transform=self.transform)
        elif self.name == "sun":
            # SUN: create random train/val split from full dataset
            full_ds = SUN397(self.data_dir, transform=self.transform, download=False)
            val_len = max(1, int(math.floor(len(full_ds) * self.val_split)))  # Ensure at least 1 sample
            train_len = len(full_ds) - val_len
            generator = torch.Generator().manual_seed(self.split_seed)  # For reproducible splits
            self.train_ds, self.val_ds = random_split(
                full_ds, [train_len, val_len], generator=generator
            )
        else:
            raise ValueError(f"Unknown dataset {self.name}")

    def train_dataloader(self):
        """Return the training dataloader for the configured dataset."""
        # Create training dataloader with shuffling for stochastic training
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,  # Use configured batch size
            shuffle=True,  # Shuffle training data each epoch
            num_workers=self.num_workers,  # Use configured number of workers
            pin_memory=self.pin_memory,  # Pin memory for faster GPU transfer
        )

    def val_dataloader(self):
        """Return the validation dataloader for the configured dataset."""
        # Create validation dataloader without shuffling for consistent evaluation
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,  # Use same batch size as training
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.num_workers,  # Use same number of workers
            pin_memory=self.pin_memory,  # Pin memory for faster GPU transfer
        )
