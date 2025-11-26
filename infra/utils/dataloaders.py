"""Lightning datamodule abstraction for MNIST and SUN datasets.

SUN397 is loaded via Hugging Face Datasets using streaming to avoid
downloading the full dataset to disk.
"""
# Mathematical operations for dataset splitting calculations
import math
import os

# PyTorch libraries for data handling and image transformations
import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, IterableDataset, get_worker_info
from torchvision import datasets as TVDatasets
from torchvision.datasets import MNIST

# Hugging Face datasets for streaming SUN397
try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional import, validated at runtime
    load_dataset = None

from typing import Iterator, Optional, Tuple
from PIL import Image
import hashlib
from io import BytesIO


class _HFSun397Iterable(IterableDataset):
    """Torch IterableDataset wrapping a streaming HF SUN397 dataset.

    Splits deterministically into train/val by hashing a stable per-example
    identifier with a seed, without requiring dataset length or local files.
    """

    def __init__(
        self,
        hf_stream,
        transform: T.Compose,
        role: str,
        val_split: float,
        seed: int,
        shuffle_buffer: int = 10_000,
    ) -> None:
        super().__init__()
        self.role = role  # 'train' or 'val'
        self.val_split = float(val_split)
        self.seed = int(seed)
        # Shuffle for better mixing but keep deterministic order via seed
        try:
            self._base_stream = hf_stream.shuffle(seed=self.seed, buffer_size=shuffle_buffer)
        except Exception:
            # Fallback to unshuffled stream if shuffle not available
            self._base_stream = hf_stream
        self.transform = transform

    def _stable_id(self, ex: dict) -> bytes:
        """Derive a stable identifier for deterministic split assignment."""
        # Prefer a path-like identifier if available
        img = ex.get("image")
        candidate: Optional[str] = None
        # 1) PIL image with filename attribute
        if isinstance(img, Image.Image) and getattr(img, "filename", None):
            candidate = img.filename
        # 2) Image dict coming from datasets (path/bytes)
        elif isinstance(img, dict) and ("path" in img or "filename" in img):
            candidate = img.get("path") or img.get("filename")
        # 3) Common alternative keys
        elif "image_path" in ex:
            candidate = ex["image_path"]
        elif "file_name" in ex:
            candidate = ex["file_name"]
        elif "path" in ex:
            candidate = ex["path"]

        if candidate:
            key = f"{self.seed}::{candidate}".encode("utf-8", errors="ignore")
            return hashlib.md5(key).digest()

        # 4) As a last resort, hash image bytes to get a consistent id
        try:
            if isinstance(img, Image.Image):
                buf = BytesIO()
                img.save(buf, format="PNG")
                payload = buf.getvalue()
            elif isinstance(img, (bytes, bytearray)):
                payload = bytes(img)
            elif isinstance(img, dict) and "bytes" in img:
                payload = img["bytes"]
            else:
                payload = repr(sorted(ex.items())).encode("utf-8", errors="ignore")
        except Exception:
            payload = repr(sorted(ex.items())).encode("utf-8", errors="ignore")
        key = f"{self.seed}::".encode("utf-8") + payload
        return hashlib.md5(key).digest()

    def _is_val(self, ex: dict) -> bool:
        digest = self._stable_id(ex)
        # Map first 4 bytes to [0,1)
        bucket = int.from_bytes(digest[:4], "big") / float(2**32)
        return bucket < self.val_split

    def _to_tensor_label(self, ex: dict) -> Tuple[torch.Tensor, int]:
        # Decode image to PIL if needed
        img = ex.get("image")
        if not isinstance(img, Image.Image):
            try:
                if isinstance(img, dict) and ("path" in img or "bytes" in img):
                    if "path" in img:
                        img = Image.open(img["path"]).convert("RGB")
                    else:
                        img = Image.open(BytesIO(img["bytes"]))
                elif isinstance(img, (bytes, bytearray)):
                    img = Image.open(BytesIO(bytes(img)))
                else:
                    # Last resort: try PIL construction directly
                    img = Image.fromarray(img)
            except Exception:
                # If decoding fails, raise to surface data issues early
                raise
        # Apply configured transforms
        x = self.transform(img)
        # Label if available; otherwise 0
        y = ex.get("label", 0)
        try:
            y = int(y)
        except Exception:
            y = 0
        return x, y

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        # Ensure worker-aware sharding to avoid duplicating work across workers
        worker = get_worker_info()
        stream = self._base_stream
        try:
            if worker is not None:
                stream = stream.shard(num_shards=worker.num_workers, index=worker.id)
        except Exception:
            # If sharding is unsupported, fall back to shared stream (may duplicate per worker)
            pass

        want_val = self.role == "val"
        for ex in stream:
            is_val = self._is_val(ex)
            if want_val != is_val:
                continue
            yield self._to_tensor_label(ex)

    # Note: We intentionally do not provide __len__; the HF streaming dataset
    # may not know its size without scanning. Lightning can handle IterableDatasets.


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
        sun_local_dir: Optional[str] = None,  # Optional local SUN397 (subset or full) ImageFolder root
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
        self.sun_local_dir = sun_local_dir  # Optional local SUN folder with train/ and test/ subfolders
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
            # Local SUN datasets do not require download; HF streaming also skips download
            pass

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
            # SUN: prefer local ImageFolder if provided; otherwise use HF streaming
            if self.sun_local_dir:
                # Expected structure:
                #   sun_local_dir/
                #       train/<class>/*.jpg
                #       test/<class>/*.jpg
                train_root = os.path.join(self.sun_local_dir, "train")
                test_root = os.path.join(self.sun_local_dir, "test")
                if not os.path.isdir(train_root) or not os.path.isdir(test_root):
                    raise FileNotFoundError(
                        f"SUN local dir must contain 'train' and 'test' subfolders: {self.sun_local_dir}"
                    )
                self.train_ds = TVDatasets.ImageFolder(train_root, transform=self.transform)
                self.val_ds = TVDatasets.ImageFolder(test_root, transform=self.transform)
            else:
                if load_dataset is None:
                    raise ImportError(
                        "datasets package is required for SUN397 streaming. Add `datasets` to requirements, "
                        "or set `sun_local_dir` to a local ImageFolder root."
                    )

                # Create two independent streaming iterables; partition deterministically into train/val
                try:
                    stream1 = load_dataset("tanganke/sun397", split="train", streaming=True)
                    stream2 = load_dataset("tanganke/sun397", split="train", streaming=True)
                except Exception:
                    # Some dataset variants expose a default split name; try without explicit split
                    stream1 = load_dataset("tanganke/sun397", streaming=True)["train"]
                    stream2 = load_dataset("tanganke/sun397", streaming=True)["train"]

                self.train_ds = _HFSun397Iterable(
                    stream1, transform=self.transform, role="train", val_split=self.val_split, seed=self.split_seed
                )
                self.val_ds = _HFSun397Iterable(
                    stream2, transform=self.transform, role="val", val_split=self.val_split, seed=self.split_seed
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
