# Package-level imports for utility functions and classes
# This allows importing utilities with: from utils import elbo_per_sample, ensure_dir, etc.

# Import loss functions used in VAE training
from .losses import gaussian_kl, reconstruction_bce_logits, elbo_per_sample
# Import filesystem utilities
from .filesystem import ensure_dir
# Import data loading utilities
from .dataloaders import GenericImageDataModule

# Define public API - what gets imported with "from utils import *"
__all__ = [
    "gaussian_kl",  # KL divergence between Gaussian distributions
    "reconstruction_bce_logits",  # Reconstruction loss with BCE
    "elbo_per_sample",  # Evidence Lower Bound per sample
    "ensure_dir",  # Directory creation utility
    "GenericImageDataModule",  # Generic data module for MNIST/SUN
]
