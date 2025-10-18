"""Convenience exports for model components used throughout the training scripts."""
# Package-level imports for easy access to all model components
# This allows importing all models with: from models import BaseVAE, IterativeVAE, etc.

# Import encoder/decoder architectures from VAE module
from .vae.architectures import MLPEncoder, MLPDecoder, ConvEncoder, ConvDecoder
# Import standard VAE implementation
from .vae.base import BaseVAE
# Import iterative/semi-amortized VAE implementation
from .ivae.iterative import IterativeVAE

# Define public API - what gets imported with "from models import *"
__all__ = [
    "MLPEncoder",  # Multi-layer perceptron encoder
    "MLPDecoder",  # Multi-layer perceptron decoder
    "ConvEncoder",  # Convolutional encoder
    "ConvDecoder",  # Convolutional decoder
    "BaseVAE",  # Standard VAE implementation
    "IterativeVAE",  # Iterative/semi-amortized VAE implementation
]
