"""Vanilla VAE architectures and Lightning module implementations."""
# Package for standard Variational Autoencoder implementations
# Contains both architectural components (encoders/decoders) and training logic

# Import encoder/decoder architectures
from .architectures import MLPEncoder, MLPDecoder, ConvEncoder, ConvDecoder
# Import base VAE Lightning module
from .base import BaseVAE

# Define public API for this subpackage
__all__ = [
    "MLPEncoder",  # MLP-based encoder
    "MLPDecoder",  # MLP-based decoder
    "ConvEncoder",  # Convolutional encoder
    "ConvDecoder",  # Convolutional decoder
    "BaseVAE",  # Standard VAE Lightning module
]
