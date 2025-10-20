"""Research-side tools (losses, metrics, helpers).

Public API exposes common loss/ELBO utilities for convenience.
"""

from .losses import (
    elbo_per_sample,
    gaussian_kl,
    reconstruction_bce_logits,
)

__all__ = [
    "elbo_per_sample",
    "gaussian_kl",
    "reconstruction_bce_logits",
]
