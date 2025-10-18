"""Iterative/semi-amortized VAE variants."""
# Package for iterative and semi-amortized VAE implementations
# Contains models that refine amortized inference with iterative optimization

# Import iterative VAE implementation
from .iterative import IterativeVAE

# Define public API for this subpackage
__all__ = ["IterativeVAE"]  # Semi-amortized VAE with SVI refinement
