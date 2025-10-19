"""MLP-based encoder/decoder blocks for VAE models."""

import math  # Mathematical operations for dimension calculations
from typing import Optional, Sequence, Tuple  # Type hints for function signatures

import torch  # PyTorch for tensor operations
import torch.nn as nn  # Neural network modules


def _prod(shape: Sequence[int]) -> int:
    """Compute the flattened size for a CxHxW tensor."""
    return int(math.prod(shape))


class MLPEncoder(nn.Module):
    """Two-layer MLP that outputs Gaussian posterior parameters."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        input_dim: Optional[int] = None,
        h1: int = 512,
        h2: int = 256,
        z_dim: int = 15,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim or _prod(input_shape)  # Calculate flattened input size
        self.fc1 = nn.Linear(self.input_dim, h1)  # First hidden layer
        self.fc2 = nn.Linear(h1, h2)  # Second hidden layer
        self.fc_mu = nn.Linear(h2, z_dim)  # Mean of latent distribution
        self.fc_logvar = nn.Linear(h2, z_dim)  # Log variance of latent distribution
        self.act = torch.tanh  # Activation function

    def forward(self, x: torch.Tensor):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten input if it's not already
        h = self.act(self.fc1(x))  # First hidden layer with activation
        h = self.act(self.fc2(h))  # Second hidden layer with activation
        mu = self.fc_mu(h)  # Mean of latent distribution
        logvar = self.fc_logvar(h)  # Log variance of latent distribution
        return mu, logvar


class MLPDecoder(nn.Module):
    """Two-layer MLP that maps latent samples back to the data space."""

    def __init__(
        self,
        output_shape: Tuple[int, int, int] = (1, 28, 28),
        output_dim: Optional[int] = None,
        h1: int = 256,
        h2: int = 512,
        z_dim: int = 15,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.output_dim = output_dim or _prod(output_shape)  # Calculate flattened output size
        self.fc1 = nn.Linear(z_dim, h1)  # First hidden layer
        self.fc2 = nn.Linear(h1, h2)  # Second hidden layer
        self.fc_out = nn.Linear(h2, self.output_dim)  # Output layer
        self.act = torch.tanh  # Activation function

    def forward(self, z: torch.Tensor):
        h = self.act(self.fc1(z))  # First hidden layer with activation
        h = self.act(self.fc2(h))  # Second hidden layer with activation
        x_logit = self.fc_out(h)  # Output logits
        return x_logit.view(z.size(0), *self.output_shape)  # Reshape to original image dimensions
