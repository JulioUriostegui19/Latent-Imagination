"""Neural building blocks (MLP/Conv) used by the VAE models."""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _prod(shape: Sequence[int]) -> int:
    """Utility to turn CxHxW shapes into flattened dimensions."""
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
        self.input_dim = input_dim or _prod(input_shape)
        self.fc1 = nn.Linear(self.input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_mu = nn.Linear(h2, z_dim)
        self.fc_logvar = nn.Linear(h2, z_dim)
        self.act = torch.tanh

    def forward(self, x: torch.Tensor):
        """Compute mean/log-variance of q(z|x)."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
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
        self.output_dim = output_dim or _prod(output_shape)
        self.fc1 = nn.Linear(z_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_out = nn.Linear(h2, self.output_dim)
        self.act = torch.tanh

    def forward(self, z: torch.Tensor):
        """Return logits matching the original data shape."""
        h = self.act(self.fc1(z))
        h = self.act(self.fc2(h))
        x_logit = self.fc_out(h)
        return x_logit.view(z.size(0), *self.output_shape)


class ConvEncoder(nn.Module):
    """Down-sampling convolutional encoder that produces Gaussian parameters."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dims: Sequence[int] = (32, 64),
        z_dim: int = 32,
    ):
        super().__init__()
        in_channels = input_shape[0]
        modules = []
        channels = in_channels
        for h_dim in hidden_dims:
            modules.extend(
                [nn.Conv2d(channels, h_dim, kernel_size=4, stride=2, padding=1), nn.ReLU()]
            )
            channels = h_dim
        modules.append(nn.Flatten())
        self.conv = nn.Sequential(*modules)
        self.hidden_dims = tuple(hidden_dims)
        self.input_shape = input_shape
        final_h = input_shape[1] // (2 ** len(hidden_dims))
        final_w = input_shape[2] // (2 ** len(hidden_dims))
        if final_h == 0 or final_w == 0:
            raise ValueError(
                f"Input spatial dims {input_shape[1:]} too small for {len(hidden_dims)} conv layers"
            )
        self._feature_shape = (channels, final_h, final_w)
        self._flatten_dim = channels * final_h * final_w
        self.fc_mu = nn.Linear(self._flatten_dim, z_dim)
        self.fc_logvar = nn.Linear(self._flatten_dim, z_dim)

    def forward(self, x: torch.Tensor):
        """Return mean and log-variance after conv feature extraction."""
        h = self.conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @property
    def feature_shape(self) -> Tuple[int, int, int]:
        return self._feature_shape


class ConvDecoder(nn.Module):
    """Transpose-convolutional decoder that mirrors the encoder architecture."""

    def __init__(
        self,
        output_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dims: Sequence[int] = (64, 32),
        z_dim: int = 32,
    ):
        super().__init__()
        c, h, w = output_shape
        num_layers = len(hidden_dims)
        if h % (2**num_layers) != 0 or w % (2**num_layers) != 0:
            raise ValueError(
                f"Output spatial dims {(h, w)} incompatible with number of conv layers {num_layers}"
            )
        init_h = h // (2**num_layers)
        init_w = w // (2**num_layers)
        init_channels = hidden_dims[0]
        self.output_shape = output_shape
        self.hidden_dims = tuple(hidden_dims)
        self.fc = nn.Linear(z_dim, init_channels * init_h * init_w)
        modules = [
            nn.Unflatten(1, (init_channels, init_h, init_w)),
        ]
        channels = init_channels
        for h_dim in hidden_dims[1:]:
            modules.append(nn.ConvTranspose2d(channels, h_dim, kernel_size=4, stride=2, padding=1))
            modules.append(nn.ReLU())
            channels = h_dim
        modules.append(nn.ConvTranspose2d(channels, c, kernel_size=4, stride=2, padding=1))
        self.net = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor):
        """Generate reconstruction logits from latent samples."""
        h = self.fc(z)
        return self.net(h)
