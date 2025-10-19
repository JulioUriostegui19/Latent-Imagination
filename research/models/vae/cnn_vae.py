"""Convolutional encoder/decoder blocks for VAE models."""

from typing import Any, Dict, Optional, Sequence, Tuple  # Type hints for function signatures

import torch  # PyTorch for tensor operations
import torch.nn as nn  # Neural network modules


class ConvEncoder(nn.Module):
    """Down-sampling convolutional encoder that produces Gaussian parameters."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        hidden_dims: Sequence[int] = (32, 64),
        z_dim: int = 32,
    ):
        super().__init__()
        in_channels = input_shape[0]  # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        modules = []
        channels = in_channels
        for h_dim in hidden_dims:  # Build convolutional layers with stride=2 for downsampling
            modules.extend(
                [nn.Conv2d(channels, h_dim, kernel_size=4, stride=2, padding=1), nn.ReLU()]
            )
            channels = h_dim
        modules.append(nn.Flatten())  # Flatten spatial dimensions for fully connected layers
        self.conv = nn.Sequential(*modules)
        self.hidden_dims = tuple(hidden_dims)
        self.input_shape = input_shape
        final_h = input_shape[1] // (2 ** len(hidden_dims))  # Calculate final height after pooling
        final_w = input_shape[2] // (2 ** len(hidden_dims))  # Calculate final width after pooling
        if final_h == 0 or final_w == 0:
            raise ValueError(
                f"Input spatial dims {input_shape[1:]} too small for {len(hidden_dims)} conv layers"
            )
        self._feature_shape = (channels, final_h, final_w)
        self._flatten_dim = channels * final_h * final_w
        self.fc_mu = nn.Linear(self._flatten_dim, z_dim)  # Mean of latent distribution
        self.fc_logvar = nn.Linear(self._flatten_dim, z_dim)  # Log variance of latent distribution

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @property
    def feature_shape(self) -> Tuple[int, int, int]:
        """Return the shape of the feature map after convolutional layers."""
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
        c, h, w = output_shape  # Extract channels, height, width
        num_layers = len(hidden_dims)
        if h % (2**num_layers) != 0 or w % (2**num_layers) != 0:
            raise ValueError(
                f"Output spatial dims {(h, w)} incompatible with number of conv layers {num_layers}"
            )
        init_h = h // (2**num_layers)  # Initial height after upsampling
        init_w = w // (2**num_layers)  # Initial width after upsampling
        init_channels = hidden_dims[0]
        self.output_shape = output_shape
        self.hidden_dims = tuple(hidden_dims)
        self.fc = nn.Linear(z_dim, init_channels * init_h * init_w)  # Project latent to spatial dimensions
        modules = [
            nn.Unflatten(1, (init_channels, init_h, init_w)),  # Reshape to spatial tensor
        ]
        channels = init_channels
        for h_dim in hidden_dims[1:]:  # Build transposed convolutional layers for upsampling
            modules.append(nn.ConvTranspose2d(channels, h_dim, kernel_size=4, stride=2, padding=1))
            modules.append(nn.ReLU())
            channels = h_dim
        modules.append(nn.ConvTranspose2d(channels, c, kernel_size=4, stride=2, padding=1))  # Final output layer
        self.net = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor):
        h = self.fc(z)
        return self.net(h)


    
