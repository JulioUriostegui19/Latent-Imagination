"""Neural building blocks (MLP/Conv) used by the VAE models."""
# Mathematical operations for dimension calculations
import math
# Type hints for better code documentation and IDE support
from typing import Optional, Sequence, Tuple

# PyTorch core library for tensor operations and neural network layers
import torch
import torch.nn as nn


def _prod(shape: Sequence[int]) -> int:
    """Utility to turn CxHxW shapes into flattened dimensions."""
    # Calculate product of all dimensions in the shape tuple (e.g., 1*28*28 = 784)
    return int(math.prod(shape))


class MLPEncoder(nn.Module):
    """Two-layer MLP that outputs Gaussian posterior parameters."""
    # Multi-layer perceptron encoder that maps input images to latent distribution parameters

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),  # Expected input image shape (channels, height, width)
        input_dim: Optional[int] = None,  # Flattened input dimension, calculated if not provided
        h1: int = 512,  # Size of first hidden layer
        h2: int = 256,  # Size of second hidden layer
        z_dim: int = 15,  # Dimensionality of latent space (output size for mu and logvar)
    ):
        super().__init__()
        self.input_shape = input_shape  # Store expected input shape for reference
        self.input_dim = input_dim or _prod(input_shape)  # Calculate flattened dimension if not provided
        self.fc1 = nn.Linear(self.input_dim, h1)  # First linear layer: input_dim -> h1
        self.fc2 = nn.Linear(h1, h2)  # Second linear layer: h1 -> h2
        self.fc_mu = nn.Linear(h2, z_dim)  # Output layer for mean of latent distribution
        self.fc_logvar = nn.Linear(h2, z_dim)  # Output layer for log variance of latent distribution
        self.act = torch.tanh  # Activation function (tanh) for hidden layers

    def forward(self, x: torch.Tensor):
        """Compute mean/log-variance of q(z|x)."""
        # Flatten input tensor if it has spatial dimensions (batch_size, channels, height, width) -> (batch_size, -1)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        # Pass through first hidden layer with activation
        h = self.act(self.fc1(x))
        # Pass through second hidden layer with activation
        h = self.act(self.fc2(h))
        # Generate mean parameter for latent distribution
        mu = self.fc_mu(h)
        # Generate log variance parameter for latent distribution
        logvar = self.fc_logvar(h)
        # Return latent distribution parameters (mu and log variance)
        return mu, logvar


class MLPDecoder(nn.Module):
    """Two-layer MLP that maps latent samples back to the data space."""
    # Multi-layer perceptron decoder that maps latent codes back to reconstructed images

    def __init__(
        self,
        output_shape: Tuple[int, int, int] = (1, 28, 28),  # Expected output image shape (channels, height, width)
        output_dim: Optional[int] = None,  # Flattened output dimension, calculated if not provided
        h1: int = 256,  # Size of first hidden layer (note: typically smaller than encoder)
        h2: int = 512,  # Size of second hidden layer
        z_dim: int = 15,  # Dimensionality of latent space (input size)
    ):
        super().__init__()
        self.output_shape = output_shape  # Store expected output shape for reference
        self.output_dim = output_dim or _prod(output_shape)  # Calculate flattened dimension if not provided
        self.fc1 = nn.Linear(z_dim, h1)  # First linear layer: z_dim -> h1
        self.fc2 = nn.Linear(h1, h2)  # Second linear layer: h1 -> h2
        self.fc_out = nn.Linear(h2, self.output_dim)  # Output layer: h2 -> flattened output
        self.act = torch.tanh  # Activation function (tanh) for hidden layers

    def forward(self, z: torch.Tensor):
        """Return logits matching the original data shape."""
        # Pass latent vector through first hidden layer with activation
        h = self.act(self.fc1(z))
        # Pass through second hidden layer with activation
        h = self.act(self.fc2(h))
        # Generate flattened output logits
        x_logit = self.fc_out(h)
        # Reshape flattened logits back to original image dimensions
        return x_logit.view(z.size(0), *self.output_shape)


class ConvEncoder(nn.Module):
    """Down-sampling convolutional encoder that produces Gaussian parameters."""
    # Convolutional encoder that progressively downsamples input through strided convolutions

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),  # Expected input image shape (channels, height, width)
        hidden_dims: Sequence[int] = (32, 64),  # Number of channels in each conv layer (increasing)
        z_dim: int = 32,  # Dimensionality of latent space (output size for mu and logvar)
    ):
        super().__init__()
        in_channels = input_shape[0]  # Extract number of input channels from shape
        modules = []  # List to collect all layers
        channels = in_channels  # Track current number of channels (starts with input channels)
        # Build convolutional layers with stride 2 for downsampling
        for h_dim in hidden_dims:
            modules.extend(
                [nn.Conv2d(channels, h_dim, kernel_size=4, stride=2, padding=1), nn.ReLU()]
            )
            channels = h_dim  # Update current channels for next iteration
        modules.append(nn.Flatten())  # Flatten spatial dimensions for fully connected layers
        self.conv = nn.Sequential(*modules)  # Create sequential container for all conv layers
        self.hidden_dims = tuple(hidden_dims)  # Store hidden dimensions for reference
        self.input_shape = input_shape  # Store input shape for reference
        # Calculate final spatial dimensions after all downsampling (divide by 2 for each layer)
        final_h = input_shape[1] // (2 ** len(hidden_dims))
        final_w = input_shape[2] // (2 ** len(hidden_dims))
        # Validate that input dimensions are large enough for the downsampling
        if final_h == 0 or final_w == 0:
            raise ValueError(
                f"Input spatial dims {input_shape[1:]} too small for {len(hidden_dims)} conv layers"
            )
        self._feature_shape = (channels, final_h, final_w)  # Store shape after conv layers
        self._flatten_dim = channels * final_h * final_w  # Calculate flattened dimension
        self.fc_mu = nn.Linear(self._flatten_dim, z_dim)  # FC layer for mean of latent distribution
        self.fc_logvar = nn.Linear(self._flatten_dim, z_dim)  # FC layer for log variance of latent distribution

    def forward(self, x: torch.Tensor):
        """Return mean and log-variance after conv feature extraction."""
        # Extract features using convolutional layers (downsampling + feature extraction)
        h = self.conv(x)
        # Generate mean parameter for latent distribution from conv features
        # Generate log variance parameter for latent distribution from conv features
        return self.fc_mu(h), self.fc_logvar(h)

    @property
    def feature_shape(self) -> Tuple[int, int, int]:
        """Return the shape of the feature map after convolutional layers."""
        # Property to access the shape of conv features (channels, height, width)
        return self._feature_shape


class ConvDecoder(nn.Module):
    """Transpose-convolutional decoder that mirrors the encoder architecture."""
    # Convolutional decoder that progressively upsamples latent codes back to images using transpose convolutions

    def __init__(
        self,
        output_shape: Tuple[int, int, int] = (1, 28, 28),  # Expected output image shape (channels, height, width)
        hidden_dims: Sequence[int] = (64, 32),  # Number of channels in each transpose conv layer (decreasing)
        z_dim: int = 32,  # Dimensionality of latent space (input size)
    ):
        super().__init__()
        c, h, w = output_shape  # Unpack output shape into channels, height, width
        num_layers = len(hidden_dims)  # Number of transpose convolution layers
        # Validate that output dimensions are compatible with upsampling (must be divisible by 2^num_layers)
        if h % (2**num_layers) != 0 or w % (2**num_layers) != 0:
            raise ValueError(
                f"Output spatial dims {(h, w)} incompatible with number of conv layers {num_layers}"
            )
        # Calculate initial spatial dimensions before upsampling (divide by 2^num_layers)
        init_h = h // (2**num_layers)
        init_w = w // (2**num_layers)
        init_channels = hidden_dims[0]  # Initial number of channels (first in sequence)
        self.output_shape = output_shape  # Store expected output shape for reference
        self.hidden_dims = tuple(hidden_dims)  # Store hidden dimensions for reference
        # Fully connected layer to project latent code to initial feature map
        self.fc = nn.Linear(z_dim, init_channels * init_h * init_w)
        modules = [
            nn.Unflatten(1, (init_channels, init_h, init_w)),  # Reshape flattened features to spatial format
        ]
        channels = init_channels  # Track current number of channels
        # Build transpose convolutional layers for upsampling (stride 2 doubles spatial dims)
        for h_dim in hidden_dims[1:]:
            modules.append(nn.ConvTranspose2d(channels, h_dim, kernel_size=4, stride=2, padding=1))
            modules.append(nn.ReLU())  # Activation after each transpose conv
            channels = h_dim  # Update current channels for next iteration
        # Final transpose convolution to generate output image (no activation - logits)
        modules.append(nn.ConvTranspose2d(channels, c, kernel_size=4, stride=2, padding=1))
        self.net = nn.Sequential(*modules)  # Create sequential container for all layers

    def forward(self, z: torch.Tensor):
        """Generate reconstruction logits from latent samples."""
        # Project latent code to initial feature map dimensions using fully connected layer
        h = self.fc(z)
        # Apply transpose convolutions to upsample and generate final image reconstruction logits
        return self.net(h)
