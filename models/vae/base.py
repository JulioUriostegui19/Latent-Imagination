"""Base amortized VAE LightningModule with standard ELBO training."""
# Standard library imports for tensor operations and neural network building blocks
import torch
import torch.nn as nn
import pytorch_lightning as pl
# Import custom loss functions for VAE training from utils module
from utils.losses import elbo_per_sample, gaussian_kl, reconstruction_bce_logits


class BaseVAE(pl.LightningModule):
    """Implements the canonical VAE objective (ELBO) inside Lightning."""
    # BaseVAE inherits from PyTorch LightningModule for automatic training loop management

    def __init__(
        self,
        encoder: nn.Module,  # Neural network that maps input x to latent parameters (mu, logvar)
        decoder: nn.Module,  # Neural network that maps latent z back to reconstructed input
        input_shape=(1, 28, 28),  # Expected shape of input data (channels, height, width)
        z_dim: int = 15,  # Dimensionality of the latent space (bottleneck size)
        lr: float = 1e-3,  # Learning rate for Adam optimizer
        beta: float = 1.0,  # Weight for KL divergence term (beta-VAE parameter)
        weight_decay: float = 0.0,  # L2 regularization weight
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])  # Save hyperparams for checkpointing, exclude encoder/decoder as they're modules
        self.encoder = encoder  # Store reference to encoder network
        self.decoder = decoder  # Store reference to decoder network
        self.input_shape = input_shape  # Store expected input dimensions
        self.z_dim = z_dim  # Store latent space dimensionality
        self.lr = lr  # Store learning rate for optimizer
        self.beta = beta  # Store beta parameter for KL weighting
        self.weight_decay = weight_decay  # Store weight decay parameter

    def forward(self, x: torch.Tensor):
        """Sample latent z via reparameterisation and decode to logits."""
        # Pass input through encoder to get latent distribution parameters
        mu, logvar = self.encoder(x)
        # Convert log variance to standard deviation for reparameterization
        std = torch.exp(0.5 * logvar)
        # Sample random noise with same shape as std for reparameterization trick
        eps = torch.randn_like(std)
        # Apply reparameterization trick: z = mu + sigma * epsilon
        z = mu + eps * std
        # Decode latent sample back to reconstructed input logits
        x_logit = self.decoder(z)
        # Return reconstruction and latent parameters for loss computation
        return x_logit, mu, logvar

    def training_step(self, batch, batch_idx):
        """Optimise ELBO (reconstruction + KL) on a mini-batch."""
        # Extract input data from batch tuple (ignore labels)
        x, _ = batch
        # Compute ELBO loss per sample using the custom loss function
        loss_per_sample, x_logit = elbo_per_sample(
            x, self.decoder, *self.encoder(x), beta=self.beta
        )
        # Average loss across the mini-batch
        loss = loss_per_sample.mean()
        # Log the main training loss to TensorBoard/CSV
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Compute and log KL divergence component for monitoring
        kl = gaussian_kl(*self.encoder(x)).mean()
        # Compute and log reconstruction loss component for monitoring
        rec = reconstruction_bce_logits(x, x_logit).mean()
        # Log individual loss components for analysis
        self.log("train/kl", kl, on_epoch=True)
        self.log("train/rec", rec, on_epoch=True)
        # Return loss for backpropagation
        return loss

    def validation_step(self, batch, batch_idx):
        """Report validation ELBO without latent sampling noise."""
        # Extract input data from batch tuple (ignore labels)
        x, _ = batch
        # Compute validation loss without sampling noise (sample_z=False)
        loss_per_sample, x_logit = elbo_per_sample(
            x, self.decoder, *self.encoder(x), beta=self.beta, sample_z=False
        )
        # Average loss across the validation batch
        loss = loss_per_sample.mean()
        # Log validation loss to TensorBoard/CSV with progress bar visibility
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        # Return loss for validation epoch end aggregation
        return loss

    def configure_optimizers(self):
        """Use Adam on all parameters by default."""
        # Create Adam optimizer with stored learning rate and weight decay
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # Return optimizer - Lightning will handle learning rate scheduling if needed
        return optimizer
