"""Base amortized VAE LightningModule with standard ELBO training."""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.losses import elbo_per_sample, gaussian_kl, reconstruction_bce_logits


class BaseVAE(pl.LightningModule):
    """Implements the canonical VAE objective (ELBO) inside Lightning."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_shape=(1, 28, 28),
        z_dim: int = 15,
        lr: float = 1e-3,
        beta: float = 1.0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor):
        """Sample latent z via reparameterisation and decode to logits."""
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_logit = self.decoder(z)
        return x_logit, mu, logvar

    def training_step(self, batch, batch_idx):
        """Optimise ELBO (reconstruction + KL) on a mini-batch."""
        x, _ = batch
        loss_per_sample, x_logit = elbo_per_sample(
            x, self.decoder, *self.encoder(x), beta=self.beta
        )
        loss = loss_per_sample.mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        kl = gaussian_kl(*self.encoder(x)).mean()
        rec = reconstruction_bce_logits(x, x_logit).mean()
        self.log("train/kl", kl, on_epoch=True)
        self.log("train/rec", rec, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Report validation ELBO without latent sampling noise."""
        x, _ = batch
        loss_per_sample, x_logit = elbo_per_sample(
            x, self.decoder, *self.encoder(x), beta=self.beta, sample_z=False
        )
        loss = loss_per_sample.mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Use Adam on all parameters by default."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
