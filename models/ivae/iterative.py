"""Iterative/semi-amortized VAE module with inner-loop SVI refinement."""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.losses import elbo_per_sample, gaussian_kl


class IterativeVAE(pl.LightningModule):
    """LightningModule that alternates between amortized inference and inner-loop SVI."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_shape=(1, 28, 28),
        z_dim: int = 15,
        lr_model: float = 1e-4,
        lr_inf: float = 1e-2,
        svi_steps: int = 20,
        beta: float = 1.0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.lr_model = lr_model
        self.lr_inf = lr_inf
        self.svi_steps = svi_steps
        self.beta = beta
        self.weight_decay = weight_decay
        self.automatic_optimization = False

    def configure_optimizers(self):
        """Create independent optimizers for encoder (inference net) and decoder (model)."""
        opt_enc = torch.optim.Adam(
            self.encoder.parameters(), lr=self.lr_model, weight_decay=self.weight_decay
        )
        opt_dec = torch.optim.Adam(
            self.decoder.parameters(), lr=self.lr_model, weight_decay=self.weight_decay
        )
        return [opt_enc, opt_dec]

    def svi_infer(self, x: torch.Tensor, mu0: torch.Tensor, logvar0: torch.Tensor, steps: int):
        """Run gradient-based SVI updates starting from amortized parameters."""
        mu = mu0.clone().detach().to(x.device).requires_grad_(True)
        logvar = logvar0.clone().detach().to(x.device).requires_grad_(True)
        traj = []
        for _ in range(steps):
            loss_per_sample, _ = elbo_per_sample(
                x, self.decoder, mu, logvar, beta=self.beta
            )
            loss = loss_per_sample.mean()
            grads = torch.autograd.grad(loss, [mu, logvar], retain_graph=False)
            with torch.no_grad():
                # Gradient descent step on latent statistics.
                mu = mu - self.lr_inf * grads[0]
                logvar = logvar - self.lr_inf * grads[1]
                mu.requires_grad_()
                logvar.requires_grad_()
            traj.append((mu.detach().cpu().clone(), logvar.detach().cpu().clone()))
        return mu.detach(), logvar.detach(), traj

    def training_step(self, batch, batch_idx):
        """Perform alternating optimization between decoder and encoder using SVI results."""
        x, _ = batch
        opt_enc, opt_dec = self.optimizers()
        mu0, logvar0 = self.encoder(x)
        mu_k, logvar_k, _traj = self.svi_infer(x, mu0, logvar0, steps=self.svi_steps)

        opt_dec.zero_grad()
        loss_dec_per_sample, _ = elbo_per_sample(
            x, self.decoder, mu_k.detach(), logvar_k.detach(), beta=self.beta
        )
        loss_dec = loss_dec_per_sample.mean()
        self.manual_backward(loss_dec)
        opt_dec.step()

        opt_enc.zero_grad()
        loss_enc_per_sample, _ = elbo_per_sample(
            x, self.decoder, mu0, logvar0, beta=self.beta
        )
        loss_enc = loss_enc_per_sample.mean()
        self.manual_backward(loss_enc)
        opt_enc.step()

        self.log("train/loss_dec", loss_dec, on_epoch=True, prog_bar=True)
        self.log("train/loss_enc", loss_enc, on_epoch=True)
        kl = gaussian_kl(mu0, logvar0).mean()
        self.log("train/kl_amortized", kl, on_epoch=True)
        return {"loss_dec": loss_dec, "loss_enc": loss_enc}

    def validation_step(self, batch, batch_idx):
        """Evaluate ELBO after running inner-loop inference."""
        x, _ = batch
        mu0, logvar0 = self.encoder(x)
        mu_k, logvar_k, _ = self.svi_infer(x, mu0, logvar0, steps=self.svi_steps)
        loss_per_sample, x_logit = elbo_per_sample(
            x, self.decoder, mu_k, logvar_k, beta=self.beta, sample_z=False
        )
        loss = loss_per_sample.mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss
