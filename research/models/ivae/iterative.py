"""Iterative/semi-amortized VAE module with inner-loop SVI refinement."""
# Core PyTorch libraries for tensor operations, neural networks, and Lightning framework
import torch
import torch.nn as nn
import pytorch_lightning as pl
# Import custom loss functions for VAE training from utils module
from research.tools.losses import elbo_per_sample, gaussian_kl


class IterativeVAE(pl.LightningModule):
    """LightningModule that alternates between amortized inference and inner-loop SVI."""
    # Semi-amortized VAE that refines amortized inference with iterative stochastic variational inference
    # This implements the "iterative refinement" approach where we first get a quick estimate from the encoder,
    # then refine it with gradient-based optimization on the latent variables themselves

    def __init__(
        self,
        encoder: nn.Module,  # Neural network for amortized inference (quick initial estimate)
        decoder: nn.Module,  # Neural network for decoding latent codes to reconstructions
        input_shape=(1, 28, 28),  # Expected input data shape (channels, height, width)
        z_dim: int = 15,  # Dimensionality of latent space
        lr_model: float = 1e-4,  # Learning rate for encoder/decoder parameters (model learning)
        lr_inf: float = 1e-2,  # Learning rate for latent variable optimization (inference learning)
        svi_steps: int = 20,  # Number of SVI refinement steps per training iteration
        beta: float = 1.0,  # Weight for KL divergence term in ELBO
        weight_decay: float = 0.0,  # L2 regularization weight
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])  # Save hyperparams for checkpointing
        self.encoder = encoder  # Store reference to encoder network
        self.decoder = decoder  # Store reference to decoder network
        self.input_shape = input_shape  # Store expected input shape
        self.z_dim = z_dim  # Store latent space dimensionality
        self.lr_model = lr_model  # Store learning rate for model parameters
        self.lr_inf = lr_inf  # Store learning rate for inference refinement
        self.svi_steps = svi_steps  # Store number of SVI refinement steps
        self.beta = beta  # Store beta parameter for KL weighting
        self.weight_decay = weight_decay  # Store weight decay parameter
        self.automatic_optimization = False  # Disable automatic optimization - we'll handle it manually

    def configure_optimizers(self):
        """Create independent optimizers for encoder (inference net) and decoder (model)."""
        # Create separate optimizer for encoder parameters (amortized inference network)
        opt_enc = torch.optim.Adam(
            self.encoder.parameters(), lr=self.lr_model, weight_decay=self.weight_decay
        )
        # Create separate optimizer for decoder parameters (generative model)
        opt_dec = torch.optim.Adam(
            self.decoder.parameters(), lr=self.lr_model, weight_decay=self.weight_decay
        )
        # Return both optimizers - Lightning will manage them separately
        return [opt_enc, opt_dec]

    def svi_infer(self, x: torch.Tensor, mu0: torch.Tensor, logvar0: torch.Tensor, steps: int):
        """Run gradient-based SVI updates starting from amortized parameters."""
        # Clone amortized parameters and enable gradients for SVI optimization
        mu = mu0.clone().detach().to(x.device).requires_grad_(True)
        logvar = logvar0.clone().detach().to(x.device).requires_grad_(True)
        traj = []  # List to store optimization trajectory for analysis
        # Perform gradient-based optimization on latent parameters
        for _ in range(steps):
            # Compute ELBO loss using current latent parameters
            loss_per_sample, _ = elbo_per_sample(
                x, self.decoder, mu, logvar, beta=self.beta
            )
            # Debug: show per-sample loss shape and dtype during SVI
            try:
                if self.training:
                    print(
                        f"[SVI] loss_per_sample: shape={tuple(loss_per_sample.shape)} dtype={loss_per_sample.dtype}"
                    )
            except Exception:
                pass
            loss = loss_per_sample.mean()
            try:
                if self.training:
                    print(f"[SVI] loss (mean): shape={tuple(loss.shape)} dtype={loss.dtype}")
            except Exception:
                pass
            # Compute gradients of ELBO with respect to latent parameters
            grads = torch.autograd.grad(loss, [mu, logvar], retain_graph=False)
            with torch.no_grad():
                # Gradient descent step on latent statistics (mu and logvar).
                mu = mu - self.lr_inf * grads[0]
                logvar = logvar - self.lr_inf * grads[1]
                # Re-enable gradients for next iteration
                mu.requires_grad_()
                logvar.requires_grad_()
            # Store current parameters in trajectory for monitoring/analysis
            traj.append((mu.detach().cpu().clone(), logvar.detach().cpu().clone()))
        # Return refined parameters and optimization trajectory
        return mu.detach(), logvar.detach(), traj

    def training_step(self, batch, batch_idx):
        """Perform alternating optimization between decoder and encoder using SVI results."""
        # Extract input data from batch tuple (ignore labels)
        x, _ = batch
        # Get both optimizers (encoder and decoder have separate optimizers)
        opt_enc, opt_dec = self.optimizers()
        # Get initial amortized estimates from encoder
        mu0, logvar0 = self.encoder(x)
        # Refine latent parameters using SVI optimization
        mu_k, logvar_k, _traj = self.svi_infer(x, mu0, logvar0, steps=self.svi_steps)

        # Update decoder parameters using refined latent parameters (detached to avoid encoder gradients)
        opt_dec.zero_grad()  # Clear decoder gradients
        loss_dec_per_sample, _ = elbo_per_sample(
            x, self.decoder, mu_k.detach(), logvar_k.detach(), beta=self.beta
        )
        # Debug: decoder loss shapes/dtypes
        try:
            print(
                f"[Train] loss_dec_per_sample: shape={tuple(loss_dec_per_sample.shape)} dtype={loss_dec_per_sample.dtype}"
            )
        except Exception:
            pass
        loss_dec = loss_dec_per_sample.mean()
        try:
            print(f"[Train] loss_dec (mean): shape={tuple(loss_dec.shape)} dtype={loss_dec.dtype}")
        except Exception:
            pass
        self.manual_backward(loss_dec)  # Compute gradients for decoder
        opt_dec.step()  # Update decoder parameters

        # Update encoder parameters using amortized estimates (standard VAE loss)
        opt_enc.zero_grad()  # Clear encoder gradients
        loss_enc_per_sample, _ = elbo_per_sample(
            x, self.decoder, mu0, logvar0, beta=self.beta
        )
        # Debug: encoder loss shapes/dtypes
        try:
            print(
                f"[Train] loss_enc_per_sample: shape={tuple(loss_enc_per_sample.shape)} dtype={loss_enc_per_sample.dtype}"
            )
        except Exception:
            pass
        loss_enc = loss_enc_per_sample.mean()
        try:
            print(f"[Train] loss_enc (mean): shape={tuple(loss_enc.shape)} dtype={loss_enc.dtype}")
        except Exception:
            pass
        self.manual_backward(loss_enc)  # Compute gradients for encoder
        opt_enc.step()  # Update encoder parameters

        # Log losses for monitoring
        self.log("train/loss_dec", loss_dec, on_epoch=True, prog_bar=True)
        self.log("train/loss_enc", loss_enc, on_epoch=True)
        # Log KL divergence of amortized estimates for analysis
        kl = gaussian_kl(mu0, logvar0).mean()
        self.log("train/kl_amortized", kl, on_epoch=True)
        # Return both losses for potential use by Lightning
        return {"loss_dec": loss_dec, "loss_enc": loss_enc}

    def validation_step(self, batch, batch_idx):
        """Evaluate ELBO after running inner-loop inference."""
        # Extract input data from batch tuple (ignore labels)
        x, _ = batch
        # Get initial amortized estimates from encoder
        mu0, logvar0 = self.encoder(x)
        # Refine latent parameters using SVI optimization (same as training)
        # Lightning disables grad in validation; enable it for SVI refinement.
        with torch.enable_grad():
            mu_k, logvar_k, _ = self.svi_infer(x, mu0, logvar0, steps=self.svi_steps)
        # Compute validation loss using refined parameters (no sampling for stability)
        loss_per_sample, x_logit = elbo_per_sample(
            x, self.decoder, mu_k, logvar_k, beta=self.beta, sample_z=False
        )
        # Debug: validation loss shapes/dtypes
        try:
            print(
                f"[Val] loss_per_sample: shape={tuple(loss_per_sample.shape)} dtype={loss_per_sample.dtype}"
            )
        except Exception:
            pass
        loss = loss_per_sample.mean()
        try:
            print(f"[Val] loss (mean): shape={tuple(loss.shape)} dtype={loss.dtype}")
        except Exception:
            pass
        # Log validation loss to TensorBoard/CSV with progress bar visibility
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        # Return loss for validation epoch end aggregation
        return loss
