# Core PyTorch libraries for tensor operations and neural network functions
import torch
import torch.nn.functional as F


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor, mu_p=None, logvar_p=None):
    """Compute KL divergence between two Gaussian distributions element-wise."""
    if mu_p is None:
        mu_p = torch.zeros_like(mu)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar)
    var = torch.exp(logvar)
    var_p = torch.exp(logvar_p)
    term = logvar_p - logvar + (var + (mu - mu_p) ** 2) / var_p - 1.0
    return 0.5 * torch.sum(term, dim=1)


def reconstruction_mse(x: torch.Tensor, x_recon: torch.Tensor):
    """Compute reconstruction loss as mean squared error per sample.

    Assumes x and x_recon are in the same scale, here [-1, 1].
    """
    mse = F.mse_loss(x_recon, x, reduction="none").view(x.shape[0], -1).mean(dim=1)
    return mse


def elbo_per_sample(
    x: torch.Tensor,
    decoder: torch.nn.Module,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    sample_z: bool = True,
):
    """Compute per-sample objective with MSE recon term and Gaussian KL.

    Decoder is expected to output values in [-1, 1] (tanh activation).
    """
    std = torch.exp(0.5 * logvar)
    if sample_z:
        eps = torch.randn_like(std)
        z = mu + eps * std
    else:
        z = mu
    x_recon = decoder(z)
    rec = reconstruction_mse(x, x_recon)
    kl = gaussian_kl(mu, logvar)
    loss_per_sample = rec + beta * kl
    return loss_per_sample, x_recon
