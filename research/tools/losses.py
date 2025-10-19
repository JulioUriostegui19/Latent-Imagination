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


def reconstruction_bce_logits(x: torch.Tensor, x_logit: torch.Tensor):
    """Compute reconstruction loss using binary cross-entropy with logits."""
    bce = F.binary_cross_entropy_with_logits(x_logit, x, reduction="none")
    return torch.sum(bce.view(x.shape[0], -1), dim=1)


def elbo_per_sample(
    x: torch.Tensor,
    decoder: torch.nn.Module,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    sample_z: bool = True,
):
    """Compute Evidence Lower BOund (ELBO) loss per sample for VAE training."""
    std = torch.exp(0.5 * logvar)
    if sample_z:
        eps = torch.randn_like(std)
        z = mu + eps * std
    else:
        z = mu
    x_logit = decoder(z)
    rec = reconstruction_bce_logits(x, x_logit)
    kl = gaussian_kl(mu, logvar)
    loss_per_sample = rec + beta * kl
    return loss_per_sample, x_logit

