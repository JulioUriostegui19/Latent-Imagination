# Core PyTorch libraries for tensor operations and neural network functions
import torch
import torch.nn.functional as F


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor, mu_p=None, logvar_p=None):
    """Compute KL divergence between two Gaussian distributions element-wise."""
    # Set default prior parameters to standard normal (N(0, I)) if not provided
    if mu_p is None:
        mu_p = torch.zeros_like(mu)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar)
    # Convert log variances to variances for KL calculation
    var = torch.exp(logvar)
    var_p = torch.exp(logvar_p)
    # Compute KL divergence term by term: KL(q||p) = E_q[log q(z) - log p(z)]
    # For Gaussians: 0.5 * [log(σp²/σq²) + (σq² + (μq-μp)²)/σp² - 1]
    term = logvar_p - logvar + (var + (mu - mu_p) ** 2) / var_p - 1.0
    # Sum over latent dimensions and apply 0.5 factor, return per-sample KL values
    return 0.5 * torch.sum(term, dim=1)


def reconstruction_bce_logits(x: torch.Tensor, x_logit: torch.Tensor):
    """Compute reconstruction loss using binary cross-entropy with logits."""
    # Apply sigmoid activation and compute binary cross-entropy loss
    # Uses logits for numerical stability (avoids explicit sigmoid)
    bce = F.binary_cross_entropy_with_logits(x_logit, x, reduction="none")
    # Sum over all pixels/channels to get per-sample reconstruction loss
    return torch.sum(bce.view(x.shape[0], -1), dim=1)


def elbo_per_sample(
    x: torch.Tensor,  # Input data (batch of images)
    decoder: torch.nn.Module,  # Decoder network that maps latent codes to reconstructions
    mu: torch.Tensor,  # Mean of latent distribution q(z|x)
    logvar: torch.Tensor,  # Log variance of latent distribution q(z|x)
    beta: float = 1.0,  # Weight for KL divergence term (beta-VAE parameter)
    sample_z: bool = True,  # Whether to sample z or use mean directly
):
    """Compute Evidence Lower BOund (ELBO) loss per sample for VAE training."""
    # Convert log variance to standard deviation for reparameterization
    std = torch.exp(0.5 * logvar)
    if sample_z:
        # Sample latent codes using reparameterization trick (adds stochasticity)
        eps = torch.randn_like(std)
        z = mu + eps * std
    else:
        # Use mean directly (no sampling - deterministic, used for validation)
        z = mu
    # Decode latent codes to reconstructed images (logits)
    x_logit = decoder(z)
    # Compute reconstruction loss (negative log-likelihood)
    rec = reconstruction_bce_logits(x, x_logit)
    # Compute KL divergence between q(z|x) and p(z) (regularization term)
    kl = gaussian_kl(mu, logvar)
    # Combine reconstruction and KL terms with beta weighting (beta-VAE)
    loss_per_sample = rec + beta * kl
    # Return per-sample losses and reconstruction logits for monitoring
    return loss_per_sample, x_logit
