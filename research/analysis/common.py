"""Common analytics utilities shared across test tasks.

Includes:
- ModelSpec container used across tests
- Reconstruction metrics and iterative refinement helper
- Input corruption helpers used by OOD tests
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


@dataclass
class ModelSpec:
    """Container describing a loaded checkpoint for evaluation."""

    name: str
    module: torch.nn.Module
    beta: float
    lr: float
    lr_inf: Optional[float]


def recon_mse(recon: torch.Tensor, target: torch.Tensor) -> Tuple[np.ndarray, float]:
    """Return per-sample and mean MSE between recons and ground truth in [-1,1]."""
    mse_per_sample = (
        F.mse_loss(recon, target, reduction="none").view(recon.size(0), -1).mean(dim=1)
    )
    return mse_per_sample.cpu().numpy(), float(mse_per_sample.mean())


def iterative_recon_mse(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    x: torch.Tensor,
    n_steps: int,
    lr_eval: float,
    beta: float,
    device: torch.device,
    save_latent_traj: bool = False,
) -> Tuple[np.ndarray, List[torch.Tensor], Optional[torch.Tensor]]:
    """Run iterative refinement on latent parameters and track reconstruction error."""
    from research.tools.losses import elbo_per_sample

    encoder.eval()
    decoder.eval()

    x = x.to(device)
    with torch.no_grad():
        mu0, logvar0 = encoder(x)

    mu = mu0.clone().detach().requires_grad_(True)
    logvar = logvar0.clone().detach().requires_grad_(True)

    mses: List[float] = []
    recons: List[torch.Tensor] = []
    z_traj: List[torch.Tensor] = []

    def record_state():
        with torch.no_grad():
            x_recon = decoder(mu)
        recons.append(x_recon.detach().cpu())
        _, mean_mse = recon_mse(x_recon, x)
        mses.append(mean_mse)
        if save_latent_traj:
            z_traj.append(mu.detach().cpu())

    record_state()

    for _ in range(n_steps):
        loss_per_sample, _ = elbo_per_sample(x, decoder, mu, logvar, beta=beta)
        loss = loss_per_sample.mean()
        grads = torch.autograd.grad(loss, [mu, logvar], retain_graph=False)
        with torch.no_grad():
            mu -= lr_eval * grads[0]
            logvar -= lr_eval * grads[1]
            mu.requires_grad_()
            logvar.requires_grad_()
        record_state()

    mse_curve = np.asarray(mses)
    latent_traj = torch.stack(z_traj) if save_latent_traj else None
    return mse_curve, recons, latent_traj


def add_white_noise(x: torch.Tensor, sigma: float = 0.6) -> torch.Tensor:
    """Add i.i.d. Gaussian noise to a tensor in [-1, 1] and clamp to range."""
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, -1.0, 1.0)


def add_gaussian_blur(
    x: torch.Tensor, kernel_size: int = 5, sigma: float = 2.0
) -> torch.Tensor:
    """Apply Gaussian blur. Works directly on [-1,1] values."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return TF.gaussian_blur(x, kernel_size=(kernel_size, kernel_size), sigma=sigma)


def add_salt_pepper_noise(x: torch.Tensor, prob: float = 0.4) -> torch.Tensor:
    """Randomly set pixels to -1 or +1 with probability prob/2 each."""
    rnd = torch.rand_like(x)
    x = x.clone()
    x[rnd < (prob / 2)] = -1.0
    x[rnd > 1 - (prob / 2)] = 1.0
    return x


def add_cutout(x: torch.Tensor, size: int = 8, mode: str = "center") -> torch.Tensor:
    """Mask out a square patch from the image.

    Args:
        x: Tensor (B,C,H,W)
        size: side length of the square cutout
        mode: 'center' for a centered square, 'random' for a random location
    """
    b, c, h, w = x.shape
    size = max(1, min(size, min(h, w)))
    y = x.clone()
    if mode == "center":
        cy, cx = h // 2, w // 2
        y0 = max(0, cy - size // 2)
        x0 = max(0, cx - size // 2)
    else:
        y0 = int(torch.randint(0, max(1, h - size + 1), (1,)).item())
        x0 = int(torch.randint(0, max(1, w - size + 1), (1,)).item())
    y[:, :, y0 : y0 + size, x0 : x0 + size] = -1.0
    return y

