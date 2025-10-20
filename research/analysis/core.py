"""Analytics helpers for evaluating trained VAE checkpoints.

Also provides a simple registry so new tasks can be plugged in easily.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from research.models import BaseVAE, IterativeVAE, ConvDecoder, ConvEncoder, MLPDecoder, MLPEncoder
from research.tools.losses import elbo_per_sample


def recon_mse(recon: torch.Tensor, target: torch.Tensor) -> Tuple[np.ndarray, float]:
    """Return per-sample and mean MSE between reconstructions and ground truth."""
    # Compute pixel-wise MSE without reduction so we can aggregate by sample.
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
        """Record current reconstruction + metric (and optionally latent state)."""
        with torch.no_grad():
            x_logit = decoder(mu)
            x_recon = torch.sigmoid(x_logit)
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
            # Gradient descent on variational parameters (semi-amortized refinement).
            mu -= lr_eval * grads[0]
            logvar -= lr_eval * grads[1]
            mu.requires_grad_()
            logvar.requires_grad_()
        record_state()

    mse_curve = np.asarray(mses)
    latent_traj = torch.stack(z_traj) if save_latent_traj else None
    return mse_curve, recons, latent_traj


def plot_iterative_curves(
    curves: Mapping[str, np.ndarray],
    baseline: Mapping[str, float],
    n_steps: int,
    save_path: Path,
):
    """Plot reconstruction MSE curves for each model."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(0, n_steps + 1)
    for name, curve in curves.items():
        ax.plot(steps, curve, label=f"{name} (iterative)")
        ax.plot(
            steps,
            np.ones_like(curve) * baseline[name],
            linestyle="--",
            label=f"{name} (amortized)",
        )
    ax.set_xlabel("SVI iteration")
    ax.set_ylabel("Reconstruction MSE ↓")
    ax.set_title("Iterative inference reconstruction error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_latent_evolution(
    decoder: torch.nn.Module,
    z_traj: torch.Tensor,
    x_true: Optional[torch.Tensor],
    save_path: Path,
    num_samples: int = 5,
    num_steps_to_show: int = 5,
):
    """Visualise decoded reconstructions along the latent trajectory."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps, batch, _ = z_traj.shape
    num_samples = min(num_samples, batch)
    step_indices = np.linspace(0, steps - 1, num_steps_to_show, dtype=int)

    decoder.eval()
    device = next(decoder.parameters()).device

    fig, axes = plt.subplots(
        num_samples,
        len(step_indices) + (1 if x_true is not None else 0),
        figsize=(2.4 * len(step_indices), 2.4 * num_samples),
    )
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        for col, step in enumerate(step_indices):
            z = z_traj[step, i].unsqueeze(0).to(device)
            with torch.no_grad():
                recon = torch.sigmoid(decoder(z)).cpu()
            axes[i, col].imshow(recon[0].squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[i, col].axis("off")
            if i == 0:
                axes[i, col].set_title(f"iter {step}")
        if x_true is not None:
            axes[i, -1].imshow(x_true[i].cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[i, -1].axis("off")
            if i == 0:
                axes[i, -1].set_title("target")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def add_white_noise(x: torch.Tensor, sigma: float = 0.6) -> torch.Tensor:
    """Add i.i.d. Gaussian noise to a tensor in [0, 1]."""
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def add_gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 2.0) -> torch.Tensor:
    """Apply Gaussian blur using torchvision utilities."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # ensure odd
    return TF.gaussian_blur(x, kernel_size=(kernel_size, kernel_size), sigma=sigma)


def add_salt_pepper_noise(x: torch.Tensor, prob: float = 0.4) -> torch.Tensor:
    """Randomly set pixels to 0 or 1 with probability prob/2 each."""
    rnd = torch.rand_like(x)
    x = x.clone()
    x[rnd < (prob / 2)] = 0.0
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
    y[:, :, y0 : y0 + size, x0 : x0 + size] = 0.0
    return y


def plot_reconstruction_timeline(
    recons: Sequence[torch.Tensor],
    x_true: Optional[torch.Tensor],
    x_corr: Optional[torch.Tensor],
    save_path: Path,
    num_steps_to_show: int,
    title: str,
):
    """Plot reconstructions across a subset of iterations for a single example."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = len(recons)
    indices = np.linspace(0, steps - 1, num_steps_to_show, dtype=int)
    cols = num_steps_to_show + (2 if x_true is not None else 1)
    fig, axes = plt.subplots(1, cols, figsize=(2.5 * cols, 3))

    axes = np.atleast_1d(axes)
    axes[0].imshow(x_corr.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("corrupted")
    axes[0].axis("off")

    for idx, step in enumerate(indices):
        axes[idx + 1].imshow(recons[step].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
        axes[idx + 1].set_title(f"iter {step}")
        axes[idx + 1].axis("off")

    if x_true is not None:
        axes[-1].imshow(x_true.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
        axes[-1].set_title("target")
        axes[-1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_ood_accuracy_bars(
    corruption_names: Sequence[str],
    model_scores: Mapping[str, Sequence[float]],
    save_path: Path,
):
    """Create grouped bar plot comparing reconstruction error under corruptions."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.arange(len(corruption_names))
    width = 0.8 / max(1, len(model_scores))

    for i, (model_name, scores) in enumerate(model_scores.items()):
        ax.bar(idx + i * width, scores, width=width, label=model_name)

    ax.set_xticks(idx + width * (len(model_scores) - 1) / 2)
    ax.set_xticklabels(corruption_names, rotation=20)
    ax.set_ylabel("Reconstruction MSE ↓")
    ax.set_title("OOD corruption robustness")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


@dataclass
class ModelSpec:
    """Container describing a loaded checkpoint for evaluation."""

    name: str
    module: torch.nn.Module
    beta: float
    lr: float
    lr_inf: Optional[float]


def run_iterative_inference_test(
    models: Sequence[ModelSpec],
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: Mapping[str, float],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Dict[str, Sequence[float]]]:
    """Evaluate iterative inference curves for a batch of validation samples."""
    # Pull sampling/inference parameters out of the test config.
    n_examples = int(cfg.get("n_examples", 128))
    n_steps = int(cfg.get("n_steps", 50))
    lr_factor = float(cfg.get("lr_factor", 0.1))
    save_latent = bool(cfg.get("save_latent_traj", False))
    latent_samples = int(cfg.get("latent_num_samples", 5))
    latent_steps = int(cfg.get("latent_num_steps", 5))

    xs: List[torch.Tensor] = []
    for batch_x, _ in loader:
        xs.append(batch_x)
        if sum(t.size(0) for t in xs) >= n_examples:
            break
    if not xs:
        raise RuntimeError("Validation loader produced no batches.")
    x = torch.cat(xs, dim=0)[:n_examples].to(device)

    curves: Dict[str, np.ndarray] = {}
    baselines: Dict[str, float] = {}
    metrics: Dict[str, Dict[str, Sequence[float]]] = {}

    for model in models:
        module = model.module.to(device)
        module.eval()
        with torch.no_grad():
            x_logit, _, _ = module(x)
            x_recon = torch.sigmoid(x_logit)
        _, baseline_mse = recon_mse(x_recon, x)
        baselines[model.name] = baseline_mse

        lr_eval = lr_factor * (model.lr_inf if model.lr_inf is not None else model.lr)
        curve, recons, latent_traj = iterative_recon_mse(
            module.encoder,
            module.decoder,
            x,
            n_steps=n_steps,
            lr_eval=lr_eval,
            beta=model.beta,
            device=device,
            save_latent_traj=save_latent,
        )
        curves[model.name] = curve
        metrics[model.name] = {
            "baseline_mse": [baseline_mse],
            "iterative_curve": curve.tolist(),
        }

        if save_latent and latent_traj is not None:
            model_dir = output_dir / "models" / model.name
            model_dir.mkdir(parents=True, exist_ok=True)
            plot_latent_evolution(
                decoder=module.decoder,
                z_traj=latent_traj,
                x_true=x[:latent_samples].detach().cpu(),
                save_path=model_dir / "latent_evolution.png",
                num_samples=latent_samples,
                num_steps_to_show=latent_steps,
            )

    plot_iterative_curves(
        curves=curves,
        baseline=baselines,
        n_steps=n_steps,
        save_path=output_dir / "iterative_inference_mse.png",
    )
    return metrics


# Simple registry to make adding tests easy
from typing import Callable as _Callable

TEST_REGISTRY: Dict[str, _Callable[..., Dict[str, object]]] = {
    "iterative": run_iterative_inference_test,
    "ood": run_ood_test,
}


def run_test_by_name(
    name: str,
    *,
    models: Sequence[ModelSpec],
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: Mapping[str, object],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    if name not in TEST_REGISTRY:
        raise ValueError(f"Unknown test '{name}'. Available: {sorted(TEST_REGISTRY)}")
    fn = TEST_REGISTRY[name]
    return fn(models=models, loader=loader, cfg=cfg, device=device, output_dir=output_dir)


def _resolve_corruption_fn(name: str, params: Mapping[str, float]) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "gaussian_blur":
        return lambda x: add_gaussian_blur(
            x,
            kernel_size=int(params.get("kernel_size", 5)),
            sigma=float(params.get("sigma", 2.0)),
        )
    if name == "white_noise":
        return lambda x: add_white_noise(x, sigma=float(params.get("sigma", 0.6)))
    if name == "salt_pepper":
        return lambda x: add_salt_pepper_noise(x, prob=float(params.get("prob", 0.4)))
    if name == "cutout":
        return lambda x: add_cutout(
            x,
            size=int(params.get("size", 8)),
            mode=str(params.get("mode", "center")),
        )
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unknown corruption function '{name}'")


def run_ood_test(
    models: Sequence[ModelSpec],
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: Mapping[str, object],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Evaluate robustness under input corruptions and create comparison plots."""
    # Config structure mirrors the notebook: choose how many samples/steps/corruptions to test.
    n_examples = int(cfg.get("n_examples", 64))
    n_steps = int(cfg.get("n_steps", 30))
    lr_factor = float(cfg.get("lr_factor", 0.1))
    num_timeline_steps = int(cfg.get("timeline_steps", 6))
    corruption_cfg = cfg.get("corruptions", {})
    if not isinstance(corruption_cfg, Mapping):
        raise TypeError("`corruptions` must be a mapping of name -> params.")

    xs: List[torch.Tensor] = []
    for batch_x, _ in loader:
        xs.append(batch_x)
        if sum(t.size(0) for t in xs) >= n_examples:
            break
    x = torch.cat(xs, dim=0)[:n_examples].to(device)

    corruption_names = []
    corruption_fns = []
    for name, params in corruption_cfg.items():
        corruption_names.append(name)
        corruption_fns.append(_resolve_corruption_fn(name, params or {}))

    model_scores: Dict[str, List[float]] = {model.name: [] for model in models}
    metrics: Dict[str, Dict[str, float]] = {model.name: {} for model in models}

    for corr_name, corr_fn in zip(corruption_names, corruption_fns):
        x_corr = corr_fn(x)
        for model in models:
            module = model.module.to(device)
            module.eval()
            with torch.no_grad():
                recon_logits, _, _ = module(x_corr)
                recon = torch.sigmoid(recon_logits)
            _, baseline = recon_mse(recon, x_corr)

            lr_eval = lr_factor * (model.lr_inf if model.lr_inf is not None else model.lr)
            curve, recons, _ = iterative_recon_mse(
                module.encoder,
                module.decoder,
                x_corr,
                n_steps=n_steps,
                lr_eval=lr_eval,
                beta=model.beta,
                device=device,
                save_latent_traj=False,
            )
            final_mse = float(curve[-1])
            model_scores[model.name].append(final_mse)
            metrics[model.name][f"{corr_name}_baseline"] = baseline
            metrics[model.name][f"{corr_name}_iterative"] = final_mse

            # Save reconstruction timeline for the first example.
            recons_single = [rec[0] for rec in recons]
            plot_reconstruction_timeline(
                recons=recons_single,
                x_true=x[0],
                x_corr=x_corr[0],
                save_path=output_dir / f"{model.name}_{corr_name}_timeline.png",
                num_steps_to_show=num_timeline_steps,
                title=f"{model.name} – {corr_name}",
            )

    plot_ood_accuracy_bars(
        corruption_names=corruption_names,
        model_scores=model_scores,
        save_path=output_dir / "ood_reconstruction_mse.png",
    )
    return metrics
