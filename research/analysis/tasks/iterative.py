"""Iterative inference test and plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from research.analysis.common import (
    ModelSpec,
    recon_mse,
    iterative_recon_mse,
)


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
    import numpy as np

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

    def _to_img(arr: torch.Tensor):
        x = torch.clamp((arr.detach().cpu() + 1.0) / 2.0, 0.0, 1.0)
        if x.ndim == 3:
            if x.shape[0] == 1:
                return x.squeeze(0).numpy(), "gray"
            if x.shape[0] == 3:
                return x.permute(1, 2, 0).numpy(), None
        if x.ndim == 2:
            return x.numpy(), "gray"
        return x.squeeze().numpy(), "gray"

    for i in range(num_samples):
        for col, step in enumerate(step_indices):
            z = z_traj[step, i].unsqueeze(0).to(device)
            with torch.no_grad():
                recon = decoder(z).cpu()
            img, cmap = _to_img(recon[0])
            axes[i, col].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[i, col].axis("off")
            if i == 0:
                axes[i, col].set_title(f"iter {step}")
        if x_true is not None:
            img, cmap = _to_img(x_true[i])
            axes[i, -1].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[i, -1].axis("off")
            if i == 0:
                axes[i, -1].set_title("target")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run_iterative_inference_test(
    models: Sequence[ModelSpec],
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: Mapping[str, float],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Dict[str, Sequence[float]]]:
    """Evaluate iterative inference curves for a batch of validation samples."""
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
            x_recon, _, _ = module(x)
        _, baseline_mse = recon_mse(x_recon, x)

        lr_eval = lr_factor * (model.lr_inf if model.lr_inf is not None else model.lr)

        # Adapter: support external models exposing encode/decode instead of separate modules
        if hasattr(module, "encode") and hasattr(module, "decode"):
            class _EncAdapter(torch.nn.Module):
                def __init__(self, m: torch.nn.Module):
                    super().__init__()
                    self.m = m

                def forward(self, t: torch.Tensor):
                    out = self.m.encode(t)
                    # Accept list or tuple and normalize to tuple(mu, logvar)
                    if isinstance(out, (list, tuple)):
                        return out[0], out[1]
                    return out

            class _DecAdapter(torch.nn.Module):
                def __init__(self, m: torch.nn.Module):
                    super().__init__()
                    self.m = m

                def forward(self, z: torch.Tensor):
                    return self.m.decode(z)

            enc_mod = _EncAdapter(module)
            dec_mod = _DecAdapter(module)
        else:
            enc_mod = module.encoder
            dec_mod = module.decoder

        curve, recons, latent = iterative_recon_mse(
            enc_mod,
            dec_mod,
            x,
            n_steps=n_steps,
            lr_eval=lr_eval,
            beta=model.beta,
            device=device,
            save_latent_traj=save_latent,
        )

        curves[model.name] = curve
        baselines[model.name] = baseline_mse
        metrics[model.name] = {
            "mse_curve": list(map(float, curve)),
            "baseline_mse": float(baseline_mse),
        }

        model_dir = output_dir / model.name
        model_dir.mkdir(parents=True, exist_ok=True)
        if latent is not None:
            plot_latent_evolution(
                decoder=module.decoder,
                z_traj=latent,
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
