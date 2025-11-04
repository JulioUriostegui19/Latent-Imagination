"""Out-of-distribution corruption robustness test and plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from research.analysis.common import (
    add_cutout,
    add_gaussian_blur,
    add_salt_pepper_noise,
    add_white_noise,
    iterative_recon_mse,
    recon_mse,
    ModelSpec,
)


def plot_reconstruction_timeline(
    recons: Sequence[torch.Tensor],
    x_true: torch.Tensor,
    x_corr: torch.Tensor,
    save_path: Path,
    num_steps_to_show: int,
    title: str,
):
    """Plot reconstructions across a subset of iterations for a single example."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = len(recons)
    indices = np.linspace(0, steps - 1, num_steps_to_show, dtype=int)
    cols = num_steps_to_show + 1 + 1  # corrupted + iterates + target
    fig, axes = plt.subplots(1, cols, figsize=(2.5 * cols, 3))

    axes = np.atleast_1d(axes)
    img_corr = torch.clamp((x_corr.squeeze().cpu() + 1.0) / 2.0, 0.0, 1.0)
    axes[0].imshow(img_corr, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("corrupted")
    axes[0].axis("off")

    for idx, step in enumerate(indices):
        img = torch.clamp((recons[step].squeeze().cpu() + 1.0) / 2.0, 0.0, 1.0)
        axes[idx + 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[idx + 1].set_title(f"iter {step}")
        axes[idx + 1].axis("off")

    img_true = torch.clamp((x_true.squeeze().cpu() + 1.0) / 2.0, 0.0, 1.0)
    axes[-1].imshow(img_true, cmap="gray", vmin=0, vmax=1)
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


def _resolve_corruption_fn(
    name: str, params: Mapping[str, float]
) -> Callable[[torch.Tensor], torch.Tensor]:
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
                recon, _, _ = module(x_corr)
            _, baseline = recon_mse(recon, x_corr)

            lr_eval = lr_factor * (
                model.lr_inf if model.lr_inf is not None else model.lr
            )

            # Adapter: support external models exposing encode/decode instead of separate modules
            if hasattr(module, "encode") and hasattr(module, "decode"):
                class _EncAdapter(torch.nn.Module):
                    def __init__(self, m: torch.nn.Module):
                        super().__init__()
                        self.m = m

                    def forward(self, t: torch.Tensor):
                        out = self.m.encode(t)
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

            curve, recons, _ = iterative_recon_mse(
                enc_mod,
                dec_mod,
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
