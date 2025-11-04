"""Utilities for evaluating trained VAE models and generating analytics plots.

Refactored into modular components:
- Common helpers in `research.analysis.common`
- Task-specific tests under `research.analysis.tests.*`
- Registry/dispatcher in `infra.pipelines.test_engine`
"""

from research.analysis.common import (
    add_gaussian_blur,
    add_salt_pepper_noise,
    add_white_noise,
    iterative_recon_mse,
    ModelSpec,
    recon_mse,
)
from research.analysis.tests.iterative import (
    plot_iterative_curves,
    plot_latent_evolution,
    run_iterative_inference_test,
)
from research.analysis.tests.ood import (
    plot_ood_accuracy_bars,
    plot_reconstruction_timeline,
    run_ood_test,
)
from infra.pipelines.test_engine import run_test_by_name

__all__ = [
    "add_gaussian_blur",
    "add_salt_pepper_noise",
    "add_white_noise",
    "iterative_recon_mse",
    "ModelSpec",
    "plot_iterative_curves",
    "plot_latent_evolution",
    "plot_ood_accuracy_bars",
    "plot_reconstruction_timeline",
    "recon_mse",
    "run_iterative_inference_test",
    "run_ood_test",
    "run_test_by_name",
]
