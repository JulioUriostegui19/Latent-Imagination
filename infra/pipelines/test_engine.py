"""Test engine: registry and dispatch for analysis tests.

Collects available test tasks and exposes a simple `run_test_by_name` API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import torch

from research.analysis.common import ModelSpec
from research.analysis.tasks.iterative import run_iterative_inference_test
from research.analysis.tasks.ood import run_ood_test


# Registry of available tests
TEST_REGISTRY: Dict[str, Callable[..., Dict[str, object]]] = {
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
    return fn(
        models=models, loader=loader, cfg=cfg, device=device, output_dir=output_dir
    )
