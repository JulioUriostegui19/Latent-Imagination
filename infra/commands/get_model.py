"""CLI utility for quick model visualization using torchinfo, TensorBoard, and hiddenlayer.

Usage examples:
  - Get an encoder torchinfo summary for MNIST-like inputs:
      python get_model.py --tools torchinfo --model-type encoder --input-shape 1 28 28 --z-dim 32 --hidden-dims 32 64

  - Log a decoder graph to TensorBoard (open with: tensorboard --logdir runs):
      python get_model.py --tools tensorboard --model-type decoder --input-shape 1 28 28 --z-dim 32 --hidden-dims 64 32

  - Run multiple tools at once:
      python get_model.py --tools torchinfo hiddenlayer tensorboard --model-type encoder --input-shape 1 28 28 --hidden-dims 32 64
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

# Import model blocks from the package
from research.models import ConvDecoder, ConvEncoder


def get_mode(
    tools: Sequence[str],
    model_type: str,
    *,
    input_shape: Sequence[int] | None,
    hidden_dims: Sequence[int],
    z_dim: int,
    visual_batch_size: int = 1,
    tensorboard_log_dir: str | None = None,
) -> Dict[str, Any]:
    """Run one or more visualizers on an encoder/decoder with synthetic inputs.

    Args:
      tools: List like ["torchinfo", "tensorboard", "hiddenlayer"].
      model_type: One of {"encoder","decoder"}.
      input_shape: 3 numbers C H W (for decoder this is also the target output shape).
      hidden_dims: Channel schedule for conv stacks.
      z_dim: Latent size.
      visual_batch_size: Batch size for example inputs.
      tensorboard_log_dir: Optional explicit logdir; defaults to runs/<model_type>_visuals.
    """
    toolset = {t.lower() for t in tools}

    if model_type not in {"encoder", "decoder"}:
        raise ValueError("model_type must be 'encoder' or 'decoder'")

    if input_shape is None:
        input_shape = (1, 28, 28)
    else:
        if len(input_shape) != 3:
            raise ValueError("--input-shape must have exactly 3 integers: C H W")
        input_shape = tuple(int(x) for x in input_shape)

    hidden_dims = tuple(int(x) for x in hidden_dims) if hidden_dims else (32, 64)
    z_dim = int(z_dim)

    if model_type == "encoder":
        model = ConvEncoder(
            input_shape=input_shape, hidden_dims=hidden_dims, z_dim=z_dim
        )
    else:
        model = ConvDecoder(
            output_shape=input_shape, hidden_dims=hidden_dims, z_dim=z_dim
        )

    model.eval().to("cpu")

    if model_type == "encoder":
        example_input = torch.randn(visual_batch_size, *input_shape)
    else:
        example_input = torch.randn(visual_batch_size, z_dim)

    log_dir = tensorboard_log_dir or os.path.join("runs", f"{model_type}_visuals")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    if "torchinfo" in toolset:
        try:
            from torchinfo import summary

            info = summary(model, input_data=example_input)
            results["torchinfo"] = str(info)
        except Exception as exc:  # noqa: BLE001
            results["torchinfo"] = f"torchinfo unavailable: {exc}"

    # TensorBoard graph export is now handled by dashboard.py.
    if "tensorboard" in toolset:
        results["tensorboard"] = (
            "TensorBoard support moved. Use dashboard.py --model-overview --tools tensorboard"
        )

    if "hiddenlayer" in toolset:
        try:
            import hiddenlayer as hl

            graph = hl.build_graph(model, example_input)
            # Try to save a PNG artifact for quick viewing
            out_base = os.path.join(log_dir, f"hiddenlayer_{model_type}")
            try:
                graph.save(out_base, format="png")
                results["hiddenlayer"] = f"{out_base}.png"
            except Exception as save_exc:  # noqa: BLE001
                results["hiddenlayer"] = f"graph built (save failed: {save_exc})"
        except Exception as exc:  # noqa: BLE001
            results["hiddenlayer"] = f"hiddenlayer graph failed: {exc}"

    return results


def main():
    parser = argparse.ArgumentParser(description="Quick VAE module visualization")
    parser.add_argument(
        "--tools",
        nargs="+",
        required=True,
        help="One or more tools: torchinfo, tensorboard, hiddenlayer",
    )
    parser.add_argument(
        "--model-type",
        choices=["encoder", "decoder"],
        required=True,
        help="Select which module to visualize",
    )
    parser.add_argument(
        "--input-shape",
        nargs=3,
        type=int,
        metavar=("C", "H", "W"),
        help="Input/output shape as C H W (default: 1 28 28)",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[32, 64],
        help="Conv channel schedule (default: 32 64)",
    )
    parser.add_argument(
        "--z-dim", type=int, default=32, help="Latent dimension (default: 32)"
    )
    parser.add_argument(
        "--visual-batch-size",
        type=int,
        default=1,
        help="Batch size for synthetic example input",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        help="Optional custom logdir for TensorBoard",
    )

    args = parser.parse_args()

    results = get_mode(
        tools=args.tools,
        model_type=args.model_type,
        input_shape=args.input_shape,
        hidden_dims=args.hidden_dims,
        z_dim=args.z_dim,
        visual_batch_size=args.visual_batch_size,
        tensorboard_log_dir=args.tensorboard_log_dir,
    )

    # Pretty print results
    print("=== get_model results ===")
    for key, val in results.items():
        print(f"[{key}] {val}")


if __name__ == "__main__":
    main()
