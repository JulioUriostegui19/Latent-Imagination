"""CLI utility for quick model visualization using torchinfo and torchviz.

New usage (config-driven):
  - Point to a model YAML (e.g., configs/model/vae_conv.yaml):
      python run.py get_model configs/model/vae_conv.yaml --which encoder --tools torchinfo torchviz

Notes:
  - The YAML must include a `type` key: one of {vae_conv, vae_mlp, ivae_iterative}.
  - For conv models, reads `conv_hidden`/`deconv_hidden`; for MLP, reads
    `encoder_hidden`/`decoder_hidden`; always uses `z_dim`.
  - Input shape defaults to 1x28x28; override with --input-shape if desired.
"""

from __future__ import annotations

import argparse
import sys, os
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from omegaconf import OmegaConf

# Ensure repo root is on sys.path so 'research' package resolves
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.models import ConvDecoder, ConvEncoder, MLPEncoder, MLPDecoder


def get_mode(
    tools: Sequence[str],
    which: str,
    *,
    cfg_path: str,
    input_shape: Sequence[int] | None,
    visual_batch_size: int = 1,
    tensorboard_log_dir: str | None = None,
) -> Dict[str, Any]:
    """Run one or more visualizers on an encoder/decoder with synthetic inputs.

    Args:
      tools: List like ["torchinfo", "tensorboard", "torchviz"].
      model_type: One of {"encoder","decoder"}.
      input_shape: 3 numbers C H W (for decoder this is also the target output shape).
      hidden_dims: Channel schedule for conv stacks.
      z_dim: Latent size.
      visual_batch_size: Batch size for example inputs.
      tensorboard_log_dir: Optional explicit logdir; defaults to runs/<model_type>_visuals.
    """
    toolset = {t.lower() for t in tools}

    if which not in {"encoder", "decoder"}:
        raise ValueError("--which must be 'encoder' or 'decoder'")

    # Load model config YAML
    mcfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(mcfg, dict) or "type" not in mcfg:
        raise ValueError("Model YAML must be a mapping with a 'type' key")

    mtype = str(mcfg["type"]).lower()
    z_dim = int(mcfg.get("z_dim", 32))

    if input_shape is None:
        input_shape = (1, 28, 28)
    else:
        if len(input_shape) != 3:
            raise ValueError("--input-shape must have exactly 3 integers: C H W")
        input_shape = tuple(int(x) for x in input_shape)

    # Build the requested module from config type
    if mtype == "vae_conv":
        if which == "encoder":
            hidden = tuple(int(x) for x in mcfg.get("conv_hidden", [32, 64]))
            model = ConvEncoder(input_shape=input_shape, hidden_dims=hidden, z_dim=z_dim)
        else:
            hidden = tuple(int(x) for x in mcfg.get("deconv_hidden", [64, 32]))
            model = ConvDecoder(output_shape=input_shape, hidden_dims=hidden, z_dim=z_dim)
    elif mtype in {"vae_mlp", "ivae_iterative"}:
        # Treat IVAE as MLP blocks for the purpose of static visualization
        if which == "encoder":
            enc = mcfg.get("encoder_hidden", [512, 256])
            input_dim = int(input_shape[0] * input_shape[1] * input_shape[2])
            model = MLPEncoder(
                input_shape=input_shape,
                input_dim=input_dim,
                h1=int(enc[0]),
                h2=int(enc[1] if len(enc) > 1 else enc[0]),
                z_dim=z_dim,
            )
        else:
            dec = mcfg.get("decoder_hidden", [256, 512])
            output_dim = int(input_shape[0] * input_shape[1] * input_shape[2])
            model = MLPDecoder(
                output_shape=input_shape,
                output_dim=output_dim,
                h1=int(dec[0]),
                h2=int(dec[1] if len(dec) > 1 else dec[0]),
                z_dim=z_dim,
            )
    else:
        raise ValueError(f"Unknown model type '{mtype}' in YAML")

    model.eval().to("cpu")

    if which == "encoder":
        example_input = torch.randn(visual_batch_size, *input_shape)
    else:
        example_input = torch.randn(visual_batch_size, z_dim)

    log_dir = tensorboard_log_dir or os.path.join("runs", f"{which}_visuals")
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

    # Support torchviz (and treat 'hiddenlayer' as an alias for backward compatibility)
    if "torchviz" in toolset or "hiddenlayer" in toolset:
        try:
            from torchviz import make_dot

            # Ensure autograd tracks the graph for visualization
            ex_inp = example_input.clone().detach().requires_grad_(True)
            out = model(ex_inp)
            if isinstance(out, (tuple, list)):
                out = out[0]
            dot = make_dot(out, params=dict(model.named_parameters()))
            out_base = os.path.join(log_dir, f"torchviz_{which}")
            dot.render(out_base, format="png")
            results["torchviz"] = f"{out_base}.png"
            if "hiddenlayer" in toolset and "torchviz" not in toolset:
                # Note about the implicit aliasing
                results["note"] = (
                    "'hiddenlayer' is deprecated here; generated graph with torchviz"
                )
        except Exception as exc:  # noqa: BLE001
            results["torchviz"] = f"torchviz failed: {exc}"

    return results


def main():
    parser = argparse.ArgumentParser(description="Quick model visualization from YAML config")
    parser.add_argument("config", help="Path to model YAML (e.g., configs/model/vae_conv.yaml)")
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["torchinfo"],
        help="One or more tools: torchinfo, torchviz",
    )
    parser.add_argument(
        "--which",
        choices=["encoder", "decoder"],
        default="encoder",
        help="Which module to visualize",
    )
    parser.add_argument(
        "--input-shape",
        nargs=3,
        type=int,
        metavar=("C", "H", "W"),
        help="Override input/output shape as C H W (default: 1 28 28)",
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
        which=args.which,
        cfg_path=args.config,
        input_shape=args.input_shape,
        visual_batch_size=args.visual_batch_size,
        tensorboard_log_dir=args.tensorboard_log_dir,
    )

    # Pretty print results
    print("=== get_model results ===")
    for key, val in results.items():
        print(f"[{key}] {val}")


if __name__ == "__main__":
    main()
