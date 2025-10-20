"""Evaluation script for generating analytics on trained VAE checkpoints."""

import sys, os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math  # Mathematical operations for dimension calculations
from pathlib import Path  # Path manipulation utilities
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple  # Type hints for function signatures

import hydra  # Hydra framework for configuration management
import torch  # PyTorch for deep learning operations
from hydra.utils import to_absolute_path  # Utility to convert relative paths to absolute
from omegaconf import DictConfig, OmegaConf  # OmegaConf for configuration handling

from research.analysis import (
    ModelSpec,
    run_test_by_name,
)
from research.models import (  # Import all model components
    BaseVAE,
    IterativeVAE,
    ConvDecoder,
    ConvEncoder,
    MLPDecoder,
    MLPEncoder,
)
from infra.utils.dataloaders import GenericImageDataModule  # Data loading utilities


def _parse_model_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """Parse model specifications from command line arguments.

    Accepts either `name:path` pairs (e.g., "baseline:runs/checkpoint.ckpt")
    or bare checkpoint paths (e.g., "runs/checkpoint.ckpt"). For bare paths,
    uses the filename stem as the model name.
    """
    parsed = []
    for spec in specs:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            name = Path(spec).stem
            path = spec
        parsed.append((name, path))
    return parsed


def _infer_architecture(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Infer model architecture type from checkpoint state dictionary.

    Examines parameter names to determine if model uses convolutional layers.
    Conv models have 'encoder.conv' keys, while MLP models only have 'encoder.fc' keys.
    """
    for key in state_dict.keys():
        if key.startswith("encoder.conv"):
            return "conv"
    return "mlp"


def _build_modules(
    architecture: str,
    hparams: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Reconstruct encoder/decoder modules from checkpoint parameters.

    Dynamically determines layer dimensions from saved weights to ensure
    compatibility with the original architecture. Handles both MLP and Conv
    architectures by examining the shape of saved parameters.
    """
    input_shape = tuple(hparams.get("input_shape", (1, 28, 28)))  # Default MNIST shape
    z_dim = int(hparams.get("z_dim", 15))  # Default latent dimension
    input_dim = math.prod(input_shape)  # Calculate flattened input size

    if architecture == "mlp":
        # Extract hidden layer dimensions from weight matrices
        h1 = state_dict["encoder.fc1.weight"].shape[0]
        h2 = state_dict["encoder.fc2.weight"].shape[0]
        dec_h1 = state_dict["decoder.fc1.weight"].shape[0]
        dec_h2 = state_dict["decoder.fc2.weight"].shape[0]
        encoder = MLPEncoder(
            input_shape=input_shape,
            input_dim=input_dim,
            h1=h1,
            h2=h2,
            z_dim=z_dim,
        )
        decoder = MLPDecoder(
            output_shape=input_shape,
            output_dim=input_dim,
            h1=dec_h1,
            h2=dec_h2,
            z_dim=z_dim,
        )
    else:
        # Extract convolutional layer dimensions from weight tensors
        conv_keys = sorted(
            key
            for key in state_dict
            if key.startswith("encoder.conv") and key.endswith(".weight")
        )
        hidden_dims = tuple(int(state_dict[key].shape[0]) for key in conv_keys)
        if not hidden_dims:
            hidden_dims = (32, 64)  # Fallback default for conv layers

        # Extract decoder layer dimensions (transpose convolutions)
        decoder_keys = sorted(
            key
            for key in state_dict
            if key.startswith("decoder.net") and key.endswith(".weight")
        )
        if decoder_keys:
            init_channels = int(state_dict[decoder_keys[0]].shape[0])
            extra = [int(state_dict[key].shape[1]) for key in decoder_keys[:-1]]
            dec_hidden = tuple([init_channels, *extra])
        else:
            dec_hidden = (64, 32)  # Fallback default for decoder

        encoder = ConvEncoder(
            input_shape=input_shape, hidden_dims=hidden_dims, z_dim=z_dim
        )
        decoder = ConvDecoder(
            output_shape=input_shape, hidden_dims=dec_hidden, z_dim=z_dim
        )
    return encoder, decoder


def _load_model_spec(
    name: str, checkpoint_path: str, device: torch.device
) -> ModelSpec:
    """Load and reconstruct a model from checkpoint for evaluation.

    Handles multiple checkpoint formats:
    - PyTorch Lightning checkpoints (.ckpt files)
    - Plain PyTorch state dicts (.pth files)
    - Legacy format with separate encoder/decoder dicts

    Automatically determines architecture type and reconstructs the appropriate
    model class (BaseVAE or IterativeVAE) with original hyperparameters.
    """
    abs_path = Path(to_absolute_path(checkpoint_path))  # Convert to absolute path
    if not abs_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {abs_path}")

    ckpt = torch.load(abs_path, map_location=device)
    # Lightning `.ckpt` files store weights under `state_dict`. Plain PyTorch `.pth`
    # checkpoints might already be a flat parameter mapping, so handle both.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict: Dict[str, torch.Tensor] = ckpt["state_dict"]
        hparams: Mapping[str, object] = ckpt.get("hyper_parameters", {})
    elif isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        state_dict = ckpt
        hparams = {}
    elif isinstance(ckpt, dict) and {"encoder", "decoder"} <= ckpt.keys():
        # Legacy checkpoint format (pre-Lightning) used only for testing legacy models.
        state_dict = {f"encoder.{k}": v for k, v in ckpt["encoder"].items()}
        state_dict.update({f"decoder.{k}": v for k, v in ckpt["decoder"].items()})
        hparams = ckpt.get("hyper_parameters", {})
    else:
        raise KeyError(
            "Checkpoint must either contain a 'state_dict' entry (Lightning) "
            "or be a raw parameter dictionary."
        )

    architecture = _infer_architecture(state_dict)
    encoder, decoder = _build_modules(architecture, hparams, state_dict)

    beta = float(hparams.get("beta", 1.0))  # KL divergence weight
    lr = float(hparams.get("lr", 1e-3))
    weight_decay = float(hparams.get("weight_decay", 0.0))
    input_shape = tuple(hparams.get("input_shape", (1, 28, 28)))
    z_dim = int(hparams.get("z_dim", 15))

    if "lr_inf" in hparams:  # Check if this is an iterative VAE
        module = IterativeVAE(
            encoder,
            decoder,
            input_shape=input_shape,
            z_dim=z_dim,
            lr_model=float(hparams.get("lr_model", lr)),
            lr_inf=float(hparams["lr_inf"]),
            svi_steps=int(hparams.get("svi_steps", 20)),
            beta=beta,
            weight_decay=weight_decay,
        )
        lr_inf = float(hparams["lr_inf"])
    else:
        module = BaseVAE(
            encoder,
            decoder,
            input_shape=input_shape,
            z_dim=z_dim,
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
        )
        lr_inf = None

    module.load_state_dict(state_dict)  # Load trained weights
    module.eval()  # Set to evaluation mode
    return ModelSpec(name=name, module=module, beta=beta, lr=lr, lr_inf=lr_inf)


def _prepare_dataloader(cfg: Mapping[str, object]) -> Iterable:
    """Create validation data loader for model evaluation.

    Converts relative data directory paths to absolute paths and initializes
    the data module with the same configuration used during training.
    Returns the validation dataloader for consistent evaluation across models.
    """
    dataset_cfg = OmegaConf.to_container(cfg, resolve=True)
    dataset_cfg["data_dir"] = to_absolute_path(dataset_cfg.get("data_dir", "./data"))
    datamodule = GenericImageDataModule(**dataset_cfg)
    datamodule.prepare_data()
    datamodule.setup("test")
    return datamodule.val_dataloader()


@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig):
    """Main entry point for model evaluation and analytics generation.

    Orchestrates the complete evaluation pipeline:
    1. Parses model specifications from command line
    2. Loads and reconstructs models from checkpoints
    3. Prepares validation data loader
    4. Runs requested test suites (iterative inference, OOD analysis)
    5. Saves comprehensive metrics summary

    Args:
        cfg: Hydra configuration containing models, tests, and settings
    """
    if not cfg.models:
        raise ValueError(
            "Please provide at least one model via `models=[path_or_name:path]`."
        )
    if not cfg.tests:
        raise ValueError(
            "Please provide at least one test in `tests=[iterative,ood,...]`."
        )

    model_specs_input = _parse_model_specs(cfg.models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models on CPU first to avoid GPU memory issues with multiple models
    models = [
        _load_model_spec(name, path, device=torch.device("cpu"))
        for name, path in model_specs_input
    ]

    val_loader = _prepare_dataloader(cfg.dataset)

    # Use Hydra's run directory for outputs (e.g., outputs/YYYY-MM-DD/HH-MM-SS)
    output_root = Path.cwd()
    tests_root = output_root / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, object]] = {}

    # Execute each requested test suite
    for test_name in cfg.tests:
        test_dir = tests_root / test_name
        test_dir.mkdir(exist_ok=True, parents=True)

        # Build test-specific config
        test_cfg_node = getattr(cfg.test_settings, test_name, None)
        if test_cfg_node is None:
            raise ValueError(f"Missing test_settings for '{test_name}'")
        test_cfg = OmegaConf.to_container(test_cfg_node, resolve=True)

        # Save an overview of this test run
        overview = {
            "test": test_name,
            "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
            "test_cfg": test_cfg,
            "models": [
                {
                    "name": m.name,
                    "module": m.module.__class__.__name__,
                    "beta": m.beta,
                    "lr": m.lr,
                    "lr_inf": m.lr_inf,
                }
                for m in models
            ],
        }
        with open(test_dir / "overview.yaml", "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(overview))

        # Dispatch via registry
        metrics[test_name] = run_test_by_name(
            test_name,
            models=models,
            loader=val_loader,
            cfg=test_cfg,
            device=device,
            output_dir=test_dir,
        )

    # Save comprehensive metrics summary
    summary_path = tests_root / "metrics_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as fp:
        fp.write(OmegaConf.to_yaml(metrics))
    print(f"[✓] Evaluation complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
