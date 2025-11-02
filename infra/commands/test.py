"""Evaluation script for generating analytics on trained VAE checkpoints.

This version adds a simple, flexible way to plug in external models
without modifying the code. You can now:

- Import architectures directly via dotted strings (e.g. "pkg.mod:Class")
- Build a VAE from external Encoder/Decoder classes
- Or load a LightningModule from a .ckpt
- Load checkpoints from both .ckpt (Lightning) and .pth (PyTorch)

Backwards compatible: the old `models=[name:path]` or `models=[path]` still works.
"""

import sys, os
import importlib
from typing import Any, Optional
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONFIG_DIR = str((REPO_ROOT / "configs").resolve())

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


def _import_from_string(spec: str) -> Any:
    """Import an object from a string like 'pkg.mod:Obj' or 'pkg.mod.Obj'."""
    module_path: str
    attr_path: str
    if ":" in spec:
        module_path, attr_path = spec.split(":", 1)
    else:
        # split on last dot
        parts = spec.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid import string '{spec}'. Use 'pkg.mod:Obj' or 'pkg.mod.Obj'"
            )
        module_path, attr_path = parts
    module = importlib.import_module(module_path)
    obj = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _maybe_strip_prefix(sd: Mapping[str, torch.Tensor], prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Return a copy of state dict with any of the prefixes stripped from keys."""
    if not prefixes:
        return dict(sd)
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p) :]
                if new_k.startswith("."):
                    new_k = new_k[1:]
        out[new_k] = v
    return out


def _load_state_flex(
    module: torch.nn.Module,
    state: Mapping[str, Any],
    *,
    state_dict_key: Optional[str] = None,
    strip_prefixes: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> None:
    """Load a state dict into a module with a few helpful quality-of-life tweaks.

    - If `state_dict_key` is provided or 'state_dict' is present, it is used.
    - Supports stripping prefixes like 'model.', 'module.', etc.
    - If strict=False (default), it will ignore unexpected/missing keys and print a brief summary.
    """
    sd: Optional[Mapping[str, torch.Tensor]] = None
    if state_dict_key and isinstance(state, Mapping) and state_dict_key in state:
        sd = state[state_dict_key]  # type: ignore[index]
    elif isinstance(state, Mapping) and "state_dict" in state:
        sd = state["state_dict"]  # type: ignore[index]
    elif isinstance(state, Mapping) and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        sd = state  # raw state dict
    else:
        raise KeyError(
            "Cannot find a state dict. Provide a mapping with tensor values, or include 'state_dict'."
        )

    sd = _maybe_strip_prefix(sd, strip_prefixes or [])

    if not strict:
        # Best-effort matching: keep only intersecting keys with shape match
        current = module.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped_bad_shape: List[str] = []
        for k, v in sd.items():
            if k in current and tuple(current[k].shape) == tuple(v.shape):
                filtered[k] = v
            elif k in current:
                skipped_bad_shape.append(k)
        missing = sorted(set(current.keys()) - set(filtered.keys()))
        unexpected = sorted(set(sd.keys()) - set(current.keys()))
        module.load_state_dict(filtered, strict=False)
        if missing or unexpected or skipped_bad_shape:
            print(
                f"[load_state] partial load: missing={len(missing)} unexpected={len(unexpected)} bad_shape={len(skipped_bad_shape)}"
            )
    else:
        module.load_state_dict(sd, strict=True)


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


def _build_model_from_entry(entry: Mapping[str, Any], device: torch.device) -> ModelSpec:
    """Build a model from a structured config entry.

    Supported forms:
      1) LightningModule class + checkpoint
         - class: "pkg.mod:Class"
         - checkpoint: "path/to/model.ckpt" (format auto/ckpt)

      2) Encoder/Decoder classes wrapped into BaseVAE or IterativeVAE
         - encoder: "pkg.mod:EncoderClass"
         - decoder: "pkg.mod:DecoderClass"
         - encoder_init: {...}
         - decoder_init: {...}
         - algorithm: "base" | "iterative" (default: base)
         - algorithm_params: {...} (e.g., beta, lr, lr_inf, svi_steps, weight_decay, input_shape, z_dim)

    Checkpoint loading:
      - checkpoint.path: string, required to load weights
      - checkpoint.format: "auto" | "ckpt" | "pth" (default: auto)
      - checkpoint.state_dict_key: override for nested dict key (default: state_dict)
      - checkpoint.strip_prefixes: list[str] of prefixes to remove from keys
      - checkpoint.strict: bool (default False)
    """
    name = str(entry.get("name", "model"))
    ckpt_cfg = entry.get("checkpoint", {}) or {}
    ckpt_path = ckpt_cfg.get("path")
    ckpt_format = str(ckpt_cfg.get("format", "auto")).lower()
    ckpt_key = ckpt_cfg.get("state_dict_key")
    ckpt_strip = ckpt_cfg.get("strip_prefixes", []) or []
    ckpt_strict = bool(ckpt_cfg.get("strict", False))

    if not ckpt_path:
        raise ValueError(f"Model '{name}' is missing checkpoint.path")
    abs_ckpt_path = Path(to_absolute_path(ckpt_path))
    if not abs_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {abs_ckpt_path}")

    # Optional: extend sys.path for external repos if provided per-entry
    for p in entry.get("pythonpath", []) or []:
        ap = str(Path(to_absolute_path(p)))
        if ap not in sys.path:
            sys.path.insert(0, ap)

    module_obj: torch.nn.Module
    beta: float = float(entry.get("beta", 1.0))
    lr: float = float(entry.get("lr", 1e-3))
    lr_inf: Optional[float] = None

    if "class" in entry or "lightning_class" in entry:
        # Load a LightningModule class and prefer load_from_checkpoint when possible
        cls_path = entry.get("class") or entry.get("lightning_class")
        assert isinstance(cls_path, str)
        LM = _import_from_string(cls_path)
        if ckpt_format in ("auto", "ckpt") and hasattr(LM, "load_from_checkpoint") and str(abs_ckpt_path).endswith(".ckpt"):
            module_obj = LM.load_from_checkpoint(str(abs_ckpt_path), map_location=device)
            # Best effort to pick hyperparameters
            if hasattr(module_obj, "beta"):
                beta = float(getattr(module_obj, "beta"))
            if hasattr(module_obj, "lr_inf"):
                try:
                    lr_inf = float(getattr(module_obj, "lr_inf"))
                except Exception:
                    lr_inf = None
            if hasattr(module_obj, "lr"):
                try:
                    lr = float(getattr(module_obj, "lr"))
                except Exception:
                    pass
        else:
            # Instantiate and then load state dict
            init_kwargs = dict(entry.get("init", {}) or {})
            module_obj = LM(**init_kwargs)
            state = torch.load(abs_ckpt_path, map_location=device)
            _load_state_flex(
                module_obj,
                state,
                state_dict_key=ckpt_key,
                strip_prefixes=ckpt_strip,
                strict=ckpt_strict,
            )
    else:
        # Build from Encoder/Decoder classes and wrap into Base/Iterative VAE
        if "encoder" not in entry or "decoder" not in entry:
            raise ValueError(
                f"Model '{name}' must provide either 'class'/ 'lightning_class' or 'encoder' and 'decoder'."
            )
        Enc = _import_from_string(str(entry["encoder"]))
        Dec = _import_from_string(str(entry["decoder"]))
        enc_init = dict(entry.get("encoder_init", {}) or {})
        dec_init = dict(entry.get("decoder_init", {}) or {})
        encoder = Enc(**enc_init)
        decoder = Dec(**dec_init)

        algo = str(entry.get("algorithm", "base")).lower()
        algo_params = dict(entry.get("algorithm_params", {}) or {})
        # Defaults with override
        input_shape = tuple(algo_params.get("input_shape", enc_init.get("input_shape", (1, 28, 28))))
        z_dim = int(algo_params.get("z_dim", enc_init.get("z_dim", dec_init.get("z_dim", 15))))
        beta = float(algo_params.get("beta", entry.get("beta", 1.0)))
        lr = float(algo_params.get("lr", entry.get("lr", 1e-3)))
        weight_decay = float(algo_params.get("weight_decay", 0.0))
        if algo == "iterative":
            from research.models import IterativeVAE

            lr_model = float(algo_params.get("lr_model", lr))
            lr_inf = float(algo_params.get("lr_inf", 1e-2))
            svi_steps = int(algo_params.get("svi_steps", 20))
            module_obj = IterativeVAE(
                encoder,
                decoder,
                input_shape=input_shape,
                z_dim=z_dim,
                lr_model=lr_model,
                lr_inf=lr_inf,
                svi_steps=svi_steps,
                beta=beta,
                weight_decay=weight_decay,
            )
        else:
            from research.models import BaseVAE

            module_obj = BaseVAE(
                encoder,
                decoder,
                input_shape=input_shape,
                z_dim=z_dim,
                lr=lr,
                beta=beta,
                weight_decay=weight_decay,
            )

        # Load weights into our wrapper from provided checkpoint
        state = torch.load(abs_ckpt_path, map_location=device)
        _load_state_flex(
            module_obj,
            state,
            state_dict_key=ckpt_key,
            strip_prefixes=ckpt_strip,
            strict=ckpt_strict,
        )

    module_obj.eval()
    return ModelSpec(name=name, module=module_obj, beta=beta, lr=lr, lr_inf=lr_inf)


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


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="test")
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

    # Optional: extend sys.path for external repos from config root
    extra_pythonpath = OmegaConf.to_container(cfg.get("pythonpath", []), resolve=True)
    if isinstance(extra_pythonpath, list):
        for p in extra_pythonpath:
            ap = str(Path(to_absolute_path(str(p))))
            if ap not in sys.path:
                sys.path.insert(0, ap)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Support both legacy string list and structured model entries
    models: List[ModelSpec] = []
    models_cfg = OmegaConf.to_container(cfg.models, resolve=True) if cfg.models else []
    if isinstance(models_cfg, list) and models_cfg and isinstance(models_cfg[0], (str,)):
        # Legacy flow: list of strings like ["name:path", "path"]
        for name, path in _parse_model_specs(models_cfg):
            models.append(_load_model_spec(name, path, device=torch.device("cpu")))
    elif isinstance(models_cfg, list):
        # New flow: list of structured entries
        for entry in models_cfg:
            if not isinstance(entry, dict):
                raise TypeError(
                    "Each model entry must be a dict when using the new structured format."
                )
            models.append(_build_model_from_entry(entry, device=torch.device("cpu")))
    else:
        raise TypeError(
            "`models` must be a list of strings or a list of dicts. See configs/test.yaml for examples."
        )

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
