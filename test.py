"""Evaluation script for generating analytics on trained VAE checkpoints."""

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from analytics import ModelSpec, run_iterative_inference_test, run_ood_test
from models import BaseVAE, IterativeVAE, ConvDecoder, ConvEncoder, MLPDecoder, MLPEncoder
from utils.dataloaders import GenericImageDataModule


def _parse_model_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """Accept either `name:path` pairs or bare checkpoint paths."""
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
    """Guess whether the checkpoint corresponds to an MLP or Conv backbone."""
    for key in state_dict.keys():
        if key.startswith("encoder.conv"):
            return "conv"
    return "mlp"


def _build_modules(
    architecture: str,
    hparams: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Rebuild encoder/decoder modules that match the checkpoint layout."""
    input_shape = tuple(hparams.get("input_shape", (1, 28, 28)))
    z_dim = int(hparams.get("z_dim", 15))
    input_dim = math.prod(input_shape)

    if architecture == "mlp":
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
        conv_keys = sorted(
            key for key in state_dict if key.startswith("encoder.conv") and key.endswith(".weight")
        )
        hidden_dims = tuple(int(state_dict[key].shape[0]) for key in conv_keys)
        if not hidden_dims:
            hidden_dims = (32, 64)
        decoder_keys = sorted(
            key for key in state_dict if key.startswith("decoder.net") and key.endswith(".weight")
        )
        if decoder_keys:
            init_channels = int(state_dict[decoder_keys[0]].shape[0])
            extra = [int(state_dict[key].shape[1]) for key in decoder_keys[:-1]]
            dec_hidden = tuple([init_channels, *extra])
        else:
            dec_hidden = (64, 32)
        encoder = ConvEncoder(input_shape=input_shape, hidden_dims=hidden_dims, z_dim=z_dim)
        decoder = ConvDecoder(output_shape=input_shape, hidden_dims=dec_hidden, z_dim=z_dim)
    return encoder, decoder


def _load_model_spec(name: str, checkpoint_path: str, device: torch.device) -> ModelSpec:
    """Materialise a ModelSpec from a Lightning checkpoint."""
    abs_path = Path(to_absolute_path(checkpoint_path))
    if not abs_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {abs_path}")

    ckpt = torch.load(abs_path, map_location=device)
    state_dict: Dict[str, torch.Tensor] = ckpt["state_dict"]
    hparams: Mapping[str, object] = ckpt.get("hyper_parameters", {})

    architecture = _infer_architecture(state_dict)
    encoder, decoder = _build_modules(architecture, hparams, state_dict)

    beta = float(hparams.get("beta", 1.0))
    lr = float(hparams.get("lr", 1e-3))
    weight_decay = float(hparams.get("weight_decay", 0.0))
    input_shape = tuple(hparams.get("input_shape", (1, 28, 28)))
    z_dim = int(hparams.get("z_dim", 15))

    if "lr_inf" in hparams:
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

    module.load_state_dict(state_dict)
    module.eval()
    return ModelSpec(name=name, module=module, beta=beta, lr=lr, lr_inf=lr_inf)


def _prepare_dataloader(cfg: Mapping[str, object]) -> Iterable:
    """Instantiate the validation loader with absolute data directory resolution."""
    dataset_cfg = OmegaConf.to_container(cfg, resolve=True)
    dataset_cfg["data_dir"] = to_absolute_path(dataset_cfg.get("data_dir", "./data"))
    datamodule = GenericImageDataModule(**dataset_cfg)
    datamodule.prepare_data()
    datamodule.setup("test")
    return datamodule.val_dataloader()


@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig):
    """Hydra entrypoint for running analytics suites against checkpoints."""
    if not cfg.models:
        raise ValueError("Please provide at least one model via `models=[path_or_name:path]`.")
    if not cfg.tests:
        raise ValueError("Please provide at least one test in `tests=[iterative,ood,...]`.")

    model_specs_input = _parse_model_specs(cfg.models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        _load_model_spec(name, path, device=torch.device("cpu"))
        for name, path in model_specs_input
    ]

    val_loader = _prepare_dataloader(cfg.dataset)

    output_root = Path(to_absolute_path(cfg.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, object]] = {}

    for test_name in cfg.tests:
        if test_name == "iterative":
            test_dir = output_root / "iterative_inference"
            test_dir.mkdir(exist_ok=True, parents=True)
            metrics["iterative"] = run_iterative_inference_test(
                models=models,
                loader=val_loader,
                cfg=OmegaConf.to_container(cfg.test_settings.iterative, resolve=True),
                device=device,
                output_dir=test_dir,
            )
        elif test_name == "ood":
            test_dir = output_root / "ood_analysis"
            test_dir.mkdir(exist_ok=True, parents=True)
            metrics["ood"] = run_ood_test(
                models=models,
                loader=val_loader,
                cfg=OmegaConf.to_container(cfg.test_settings.ood, resolve=True),
                device=device,
                output_dir=test_dir,
            )
        else:
            raise ValueError(f"Unknown test requested: {test_name}")

    summary_path = output_root / "metrics_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as fp:
        fp.write(OmegaConf.to_yaml(metrics))
    print(f"[✓] Evaluation complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
