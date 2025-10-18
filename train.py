"""Entrypoint for configuring and launching VAE training runs."""

# run.py
import math
import yaml
from utils.dataloaders import GenericImageDataModule
from models import (
    MLPEncoder,
    MLPDecoder,
    ConvEncoder,
    ConvDecoder,
    BaseVAE,
    IterativeVAE,
)
from train_engine import train


def _hidden_pair(hidden):
    """Ensure hidden size lists have at least two entries for encoder/decoder symmetry."""
    if len(hidden) == 1:
        return hidden[0], hidden[0]
    return hidden[0], hidden[1]


def build_model_from_cfg(cfg, input_shape):
    """Instantiate the requested model class with hyperparameters from the config."""
    mcfg = cfg["model"]
    z_dim = mcfg.get("z_dim", 15)
    input_shape = tuple(input_shape)
    input_dim = math.prod(input_shape)
    if mcfg["type"] == "vae_mlp":
        # Fully connected encoder/decoder; default to symmetric hidden sizes.
        enc_hidden = mcfg.get("encoder_hidden", [512, 256])
        dec_hidden = mcfg.get("decoder_hidden", list(reversed(enc_hidden)))
        enc_h1, enc_h2 = _hidden_pair(enc_hidden)
        dec_h1, dec_h2 = _hidden_pair(dec_hidden)
        enc = MLPEncoder(
            input_shape=input_shape,
            input_dim=input_dim,
            h1=enc_h1,
            h2=enc_h2,
            z_dim=z_dim,
        )
        dec = MLPDecoder(
            output_shape=input_shape,
            output_dim=input_dim,
            h1=dec_h1,
            h2=dec_h2,
            z_dim=z_dim,
        )
        model = BaseVAE(
            enc,
            dec,
            input_shape=input_shape,
            z_dim=z_dim,
            lr=mcfg.get("lr", 1e-3),
            beta=mcfg.get("beta", 1.0),
        )
    elif mcfg["type"] == "vae_conv":
        # Convolutional variant: conv/transpose-conv stacks share channel schedule.
        enc_hidden = tuple(mcfg.get("conv_hidden", [32, 64]))
        dec_hidden = tuple(mcfg.get("deconv_hidden", list(reversed(enc_hidden))))
        enc = ConvEncoder(input_shape=input_shape, hidden_dims=enc_hidden, z_dim=z_dim)
        dec = ConvDecoder(output_shape=input_shape, hidden_dims=dec_hidden, z_dim=z_dim)
        model = BaseVAE(
            enc,
            dec,
            input_shape=input_shape,
            z_dim=z_dim,
            lr=mcfg.get("lr", 1e-3),
            beta=mcfg.get("beta", 1.0),
        )
    elif mcfg["type"] == "ivae_iterative":
        # Semi-amortized VAE: reuse MLPs but train with inner-loop SVI.
        enc_hidden = mcfg.get("encoder_hidden", [512, 256])
        dec_hidden = mcfg.get("decoder_hidden", list(reversed(enc_hidden)))
        enc_h1, enc_h2 = _hidden_pair(enc_hidden)
        dec_h1, dec_h2 = _hidden_pair(dec_hidden)
        enc = MLPEncoder(
            input_shape=input_shape,
            input_dim=input_dim,
            h1=enc_h1,
            h2=enc_h2,
            z_dim=z_dim,
        )
        dec = MLPDecoder(
            output_shape=input_shape,
            output_dim=input_dim,
            h1=dec_h1,
            h2=dec_h2,
            z_dim=z_dim,
        )
        model = IterativeVAE(
            enc,
            dec,
            input_shape=input_shape,
            z_dim=z_dim,
            lr_model=mcfg.get("lr", 1e-4),
            lr_inf=mcfg.get("lr_inf", 1e-2),
            svi_steps=mcfg.get("svi_steps", 20),
            beta=mcfg.get("beta", 1.0),
        )
    else:
        raise ValueError("Unknown model type")
    return model


if __name__ == "__main__":
    cfg_path = "configs/default.yaml"
    # Load experiment-wide configuration.
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dm_cfg = cfg["dataset"]
    # Instantiate datamodule with dataset-specific preprocessing options.
    data_module = GenericImageDataModule(
        name=dm_cfg.get("name", "mnist"),
        data_dir=dm_cfg.get("data_dir", "./data"),
        batch_size=dm_cfg.get("batch_size", 128),
        num_workers=dm_cfg.get("num_workers", 4),
        pin_memory=dm_cfg.get("pin_memory", True),
        val_split=dm_cfg.get("val_split", 0.1),
        sun_resize=dm_cfg.get("sun_resize", 64),
        sun_to_grayscale=dm_cfg.get("sun_to_grayscale", True),
        split_seed=dm_cfg.get("split_seed", 42),
    )
    model = build_model_from_cfg(cfg, input_shape=data_module.input_shape)

    # Kick off the Lightning training loop.
    train(cfg=cfg["train"], model=model, datamodule=data_module)
