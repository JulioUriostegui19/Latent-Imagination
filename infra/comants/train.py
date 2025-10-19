"""Entrypoint for configuring and launching VAE training runs via Hydra."""

import math  # Mathematical operations for dimension calculations
from typing import Mapping, Sequence  # Type hints for function parameters and returns

import hydra  # Hydra framework for configuration management
from hydra.utils import to_absolute_path  # Utility to convert relative paths to absolute
from omegaconf import DictConfig, OmegaConf  # OmegaConf for configuration handling

from infra.utils.dataloaders import GenericImageDataModule  # Data loading utilities
from research.models import (  # Import all model components
    MLPEncoder,  # Multi-layer perceptron encoder
    MLPDecoder,  # Multi-layer perceptron decoder
    ConvEncoder,  # Convolutional encoder
    ConvDecoder,  # Convolutional decoder
    BaseVAE,  # Standard VAE implementation
    IterativeVAE,  # Iterative VAE with SVI refinement
)
from infra.pipelines.train_engine import train  # Training engine function


def _hidden_pair(hidden: Sequence[int]) -> tuple[int, int]:  # Helper function to ensure encoder/decoder symmetry
    """Ensure hidden size lists have at least two entries for encoder/decoder symmetry."""
    # Accept single-element lists (e.g., [512]) and mirror the value for deeper layers.
    if len(hidden) == 1:
        return hidden[0], hidden[0]
    # Ignore additional entries beyond the first two to keep architecture compact.
    return hidden[0], hidden[1]


def build_model_from_cfg(model_cfg: Mapping, input_shape):  # Factory function to create models from configuration
    """Instantiate the requested model class with hyperparameters from the config."""
    # Hydrated OmegaConf mapping passed in; use regular dict access for ergonomics.
    mcfg = model_cfg
    z_dim = mcfg.get("z_dim", 15)  # Latent dimension size, default 15
    input_shape = tuple(input_shape)  # Convert input shape to tuple
    input_dim = math.prod(input_shape)  # Calculate total input dimensions for MLP layers

    if mcfg["type"] == "vae_mlp":
        enc_hidden = mcfg.get("encoder_hidden", [512, 256])  # default sensible sizes
        dec_hidden = mcfg.get("decoder_hidden", list(reversed(enc_hidden)))  # mirror encoder
        enc_h1, enc_h2 = _hidden_pair(enc_hidden)  # Extract encoder hidden dimensions
        dec_h1, dec_h2 = _hidden_pair(dec_hidden)  # Extract decoder hidden dimensions
        encoder = MLPEncoder(  # Create MLP encoder instance
            input_shape=input_shape,
            input_dim=input_dim,
            h1=enc_h1,
            h2=enc_h2,
            z_dim=z_dim,
        )
        decoder = MLPDecoder(  # Create MLP decoder instance
            output_shape=input_shape,
            output_dim=input_dim,
            h1=dec_h1,
            h2=dec_h2,
            z_dim=z_dim,
        )
        model = BaseVAE(  # Create base VAE model with MLP components
            encoder,
            decoder,
            input_shape=input_shape,
            z_dim=z_dim,
            lr=mcfg.get("lr", 1e-3),
            beta=mcfg.get("beta", 1.0),
            weight_decay=mcfg.get("weight_decay", 0.0),
        )
    elif mcfg["type"] == "vae_conv":
        enc_hidden = tuple(mcfg.get("conv_hidden", [32, 64]))  # channel schedule for downsampling
        dec_hidden = tuple(mcfg.get("deconv_hidden", list(reversed(enc_hidden))))  # upsampling schedule
        encoder = ConvEncoder(input_shape=input_shape, hidden_dims=enc_hidden, z_dim=z_dim)  # Create convolutional encoder
        decoder = ConvDecoder(output_shape=input_shape, hidden_dims=dec_hidden, z_dim=z_dim)  # Create convolutional decoder
        model = BaseVAE(  # Create base VAE model with convolutional components
            encoder,
            decoder,
            input_shape=input_shape,
            z_dim=z_dim,
            lr=mcfg.get("lr", 1e-3),
            beta=mcfg.get("beta", 1.0),
            weight_decay=mcfg.get("weight_decay", 0.0),
        )
    elif mcfg["type"] == "ivae_iterative":  # Iterative VAE with SVI refinement
        enc_hidden = mcfg.get("encoder_hidden", [512, 256])
        dec_hidden = mcfg.get("decoder_hidden", list(reversed(enc_hidden)))
        enc_h1, enc_h2 = _hidden_pair(enc_hidden)  # Extract encoder hidden dimensions
        dec_h1, dec_h2 = _hidden_pair(dec_hidden)  # Extract decoder hidden dimensions
        encoder = MLPEncoder(  # Create MLP encoder for iterative VAE
            input_shape=input_shape,
            input_dim=input_dim,
            h1=enc_h1,
            h2=enc_h2,
            z_dim=z_dim,
        )
        decoder = MLPDecoder(  # Create MLP decoder for iterative VAE
            output_shape=input_shape,
            output_dim=input_dim,
            h1=dec_h1,
            h2=dec_h2,
            z_dim=z_dim,
        )
        model = IterativeVAE(  # Create iterative VAE with SVI capabilities
            encoder,
            decoder,
            input_shape=input_shape,
            z_dim=z_dim,
            lr_model=mcfg.get("lr", 1e-4),  # Learning rate for model parameters
            lr_inf=mcfg.get("lr_inf", 1e-2),  # Learning rate for inference
            svi_steps=mcfg.get("svi_steps", 20),  # Number of SVI refinement steps
            beta=mcfg.get("beta", 1.0),
            weight_decay=mcfg.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type '{mcfg['type']}'")  # Error for unsupported model types
    return model  # Return the constructed model


@hydra.main(version_base=None, config_path="configs", config_name="config")  # Hydra decorator for configuration management
def main(cfg: DictConfig):  # Main entry point function
    """Hydra entrypoint for training configured VAE variants."""
    # Convert OmegaConf nodes to plain Python containers for Lightning/datamodule usage.
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)  # Convert dataset config to Python dict
    # Convert relative data directory to absolute path so training works from Hydra's run dir.
    dataset_cfg["data_dir"] = to_absolute_path(dataset_cfg.get("data_dir", "./data"))
    data_module = GenericImageDataModule(**dataset_cfg)  # Create data module from configuration

    model = build_model_from_cfg(cfg.model, input_shape=data_module.input_shape)  # Build model from config

    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)  # Convert training config to Python dict
    # Persist logs/checkpoints under the original project tree rather than Hydra's subdir.
    train_cfg["save_dir"] = to_absolute_path(train_cfg.get("save_dir", "./runs"))

    train(cfg=train_cfg, model=model, datamodule=data_module)  # Start training process


if __name__ == "__main__":  # Standard Python entry point guard
    main()  # Execute main function
