"""Entrypoint for configuring and launching VAE training runs."""
# Main entry point script for VAE training experiments
# Handles configuration loading, model building, and training orchestration

# run.py
import math  # Mathematical operations for dimension calculations
import yaml  # YAML configuration file parsing
from utils.dataloaders import GenericImageDataModule  # Data loading utilities
from models import (  # Import all model architectures from models package
    MLPEncoder,
    MLPDecoder,
    ConvEncoder,
    ConvDecoder,
    BaseVAE,
    IterativeVAE,
)
from train_engine import train  # Training engine with Lightning trainer setup


def _hidden_pair(hidden):
    """Ensure hidden size lists have at least two entries for encoder/decoder symmetry."""
    # Helper function to handle cases where only one hidden size is specified
    # If only one size provided, use it for both layers (symmetric architecture)
    if len(hidden) == 1:
        return hidden[0], hidden[0]
    # If two or more sizes provided, use first two (ignore extras)
    return hidden[0], hidden[1]


def build_model_from_cfg(cfg, input_shape):
    """Instantiate the requested model class with hyperparameters from the config."""
    # Extract model configuration from main config dict
    mcfg = cfg["model"]
    # Get latent dimension (default to 15 if not specified)
    z_dim = mcfg.get("z_dim", 15)
    # Convert input shape to tuple for consistency
    input_shape = tuple(input_shape)
    # Calculate flattened input dimension for MLP architectures
    input_dim = math.prod(input_shape)
    if mcfg["type"] == "vae_mlp":
        # Fully connected encoder/decoder; default to symmetric hidden sizes.
        # Get encoder hidden layer sizes (default: [512, 256])
        enc_hidden = mcfg.get("encoder_hidden", [512, 256])
        # Get decoder hidden layer sizes (default: reverse of encoder for symmetry)
        dec_hidden = mcfg.get("decoder_hidden", list(reversed(enc_hidden)))
        # Ensure we have exactly two hidden sizes for each network
        enc_h1, enc_h2 = _hidden_pair(enc_hidden)
        dec_h1, dec_h2 = _hidden_pair(dec_hidden)
        # Create MLP encoder with specified architecture
        enc = MLPEncoder(
            input_shape=input_shape,
            input_dim=input_dim,
            h1=enc_h1,
            h2=enc_h2,
            z_dim=z_dim,
        )
        # Create MLP decoder with specified architecture (typically symmetric to encoder)
        dec = MLPDecoder(
            output_shape=input_shape,
            output_dim=input_dim,
            h1=dec_h1,
            h2=dec_h2,
            z_dim=z_dim,
        )
        # Create standard VAE with MLP encoder/decoder
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
        # Get encoder convolutional channel sizes (default: [32, 64])
        enc_hidden = tuple(mcfg.get("conv_hidden", [32, 64]))
        # Get decoder deconvolutional channel sizes (default: reverse of encoder)
        dec_hidden = tuple(mcfg.get("deconv_hidden", list(reversed(enc_hidden))))
        # Create convolutional encoder with specified architecture
        enc = ConvEncoder(input_shape=input_shape, hidden_dims=enc_hidden, z_dim=z_dim)
        # Create convolutional decoder with specified architecture (mirrors encoder)
        dec = ConvDecoder(output_shape=input_shape, hidden_dims=dec_hidden, z_dim=z_dim)
        # Create standard VAE with convolutional encoder/decoder
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
        # Get encoder hidden layer sizes (default: [512, 256])
        enc_hidden = mcfg.get("encoder_hidden", [512, 256])
        # Get decoder hidden layer sizes (default: reverse of encoder for symmetry)
        dec_hidden = mcfg.get("decoder_hidden", list(reversed(enc_hidden)))
        # Ensure we have exactly two hidden sizes for each network
        enc_h1, enc_h2 = _hidden_pair(enc_hidden)
        dec_h1, dec_h2 = _hidden_pair(dec_hidden)
        # Create MLP encoder for amortized inference (initial estimate)
        enc = MLPEncoder(
            input_shape=input_shape,
            input_dim=input_dim,
            h1=enc_h1,
            h2=enc_h2,
            z_dim=z_dim,
        )
        # Create MLP decoder for reconstruction
        dec = MLPDecoder(
            output_shape=input_shape,
            output_dim=input_dim,
            h1=dec_h1,
            h2=dec_h2,
            z_dim=z_dim,
        )
        # Create iterative VAE with SVI refinement (semi-amortized inference)
        model = IterativeVAE(
            enc,
            dec,
            input_shape=input_shape,
            z_dim=z_dim,
            lr_model=mcfg.get("lr", 1e-4),  # Learning rate for model parameters
            lr_inf=mcfg.get("lr_inf", 1e-2),  # Learning rate for SVI inference
            svi_steps=mcfg.get("svi_steps", 20),  # Number of SVI refinement steps
            beta=mcfg.get("beta", 1.0),  # Beta weight for KL term
        )
    else:
        raise ValueError("Unknown model type")
    # Return the constructed model ready for training
    return model


if __name__ == "__main__":
    # Main execution entry point
    cfg_path = "configs/default.yaml"  # Default configuration file path
    # Load experiment-wide configuration from YAML file
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract dataset configuration
    dm_cfg = cfg["dataset"]
    # Instantiate datamodule with dataset-specific preprocessing options
    data_module = GenericImageDataModule(
        name=dm_cfg.get("name", "mnist"),  # Dataset name (default: MNIST)
        data_dir=dm_cfg.get("data_dir", "./data"),  # Data directory (default: ./data)
        batch_size=dm_cfg.get("batch_size", 128),  # Batch size (default: 128)
        num_workers=dm_cfg.get("num_workers", 4),  # Worker processes (default: 4)
        pin_memory=dm_cfg.get("pin_memory", True),  # Memory pinning (default: True)
        val_split=dm_cfg.get("val_split", 0.1),  # Validation split for SUN (default: 0.1)
        sun_resize=dm_cfg.get("sun_resize", 64),  # SUN resize (default: 64x64)
        sun_to_grayscale=dm_cfg.get("sun_to_grayscale", True),  # SUN grayscale (default: True)
        split_seed=dm_cfg.get("split_seed", 42),  # Random seed for splits (default: 42)
    )
    # Build model from configuration using inferred input shape from dataset
    model = build_model_from_cfg(cfg, input_shape=data_module.input_shape)

    # Kick off the PyTorch Lightning training loop with training configuration
    train(cfg=cfg["train"], model=model, datamodule=data_module)
