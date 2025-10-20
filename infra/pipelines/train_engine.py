"""Utility helpers for setting up PyTorch Lightning trainers and callbacks."""

# project/train_engine.py
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from infra.utils.filesystem import ensure_dir


def build_callbacks(cfg):
    """Create checkpointing, early stopping, and LR monitoring callbacks from config."""
    callbacks = []
    ckpt_dir = os.path.join(cfg["save_dir"], "checkpoints")
    ensure_dir(ckpt_dir)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss" if cfg.get("monitor_val_loss", True) else None,
        save_top_k=cfg.get("save_top_k", 3),
        mode="min",
    )
    callbacks.append(checkpoint_cb)

    # Early stopping
    if cfg.get("early_stopping", True):
        es = EarlyStopping(
            monitor="val/loss",
            patience=cfg.get("patience", 10),
            mode="min",
            verbose=True,
        )
        callbacks.append(es)

    # LR monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks


def train(cfg: dict, model: pl.LightningModule, datamodule: pl.LightningDataModule):
    """Run the training loop for the provided Lightning module and datamodule."""
    # Create TensorBoard logger for real-time training visualization
    tb_logger = TensorBoardLogger(
        save_dir=cfg.get("save_dir", "./logs"), name="tb_logs"
    )
    # Create CSV logger for persistent training metrics storage
    csv_logger = CSVLogger(save_dir=cfg.get("save_dir", "./logs"), name="csv_logs")

    # Build training callbacks (checkpointing, early stopping, etc.)
    callbacks = build_callbacks(cfg)

    # Configure PyTorch Lightning trainer with all necessary settings
    use_gpu = bool(cfg.get("use_gpu", True)) and torch.cuda.is_available()
    # Gradient clipping is not supported under manual optimization
    requested_clip = float(cfg.get("grad_clip", 0.0))
    if getattr(model, "automatic_optimization", True) is False and requested_clip:
        print(
            f"[train] Disabling gradient clipping (requested {requested_clip}) because manual optimization is enabled.")
        requested_clip = 0.0

    trainer_kwargs = {
        "max_epochs": cfg.get("epochs", 50),
        "logger": [tb_logger, csv_logger],
        "callbacks": callbacks,
        # Lightning >=2.0 device configuration
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": 1 if use_gpu else 1,
        "precision": 16 if cfg.get("use_amp", False) else 32,
        # Include gradient clipping only when supported/requested
        **({"gradient_clip_val": requested_clip} if requested_clip > 0 else {}),
        "deterministic": cfg.get("deterministic", False),
        "log_every_n_steps": cfg.get("log_every_n_steps", 50),
    }
    # Create PyTorch Lightning trainer with configured settings
    trainer = pl.Trainer(**trainer_kwargs)

    # Start the actual training process
    trainer.fit(model, datamodule=datamodule)

    # Print path to best checkpoint after training completes
    print(
        "Best checkpoint path:",
        (
            trainer.checkpoint_callback.best_model_path
            if hasattr(trainer, "checkpoint_callback")
            else "N/A"
        ),
    )
    # Return trainer object for potential further use (analysis, testing, etc.)
    return trainer
