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

from utils.filesystem import ensure_dir


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
    tb_logger = TensorBoardLogger(
        save_dir=cfg.get("save_dir", "./logs"), name="tb_logs"
    )
    csv_logger = CSVLogger(save_dir=cfg.get("save_dir", "./logs"), name="csv_logs")

    callbacks = build_callbacks(cfg)

    trainer_kwargs = {
        "max_epochs": cfg.get("epochs", 50),
        # Use both TensorBoard and CSV loggers for flexibility.
        "logger": [tb_logger, csv_logger],
        "callbacks": callbacks,
        # Select GPU automatically if available and allowed in config.
        "gpus": 1 if torch.cuda.is_available() and cfg.get("use_gpu", True) else 0,
        "precision": 16 if cfg.get("use_amp", False) else 32,
        "gradient_clip_val": cfg.get("grad_clip", 0.0),
        "deterministic": cfg.get("deterministic", False),
        "log_every_n_steps": cfg.get("log_every_n_steps", 50),
    }
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, datamodule=datamodule)

    print(
        "Best checkpoint path:",
        (
            trainer.checkpoint_callback.best_model_path
            if hasattr(trainer, "checkpoint_callback")
            else "N/A"
        ),
    )
    return trainer
