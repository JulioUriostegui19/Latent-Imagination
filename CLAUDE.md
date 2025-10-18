# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based machine learning research project implementing Variational Autoencoders (VAEs) for neuroscience applications. The codebase supports both standard amortized VAEs and iterative/semi-amortized VAEs with stochastic variational inference (SVI).

## Commands

### Training Models
```bash
# Train with default configuration
python train.py

# Train with custom configuration
python train.py  # Update configs/default.yaml first
```

### Monitoring Training
```bash
# Launch TensorBoard to monitor training progress
tensorboard --logdir ./runs
```

## Architecture

### Model Types
- **vae_mlp**: Fully-connected VAE with MLP encoder/decoder
- **vae_conv**: Convolutional VAE with conv/transpose-conv layers
- **ivae_iterative**: Semi-amortized VAE with inner-loop SVI refinement

### Key Components
- **train.py**: Main entry point that loads config, builds model, and starts training
- **train_engine.py**: PyTorch Lightning training utilities with callbacks and logging
- **models/**: Model implementations split between standard VAE and iterative VAE
- **utils/**: Data loading, loss functions, and filesystem utilities
- **configs/**: YAML configuration files for experiments

### Configuration System
All experiments are configured via YAML files with three main sections:
- `dataset`: Data loading parameters (name, batch_size, num_workers, etc.)
- `model`: Architecture hyperparameters (type, z_dim, lr, beta, etc.)
- `train`: Training loop settings (epochs, save_dir, use_amp, early_stopping, etc.)

### Dependencies
The project requires:
- PyTorch
- PyTorch Lightning
- Torchvision
- PyYAML
- TensorBoard

Note: No requirements.txt exists - dependencies must be installed manually.

### Training Workflow
1. Configuration loaded from `configs/default.yaml`
2. Data module instantiated based on dataset config (MNIST or SUN397)
3. Model built from config specifying architecture type and hyperparameters
4. PyTorch Lightning trainer configured with callbacks and loggers
5. Training runs with automatic checkpointing and early stopping
6. Results saved to `./runs/` directory with TensorBoard and CSV logs

### Key Design Patterns
- **Modular architecture**: Separate encoder/decoder classes that can be mixed and matched
- **Configuration-driven**: All hyperparameters externalized to YAML files
- **PyTorch Lightning**: Clean separation of model logic from training loop
- **Automatic GPU detection**: Uses GPU if available, falls back to CPU
- **Mixed precision support**: Configurable AMP training for efficiency