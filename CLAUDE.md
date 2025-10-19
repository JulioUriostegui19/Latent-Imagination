# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based machine learning research project implementing Variational Autoencoders (VAEs) for neuroscience applications. The codebase supports both standard amortized VAEs and iterative/semi-amortized VAEs with stochastic variational inference (SVI).

## Commands

### Training Models
```bash
# Train with default configuration (MNIST dataset, MLP VAE)
python train.py

# Train with different datasets
python train.py dataset=mnist
python train.py dataset=sun

# Train different model architectures
python train.py model=vae_mlp
python train.py model=vae_conv
python train.py model=ivae_iterative

# Train with parameter overrides
python train.py model.vae_mlp.z_dim=32 train.epochs=50
python train.py model.ivae_iterative.svi_steps=30 train.lr=1e-4

# Train with custom config file
python train.py --config-path configs --config-name my_experiment
```

### Model Evaluation and Testing
```bash
# Run evaluation on trained models (requires checkpoint paths)
python test.py models=["path/to/checkpoint.ckpt"] tests=["iterative","ood"]

# Evaluate multiple models with custom names
python test.py models=["baseline:path/to/baseline.ckpt","iterative:path/to/iterative.ckpt"] tests=["iterative"]

# Run specific test types
python test.py models=["model.ckpt"] tests=["iterative"]  # Iterative inference analysis
python test.py models=["model.ckpt"] tests=["ood"]        # Out-of-distribution testing
```

### Monitoring & Dashboard
```bash
# Launch TensorBoard via dashboard helper (recommended)
python dashboard.py --launch-tensorboard --logdir runs

# Print effective configs used by train/test
python get_cfg.py --train
python get_cfg.py --test

# Quick model visualization (encoder/decoder)
python get_model.py --tools torchinfo --model-type encoder --input-shape 1 28 28
python dashboard.py --model-overview --tools torchinfo tensorboard --model-type decoder --input-shape 1 28 28
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt
# Optional tools used by get_model/dashboard
pip install torchinfo hiddenlayer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Architecture

### Model Types
- **vae_mlp**: Fully-connected VAE with MLP encoder/decoder
- **vae_conv**: Convolutional VAE with conv/transpose-conv layers
- **ivae_iterative**: Semi-amortized VAE with inner-loop SVI refinement

### Key Components
- **train.py**: Main entry point that loads config, builds model, and starts training
- **train_engine.py**: PyTorch Lightning training utilities with callbacks and logging
- **test.py**: Model evaluation script for analytics and testing
- **get_cfg.py**: Prints composed/effective Hydra configs for train/test
- **get_model.py**: CLI to visualize encoder/decoder with torchinfo/hiddenlayer
- **dashboard.py**: Colab-friendly helper to launch TensorBoard, print configs, run overviews and simple sweeps
- **models/**: Model implementations split between standard VAE and iterative VAE
  - `models/vae/`: Standard VAE implementations (base.py, architectures.py)
  - `models/ivae/`: Iterative VAE with SVI refinement (iterative.py)
- **utils/**: Data loading, loss functions, and filesystem utilities
- **analytics/**: Evaluation and testing utilities
- **configs/**: YAML configuration files for experiments

### Configuration System
Hydra manages configuration through composable YAML files:
- `configs/config.yaml`: Main config defining defaults (dataset, model, train groups)
- `configs/dataset/*.yaml`: Dataset-specific parameters (MNIST, SUN397)
- `configs/model/*.yaml`: Architecture hyperparameters for each model type
- `configs/train/*.yaml`: Training-loop settings
- `configs/test.yaml`: Evaluation configuration
- `configs/test_settings/default.yaml`: Test-specific parameters

Override parameters via: `python train.py param=value` or `python train.py group=config`

### Training Workflow
1. Configuration composed via Hydra (`configs/config.yaml` + command line overrides)
2. Data module instantiated based on dataset config (MNIST or SUN397)
3. Model built from config specifying architecture type and hyperparameters
4. PyTorch Lightning trainer configured with callbacks (checkpointing, early stopping, LR monitoring)
5. Training runs with automatic checkpointing and early stopping
6. Results saved to `./runs/` directory with TensorBoard and CSV logs

### Evaluation Workflow
1. Load trained model checkpoints via test.py
2. Reconstruct encoder/decoder architecture from checkpoint state
3. Run specified tests (iterative inference, out-of-distribution)
4. Save analytics results to `./analytics/` directory

### Key Design Patterns
- **Modular architecture**: Separate encoder/decoder classes that can be mixed and matched
- **Configuration-driven**: All hyperparameters externalized to YAML files
- **PyTorch Lightning**: Clean separation of model logic from training loop
- **Hydra configuration**: Composable configs with override capabilities
- **Automatic GPU detection**: Uses GPU if available, falls back to CPU
- **Mixed precision support**: Configurable AMP training for efficiency

### Dependencies
The project requires manual installation (no requirements.txt):
- PyTorch
- PyTorch Lightning
- Torchvision
- Hydra-core
- PyYAML
- TensorBoard

### Data Support
- **MNIST**: Handwritten digits dataset (default)
- **SUN397**: Scene understanding dataset for more complex visual data

### Checkpoint Management
- Checkpoints saved to `./runs/checkpoints/` by default
- ModelCheckpoint callback saves top-k models based on validation loss
- Early stopping prevents overfitting
- Checkpoints include full model state and hyperparameters for reconstruction
