# 🧠✨ Latent-Imagination: A Cognitive Modeling Of Imagery & Perception

> *"How does the brain imagine what it has never seen?"* - This is the question that keeps me up at night (and apparently fuels my thesis writing at 3 AM).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Status-Active%20Research-green.svg)](https://github.com/juliouriostegui/Latent-Imagination)
[![Thesis](https://img.shields.io/badge/Master%20Thesis-In%20Progress-purple.svg)](https://github.com/juliouriostegui/Latent-Imagination)

## 🎯 What This Is About

Welcome to my digital laboratory! This repository contains my master thesis implementation exploring how we can model the fascinating interplay between imagination and perception using deep generative models.

Think of it as teaching computers to "imagine" like humans do - filling in missing pieces, generating new combinations of familiar concepts, and navigating the blurry boundary between what's real and what's mentally constructed.

## 🚀 Tutorials

### 1) Quickstart: Training and Testing
- Install dependencies (you'll need these for the magic to happen)
  - `pip install -r requirements.txt`
- Train with defaults (MNIST + MLP VAE):
  - `python train.py`
- Train with overrides (examples):
  - `python train.py model=vae_conv dataset=mnist train.epochs=10`
  - `python train.py model=ivae_iterative train.epochs=20`
- Test/evaluate trained checkpoints:
  - `python test.py models=["path/to/model.ckpt"] tests=["iterative","ood"]`
  - Multiple named models:
    - `python test.py models=["baseline:path/to/base.ckpt","ivae:path/to/ivae.ckpt"] tests=["iterative"]`

Notes:
- Configs live in `configs/`; compose/override via Hydra (e.g., `dataset=sun`).
- Logs and checkpoints land under `runs/` by default.

```bash
# Show effective configs (train/test)
python get_cfg.py --train
python get_cfg.py --test

# Visualize encoder/decoder modules quickly
python get_model.py --tools torchinfo --model-type encoder --input-shape 1 28 28 --hidden-dims 32 64 --z-dim 32
```

## IVAEs with MLP or CNN backbones

The iterative VAE (`ivae_iterative`) now supports both MLP and CNN encoders/decoders. The architecture can be selected explicitly via config, or inferred from a checkpoint when provided.

- Explicit selection via config:
  - MLP (default): provide `encoder_hidden`/`decoder_hidden` in the model YAML.
  - CNN: set `arch: conv` and provide `conv_hidden`/`deconv_hidden` channel schedules.

- Automatic inference from checkpoint (`model.init_weights`):
  - If a checkpoint is specified, `infra/commands/train.py` inspects parameter names to choose the backbone:
    - Keys like `encoder.fc1` or `decoder.fc1` → MLP
    - Keys like `encoder.conv` or `decoder.net` → CNN
  - Hidden sizes are still taken from the YAML when present; checkpoint inference only determines the backbone type.

Examples
- MLP IVAEs (existing config):
  - `python run.py train -cn train_ivae_mlp`

- CNN IVAEs (by config): add to your IVAEs model YAML:
  - `arch: conv`
  - `conv_hidden: [32, 64]`
  - `deconv_hidden: [64, 32]`

- CNN IVAEs (inferred from checkpoint): set in the model YAML:
  - `init_weights: path/to/conv_vae.ckpt`
  - Omit `arch` to let the loader detect CNN.

Artifacts and graph visualization
- Use `get_model` to preview architectures:
  - `python run.py get_model configs/model/vae_conv.yaml --which encoder --tools torchinfo torchviz`
- `hiddenlayer` is deprecated here; requests for it are redirected to `torchviz` and saved under `runs/<which>_visuals/torchviz_<which>.png`.

## 🧬 The Science Behind the Magic

### The Big Idea

Traditional VAEs are cool, but they miss something fundamental about how humans think. We don't just passively encode/decode information - we actively **imagine**, **predict**, and **refine** our mental representations through iterative processes.

### What Makes This Special

- **Standard VAE**: Your classic encode → decode pipeline
- **Iterative VAE**: Adds a "mental refinement" step using stochastic variational inference
- **Cognitive Inspiration**: Models don't just reconstruct - they "imagine" missing details

### Model Architectures

| Model Type | Description | Use Case |
|------------|-------------|----------|
| `vae_mlp` | Fully-connected neural network | Simple experiments, quick prototyping |
| `vae_conv` | Convolutional architecture | Image data, spatial hierarchies |
| `ivae_iterative` | Semi-amortized with SVI refinement | Cognitive modeling, iterative inference |

## 📊 Experiment Configuration

All experiments are configured through beautiful YAML files.

## 🔬 Research Questions I'm Exploring

1. **How does iterative refinement affect representation quality?**
2. **Can we model the transition from perception to imagination?**
3. **What role does uncertainty play in cognitive generation?**
4. **How do different architectures affect the "cognitive plausibility" of generated samples?**

## 📈 Results & Insights

*Coming soon!* (This is where I'll share the exciting findings once my experiments converge...)

## 🛠️  Dashboard Usage (better for visualization)

```bash
- Launch TensorBoard and keep it running while you explore:
  - `python dashboard.py --launch-tensorboard --logdir runs`

- Show the exact configs the repo would use right now:
  - `python dashboard.py --print-cfg train`
  - `python dashboard.py --print-cfg test`

- CNN VAE encoder overview (MNIST-like):
  - `python dashboard.py --model-overview --tools torchinfo hiddenlayer --model-type encoder --input-shape 1 28 28 --hidden-dims 32 64 --z-dim 32`

- CNN VAE decoder overview (MNIST-like):
  - `python dashboard.py --model-overview --tools torchinfo hiddenlayer --model-type decoder --input-shape 1 28 28 --hidden-dims 64 32 --z-dim 32`

- Whole-model visualization (graph):
  - The training loop already logs a graph to TensorBoard for inspection. Start TensorBoard with the dashboard, then run a short train to materialize the graph:
  - `python dashboard.py --launch-tensorboard --logdir runs`
  - `python train.py train.epochs=1`
  - Open the printed URL and check the Graphs tab.

- IVAE (iterative) model overview:
  - For now, use the TensorBoard graph from a short training run as above. If you want encoder/decoder-only views for IVAE, we can extend the dashboard to support MLP encoders/decoders similarly to the CNN ones.

- Hyperparameter tests (quick sequential trials):
  - `python dashboard.py --run-sweep "train.epochs=2 model.beta=0.5::train.epochs=2 model.beta=2.0"`

- Visualize current train/test parameters inside the dashboard:
  - Use the `--print-cfg` switch to print the effective Hydra config with all defaults and overrides resolved:
  - `python dashboard.py --print-cfg train`
  - `python dashboard.py --print-cfg test`

- Launch TensorBoard in a notebook (Jupyter/Colab):
  - `%load_ext tensorboard`
  - `%tensorboard --logdir runs`

Note: The dashboard’s model overview currently targets CNN encoder/decoder blocks; whole-model graphs are best viewed in TensorBoard. If you’d like, we can add support for MLP/IVAE module overviews in `get_model.py` and expose them via the dashboard.

### 3) Configs and Model Tools
- Show effective configs (train/test):
  - `python get_cfg.py --train`
  - `python get_cfg.py --test`
- Quick CNN encoder/decoder visualization (text summary/graph):
  - `python get_model.py --tools torchinfo --model-type encoder --input-shape 1 28 28 --hidden-dims 32 64 --z-dim 32`
  - `python get_model.py --tools hiddenlayer --model-type decoder --input-shape 1 28 28 --hidden-dims 64 32 --z-dim 32`

## 🗂️ Repository Structure
- research/ — research core
  - research/models — PyTorch/ML models (VAE, IVAE)
  - research/analysis — analytics, plots, evaluation helpers
  - research/tools — research-side tools (e.g., losses)
  - research/notebooks — Jupyter/Colab notebooks
  - research/experiments — experiment scripts
  - research/data — optional preprocessed/synthetic data or pointers
- infra/ — engine room
  - infra/utils — shared non-research utilities (I/O, dataloaders)
  - infra/pipelines — training/eval/data pipelines (e.g., train_engine)
  - infra/logging — logging helpers (TensorBoard, custom)
  - infra/visualization — visualization helpers
- configs/ — Hydra configuration (kept at root)
- train.py — training CLI (kept at root)
- test.py — evaluation CLI (kept at root)
- get_cfg.py — print effective configs
- get_model.py — quick CNN encoder/decoder visualizations
- dashboard.py — launch TensorBoard, print configs, overviews, sweeps

Import paths after reorg
- Models: `from research.models import BaseVAE, ConvEncoder, ...`
- Utilities: `from infra.utils import GenericImageDataModule, ...`
- Analytics: `from research.analysis import run_iterative_inference_test, ...`
- Losses/ELBO: `from research.tools.losses import elbo_per_sample, ...`

## 📓 Colab Magic Commands
- Quick dashboard in Colab with background TensorBoard and live training:
  - `!python dashboard.py --launch-tensorboard --logdir /content/runs --port 6006`
  - `!python train.py`
  - Then open the printed URL or use: `%load_ext tensorboard` and `%tensorboard --logdir /content/runs`

- Print effective configs directly in a notebook cell:
  - `!python dashboard.py --print-cfg train`
  - `!python dashboard.py --print-cfg test`

- Run a quick model overview (CNN encoder example):
  - `!python dashboard.py --model-overview --tools torchinfo hiddenlayer --model-type encoder --input-shape 1 28 28 --hidden-dims 32 64 --z-dim 32`
```

## 📚 Citation & Academic Context

If you use this work in your research (or just think it's cool), here's how to cite it:

```bibtex
@misc{Latent-Imagination-2025,
  title={Latent-Imagination: A Cognitive Modeling Of Imagery & Perception},
  author={Julio Uriostegui},
  year={2025},
  howpublished={\url{https://github.com/JulioUriostegui19/Latent-Imagination}}
}
```

## 🤝 Contributing

This is my thesis work, but I'm always open to:

- 🐛 Bug reports (please!)
- 💡 Suggestions for experiments
- 🧠 Cognitive science insights

## 🎓 About the Researcher

Hi! I'm Julio, a master's student diving deep into the intersection of **deep learning** and **cognitive science**. When I'm not training neural networks, I'm probably:

- Playing Catan
- Explaining to my family that no, I'm not "just playing with computers"
- Wondering if my brain is just a really complex VAE

**Current status**: *"It's not procrastination if it's literature review"* 📖

## 🙏 Acknowledgments

- My advisor (for keeping me scientifically honest)
- The open-source community (for making research accessible)

---

## 📬 Get in Touch

- 🔬 [Website](https://juliouriostegui.com)

*"Our future is the result of collaboration."*

---

**⭐ Star this repo if you think cognitive AI is the future!**
