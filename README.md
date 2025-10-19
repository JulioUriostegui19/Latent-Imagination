# 🧠✨ Latent-Imagination: A Cognitive Modeling Of Imagery & Perception

> *"How does the brain imagine what it has never seen?"* - This is the question that keeps me up at night (and apparently fuels my thesis writing at 3 AM).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Status-Active%20Research-green.svg)](https://github.com/juliouriostegui/cognitive-vae)
[![Thesis](https://img.shields.io/badge/Master%20Thesis-In%20Progress-purple.svg)]()

## 🎯 What This Is About

Welcome to my digital laboratory where **Variational Autoencoders** meet **cognitive neuroscience**! This repository contains my master thesis implementation exploring how we can model the fascinating interplay between imagination and perception using deep generative models.

Think of it as teaching computers to "imagine" like humans do - filling in missing pieces, generating new combinations of familiar concepts, and navigating the blurry boundary between what's real and what's mentally constructed.

## 🚀 Quick Start

```bash
# Clone this repository (and my dreams along with it)
git clone https://github.com/juliouriostegui/cognitive-vae.git
cd cognitive-vae

# Install dependencies (you'll need these for the magic to happen)
pip install torch pytorch-lightning torchvision pyyaml tensorboard

# Train a model on MNIST (because we all start with handwritten digits)
python train.py

# Watch the magic happen in real-time
tensorboard --logdir ./runs
```

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

All experiments are configured through beautiful YAML files (because researchers love their config files):

```yaml
# configs/my_experiment.yaml
dataset:
  name: mnist
  batch_size: 256

model:
  type: ivae_iterative  # The cognitive one!
  z_dim: 20            # How many dimensions for our mental space?
  lr: 1e-3
  beta: 2.0            # KL divergence weight (the balancing act)

train:
  epochs: 100
  early_stopping: true # Stop when imagination gets too wild
```

## 🔬 Research Questions I'm Exploring

1. **How does iterative refinement affect representation quality?**
2. **Can we model the transition from perception to imagination?**
3. **What role does uncertainty play in cognitive generation?**
4. **How do different architectures affect the "cognitive plausibility" of generated samples?**

## 📈 Results & Insights

*Coming soon!* (This is where I'll share the exciting findings once my experiments converge...)

## 🛠️ Development Setup

```bash
# Create a virtual environment (because hygiene matters)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run experiments with different configs
python train.py --config configs/experiment1.yaml
python train.py --config configs/experiment2.yaml
```

## 📚 Citation & Academic Context

If you use this work in your research (or just think it's cool), here's how to cite it:

```bibtex
@misc{cognitive-vae-2024,
  title={Cognitive Modeling of Imagination and Perception using Variational Autoencoders},
  author={Julio Uriostegui},
  year={2024},
  howpublished={\url{https://github.com/juliouriostegui/cognitive-vae}}
}
```

## 🤝 Contributing

This is my thesis work, but I'm always open to:
- 🐛 Bug reports (please!)
- 💡 Suggestions for experiments
- 🧠 Cognitive science insights
- ☕ Coffee recommendations for late-night coding sessions

## 🎓 About the Researcher

Hi! I'm Julio, a master's student diving deep into the intersection of **deep learning** and **cognitive science**. When I'm not training neural networks, I'm probably:
- Reading papers about predictive processing
- Wondering if my brain is just a really complex VAE
- Explaining to my family that no, I'm not "just playing with computers"

**Current status**: *"It's not procrastination if it's literature review"* 📖

## 🙏 Acknowledgments

- My advisor (for keeping me scientifically honest)
- PyTorch Lightning (for saving me from boilerplate hell)
- Coffee (for obvious reasons)
- The open-source community (for making research accessible)

---

## 📬 Get in Touch


- 🔬 [Website](https://juliouriostegui.com)

*"The mind is not a vessel to be filled, but a fire to be kindled"* - Plutarch (probably talking about neural networks)

---

**⭐ Star this repo if you think cognitive AI is the future!**
