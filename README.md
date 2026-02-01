# nanoFLA

A minimal and hackable training framework for comparing linear attention models (DeltaNet variants) with GPT-2, built on top of [nanoGPT](https://github.com/karpathy/nanoGPT).

## Overview

This repository provides a clean implementation for training and comparing different language model architectures:

- **GPT-2**: Standard transformer with self-attention
- **DeltaNet**: Linear attention using delta rule
- **OMD DeltaNet**: Online Mirror Descent variant of DeltaNet
- **Conceptual DeltaNet**: Conceptual implementation of DeltaNet

The codebase is designed to be simple, readable, and easy to hack for research purposes.

## Features

- 🚀 Clean, modular architecture (~300 lines training loop)
- 🔧 Easy to add new model architectures
- 📊 Built-in support for distributed training (DDP)
- 📈 Weights & Biases integration for experiment tracking
- 🎯 Pre-configured training recipes for Shakespeare and OpenWebText
- 🔬 Jupyter notebooks for scaling laws and transformer sizing

## Installation

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

**Dependencies:**
- pytorch
- numpy
- tiktoken (for fast BPE tokenization)
- wandb (for optional logging)
- tqdm (for progress bars)

## Quick Start

### 1. Prepare Data

**Shakespeare character-level:**
```bash
python data/shakespeare_char/prepare.py
```

**Shakespeare word-level:**
```bash
python data/shakespeare/prepare.py
```

**OpenWebText:**
```bash
python data/openwebtext/prepare.py
```

### 2. Train Models

**Train GPT-2 on Shakespeare:**
```bash
python train.py config/train_shakespeare_gpt2.py
```

**Train DeltaNet on Shakespeare:**
```bash
python train.py config/train_shakespeare_deltanet.py
```

**Train Conceptual DeltaNet on Shakespeare:**
```bash
python train.py config/train_shakespeare_conceptual_deltanet.py
```

**Train OMD DeltaNet on Shakespeare:**
```bash
python train.py config/train_shakespeare_omd_deltanet.py
```

<!-- ### 3. Sample from Trained Models

```bash
python sample.py --model_name=gpt2 --ckpt_path=out-shakespeare/ckpt_gpt2.pt
python sample.py --model_name=deltanet --ckpt_path=out-shakespeare/ckpt_deltanet.pt
``` -->

## Project Structure

```
.
├── train.py                    # Main training script (~350 lines)
├── sample.py                   # Sampling/generation script
├── configurator.py             # Config override system
├── model_list.py               # Model registry
│
├── config/                     # Training configurations
│   ├── train_shakespeare_gpt2.py
│   ├── train_shakespeare_deltanet.py
│   ├── train_shakespeare_conceptual_deltanet.py
│   └── train_shakespeare_omd_deltanet.py
│
├── models/                     # Model implementations
│   ├── gpt2/                  # Standard GPT-2
│   ├── deltanet/              # DeltaNet
│   ├── omd_deltanet/          # OMD DeltaNet
│   └── conceptual_deltanet/   # Conceptual DeltaNet
│
├── data/                       # Data loaders and preparation
│   ├── dataloader.py
│   ├── shakespeare/
│   ├── shakespeare_char/
│   └── openwebtext/
│
├── template/                   # Reusable components
│   ├── delta_rule.py
│   ├── mlp.py
│   ├── rotary.py
│   └── shortconvolution.py
│
└── utils/                      # Utilities
    └── config.py              # Configuration dataclasses
```

<!-- ## Configuration System

Training configurations use Python dataclasses for type safety:

```python
from utils.config import *
from models import *

# I/O settings
io_config = IOConfig(
    out_dir='out-shakespeare',
    eval_interval=50,
    log_interval=5,
)

# Model configuration
model_config_instance = DeltaNetConfig(
    model_name='deltanet',
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
)

# Optimizer settings
optimizer_config = OptimizerConfig(
    learning_rate=5e-4,
    max_iters=300,
)
``` -->

## Distributed Training

Run on multiple GPUs with DDP:

```bash
# Single node, 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py config/train_shakespeare_gpt2.py

# Multi-node training
# On master node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=123.456.123.456 --master_port=1234 \
  train.py config/train_gpt2.py

# On worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=123.456.123.456 --master_port=1234 \
  train.py config/train_gpt2.py
```

## Adding New Models

1. Create a new directory in `models/`:
   ```
   models/
   └── your_model/
       ├── __init__.py
       ├── configuration_your_model.py
       ├── model_your_model.py
       └── layer_your_model.py
   ```

2. Implement your model configuration and architecture

3. Register in `model_list.py`:
   ```python
   from models.your_model import YourModelLM
   
   model_map = {
       "your_model": YourModelLM,
       # ...
   }
   ```

4. Create a training config in `config/train_your_model.py`

## Notebooks

- `scaling_laws.ipynb`: Analyze scaling behavior of different architectures
- `transformer_sizing.ipynb`: Tools for sizing transformer models

## Experiments

All training runs are logged to `wandb/` and checkpoints are saved to `out_dir` specified in the config files:

- `ckpt_gpt2.pt`
- `ckpt_deltanet.pt`
- `ckpt_conceptual_deltanet.pt`
- `ckpt_omd_deltanet.pt`

## License

See [LICENSE](LICENSE)

## Acknowledgments

This project is built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. The DeltaNet implementations are based on recent research in linear attention mechanisms.

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@software{nanofla2026,
  title = {nanoFLA: Minimal Linear Attention Training Framework},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/nanofla}
}
``` -->
