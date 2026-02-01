"""Default training configuration grouped by category."""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class IOConfig:
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"  # "scratch" | "resume"


@dataclass
class WandbConfig:
    wandb_log: bool = False
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"


@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    block_size: int = 1024


@dataclass
class OptimizerConfig:
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5


@dataclass
class SystemConfig:
    backend: str = "nccl"
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True


def get_default_config() -> Dict[str, Any]:
    """Return defaults grouped by category."""
    return {
        "io": asdict(IOConfig()),
        "wandb": asdict(WandbConfig()),
        "data": asdict(DataConfig()),
        "optimizer": asdict(OptimizerConfig()),
        "system": asdict(SystemConfig()),
    }
