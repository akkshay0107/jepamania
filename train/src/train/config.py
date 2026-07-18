from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class LossConfig:
    num_subspaces: int = 16
    subspace_dim: Optional[int] = None  # Will default to latent_dim // num_subspaces
    num_slices: int = 16
    reg_weight: float = 0.02


@dataclass
class PretrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    rollout_len: int = 5
    num_workers: int = 4
    seed: int = 42
    log_every: int = 50
    val_ratio: float = 0.1
    max_cache_gb: float = 4.0


@dataclass
class FinetuneConfig:
    warmup_epochs: int = 3
    joint_epochs: int = 5
    gamma: float = 0.990
    lr_warmup: float = 5e-4
    lr_enc: float = 1e-5
    lr_val: float = 3e-4
    value_weight: float = 0.5
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    rollout_len: int = 5
    log_every: int = 50
    max_cache_gb: float = 4.0
    importance_max_episodes: int = 32
    importance_recency_decay: float = 0.95


@dataclass
class TrainConfig:
    loss: LossConfig = field(default_factory=LossConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


def load_train_config(yaml_path: str | Path = "train/config.yaml") -> TrainConfig:
    """
    Loads training configuration from a YAML file, falling back to structured defaults
    for any missing fields or if the file is not found.
    """
    base_cfg = OmegaConf.structured(TrainConfig)
    try:
        yaml_cfg = OmegaConf.load(str(yaml_path))
        return OmegaConf.merge(base_cfg, yaml_cfg)  # type: ignore
    except FileNotFoundError:
        return base_cfg  # type: ignore
