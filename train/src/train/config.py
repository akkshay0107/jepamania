from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, cast

from omegaconf import OmegaConf


def _contains_settings(saved: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return isinstance(saved, dict) and all(
            key in saved and _contains_settings(saved[key], value)
            for key, value in expected.items()
        )
    return saved == expected


def matches_manifest(saved: dict, expected: dict, *config_keys: str) -> bool:
    """Check run identity, ignoring older fields outside the phase's config."""
    return all(
        key in saved
        and (
            _contains_settings(saved[key], value)
            if key in config_keys
            else saved[key] == value
        )
        for key, value in expected.items()
    )


@dataclass
class LossConfig:
    num_subspaces: int = 16
    subspace_dim: Optional[int] = None  # Will default to latent_dim // num_subspaces
    num_slices: int = 256
    reg_weight: float = 0.02


@dataclass
class PretrainConfig:
    updates: int = 1536
    batch_size: int = 24
    lr_enc: float = 1e-4
    lr_pred: float = 3e-4
    rollout_len: int = 5
    seed: int = 42
    log_every: int = 250
    val_ratio: float = 0.1


@dataclass
class FinetuneConfig:
    ablation: str = "joint"
    gamma: float = 0.997
    lr_enc: float = 3e-5
    lr_pred: float = 1e-4
    lr_head: float = 3e-4
    batch_size: int = 24
    seed: int = 42
    log_every: int = 250
    total_epochs: int = 20
    exploration_epochs: int = 2
    drives_per_epoch: int = 5
    optimizer_attempts_per_epoch: int = 1000
    evaluation_interval: int | None = None
    evaluation_races: int = 5


@dataclass
class TrainConfig:
    loss: LossConfig = field(default_factory=LossConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


def load_train_config(yaml_path: str | Path | None = None) -> TrainConfig:
    """Return defaults, with overrides from a supplied YAML file."""
    config = OmegaConf.structured(TrainConfig)
    if yaml_path is not None:
        config = OmegaConf.merge(config, OmegaConf.load(yaml_path))
    return cast(TrainConfig, config)
