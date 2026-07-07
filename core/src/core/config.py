from dataclasses import dataclass, field
from typing import Optional

from omegaconf import OmegaConf

# Fixed TMRL Constants (TM2020FULL)
IMG_HIST_LEN: int = 4
# The TMRL Full env observation carries exactly 9 telemetry floats:
# speed, gear, rpm, plus the two previous actions (3 floats each).
# (The Openplanet plugin sends 11 floats; distance and position feed the
# reward function only and never reach the observation.)
TELEMETRY_FEATURES: int = 9

# Fixed TMRL Constants (TM2020LIDAR)
LIDAR_BEAMS: int = 19

# Steering [-1.0, 1.0] -> discretized into 7 bins
# Gas / Brake -> discretized into 5 bins
# so 35 distinct actions possible at any given moment of time
NUM_ACTIONS: int = 35


@dataclass
class TransformerConfig:
    num_layers: int = 3
    num_heads: int = 4
    mlp_ratio: float = 4.0


@dataclass
class EncoderConfig:
    latent_dim: int = 192
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


@dataclass
class PredictorConfig:
    latent_dim: int = 192
    action_embed_dim: int = 16
    hidden_dim: int = 256


@dataclass
class ValueHeadConfig:
    latent_dim: int = 192
    hidden_dim: int = 256


@dataclass
class LossConfig:
    num_subspaces: int = 16
    subspace_dim: Optional[int] = None  # Will default to latent_dim // num_subspaces
    num_slices: int = 16
    reg_weight: float = 1.0


@dataclass
class SubJepaConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def load_config(yaml_path: str = "config.yaml") -> SubJepaConfig:
    """
    Loads configuration from a YAML file, falling back to structured defaults
    for any missing fields or if the file is not found.
    """
    base_cfg = OmegaConf.structured(SubJepaConfig)
    try:
        yaml_cfg = OmegaConf.load(yaml_path)
        return OmegaConf.merge(base_cfg, yaml_cfg)  # type: ignore
    except FileNotFoundError:
        return base_cfg  # type: ignore
