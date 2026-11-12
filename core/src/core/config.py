from dataclasses import dataclass, field
from typing import cast

from omegaconf import OmegaConf

# Fixed TMRL Constants (TM2020FULL)
IMG_HIST_LEN: int = 4
IMG_SIZE: int = 64
PIXEL_SCALE: int = 255
TELEMETRY_SCALE: tuple[float, float, float] = (100.0, 6.0, 10000.0)
# The TMRL Full env observation carries exactly 9 telemetry floats:
# speed, gear, rpm, plus rtgym's two-action chronological buffer (3 floats
# each). The older buffered action comes first, followed by the most recent.
# (The Openplanet plugin sends 11 floats; distance and position feed the
# reward function only and never reach the observation.)
TELEMETRY_FEATURES: int = 9
ACTION_HISTORY_FEATURES: int = 6

# Gas / Brake -> discretized into 6 bins
# Steering [-1.0, 1.0] -> discretized into 9 bins
# so 54 distinct actions possible at any given moment of time
NUM_ACTIONS: int = 54
# Learned width; model states append the six known action-history values.
LATENT_DIM: int = 128

# The legacy recorder marks failed collection episodes for inspection.
FAILURE_PENALTY: float = -100.0


@dataclass
class TransformerConfig:
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0


@dataclass
class EncoderConfig:
    latent_dim: int = LATENT_DIM
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


@dataclass
class PredictorConfig:
    latent_dim: int = LATENT_DIM
    hidden_dim: int = 128


@dataclass
class PlannerConfig:
    sequence_len: int = 5
    cem_iters: int = 4
    cem_samples: int = 128
    cem_elites: int = 16
    cem_alpha: float = 0.7


@dataclass
class SubJepaConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)


def load_config(yaml_path: str | None = None) -> SubJepaConfig:
    """Return defaults, with overrides from a supplied YAML file."""
    config = OmegaConf.structured(SubJepaConfig)
    if yaml_path is not None:
        config = OmegaConf.merge(config, OmegaConf.load(yaml_path))
    return cast(SubJepaConfig, config)
