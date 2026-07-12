from dataclasses import dataclass, field

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

# Gas / Brake -> discretized into 6 bins
# Steering [-1.0, 1.0] -> discretized into 9 bins
# so 54 distinct actions possible at any given moment of time
NUM_ACTIONS: int = 54
LATENT_DIM: int = 192

# In TMRL Trackmania (20 FPS), max progress reward per step is ~10.0 at top speed.
# Fixed terminal penalty for getting stuck or exceeding max frames is 10x in negative.
MAX_STEP_REWARD: float = 10.0
FAILURE_PENALTY: float = -100.0


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
class PlannerConfig:
    type: str = "beam"  # "beam" | "cem" | "random"
    sequence_len: int = 10
    smoothness_weight: float = 0.5
    # Beam Search
    beam_width: int = 5
    # Cross-Entropy Method (CEM)
    cem_iters: int = 3
    cem_samples: int = 100
    cem_elites: int = 25
    cem_alpha: float = 0.1
    # Random Shooting
    rs_samples: int = 500


@dataclass
class SubJepaConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)


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
