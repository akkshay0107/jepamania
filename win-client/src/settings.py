import os
from pathlib import Path

from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_OUTPUT_DIR = BASE_DIR / "data"

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

_DEFAULTS = {
    "hdf5_chunk_size": 128,
    "agent": {
        "policy_path": None,
        "map_name": "unknown",
        "map_uid": "unknown",
        "exploration": {
            "ou_noise_mu": 0.0,
            "ou_noise_theta": 0.15,
            "ou_noise_sigma": 0.05,
        },
    },
    "episode_monitor": {
        # Speed (km/h) below which the car is considered stuck.
        "stuck_speed_kmh": 5.0,
        # Consecutive frames all below stuck_speed_kmh required to trigger reset.
        "stuck_window_frames": 80,
        # Hard cap on frames per episode (~90 s at 20 Hz).
        "max_frames_per_episode": 1800,
        # Frames discarded after each reset (physics settle-in period).
        "warmup_frames": 20,
        # Number of completed episodes to record per HDF5 shard file.
        "episodes_per_shard": 50,
    },
}

yaml_path = BASE_DIR / "settings.yaml"
if yaml_path.exists():
    _yaml_cfg = OmegaConf.load(yaml_path)
    cfg = OmegaConf.merge(OmegaConf.create(_DEFAULTS), _yaml_cfg)
else:
    cfg = OmegaConf.create(_DEFAULTS)

OmegaConf.set_struct(cfg, False)

cfg.action_dim = 3  # [steer, gas, brake]
cfg.data_output_dir = str(DATA_OUTPUT_DIR)
