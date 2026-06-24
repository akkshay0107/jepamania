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
    },
    "human": {
        "record_hotkey": "F9",
    },
    "episode_monitor": {
        # Speed (km/h) below which the car is considered stuck.
        "stuck_speed_kmh": 5.0,
        # Consecutive frames all below stuck_speed_kmh required to trigger reset.
        "stuck_window_frames": 60,
        # Hard cap on frames per episode (~150 s at 20 Hz).
        "max_frames_per_episode": 3000,
        # Frames discarded after each reset (physics settle-in period).
        "warmup_frames": 20,
        # Completed episodes per terrain before rotating to the next map.
        "terrain_episode_budget": 10,
    },
    "map_cycler": {
        # Seconds to wait after firing the URI before calling env.reset().
        # Increase if your machine takes longer to load maps.
        "map_load_wait_s": 10.0,
    },
    # Path to the map registry YAML, relative to the win-client package root.
    "terrain_maps_path": "terrain_maps.yaml",
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
