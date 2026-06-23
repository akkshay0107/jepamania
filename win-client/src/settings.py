import os
from pathlib import Path

from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_OUTPUT_DIR = BASE_DIR / "data"

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

yaml_path = BASE_DIR / "settings.yaml"
if yaml_path.exists():
    cfg = OmegaConf.load(yaml_path)
else:
    cfg = OmegaConf.create(
        {
            "hdf5_chunk_size": 128,
            "agent": {
                "policy_path": None,
                "use_noise": True,
                "noise_scale": 0.3,
                "ou_theta": 0.15,
                "ou_sigma": 0.2,
            },
            "human": {
                "record_hotkey": "F9",
            },
        }
    )

OmegaConf.set_struct(cfg, False)
cfg.compression = "gzip"
cfg.image_shape = (64, 64, 3)
cfg.action_dim = 3
cfg.data_output_dir = str(DATA_OUTPUT_DIR)
