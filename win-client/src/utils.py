import numpy as np
from core.config import IMG_HIST_LEN, LIDAR_FEATURES, TELEMETRY_FEATURES

# Spatial resolution expected by the encoder.
IMG_SIZE: int = 64


def obs_to_dict(obs) -> dict[str, np.ndarray]:
    """
    Normalise the raw observation returned by the tmrl / rtgym environment
    into a dict: {screen, lidar, telemetry}.

    The full TM20 environment (TM20FULL) returns a tuple of three arrays:
    obs[0] — screen stack  (IMG_HIST_LEN, H, W) uint8
    obs[1] — lidar stack   (IMG_HIST_LEN, LIDAR_FEATURES) float32
    obs[2] — telemetry     (TELEMETRY_FEATURES,) float32
    """
    if isinstance(obs, dict):
        return {
            "screen": np.asarray(obs["screen"], dtype=np.uint8),
            "lidar": np.asarray(obs["lidar"], dtype=np.float32),
            "telemetry": np.asarray(obs["telemetry"], dtype=np.float32),
        }

    if isinstance(obs, (tuple, list)) and len(obs) >= 3:
        return {
            "screen": np.asarray(obs[0], dtype=np.uint8),
            "lidar": np.asarray(obs[1], dtype=np.float32),
            "telemetry": np.asarray(obs[2], dtype=np.float32),
        }

    # fallback: single image array with zero-padded extras.
    arr = np.asarray(obs)
    if arr.ndim == 3 and arr.shape[0] == IMG_HIST_LEN:
        screen = arr.astype(np.uint8)
    else:
        screen = np.zeros((IMG_HIST_LEN, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    return {
        "screen": screen,
        "lidar": np.zeros((IMG_HIST_LEN, LIDAR_FEATURES), dtype=np.float32),
        "telemetry": np.zeros(TELEMETRY_FEATURES, dtype=np.float32),
    }
