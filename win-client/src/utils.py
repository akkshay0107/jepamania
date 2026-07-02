import numpy as np
from core.config import IMG_HIST_LEN, TELEMETRY_FEATURES

# Spatial resolution expected by the encoder.
IMG_SIZE: int = 64


def obs_to_dict(obs) -> dict[str, np.ndarray]:
    """
    Normalise the raw observation returned by the tmrl / rtgym environment
    into a dict: {screen, telemetry}.

    The full TM20 environment (TM2020FULL) returns a tuple of two arrays:
    obs[0] — screen stack  (IMG_HIST_LEN, H, W) uint8
    obs[1] — telemetry     (TELEMETRY_FEATURES,) float32
    """
    if isinstance(obs, dict):
        return {
            "screen": np.asarray(obs["screen"], dtype=np.uint8),
            "telemetry": np.asarray(obs["telemetry"], dtype=np.float32),
        }

    if isinstance(obs, (tuple, list)) and len(obs) >= 1:
        screen = None
        telem_parts = []
        for item in obs:
            arr = np.asarray(item)
            if screen is None and (arr.ndim >= 3 or arr.dtype == np.uint8):
                screen = arr.astype(np.uint8)
            else:
                telem_parts.append(arr.flatten().astype(np.float32))

        if screen is None:
            screen = np.zeros((IMG_HIST_LEN, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        if telem_parts:
            telemetry = np.concatenate(telem_parts, axis=0)
        else:
            telemetry = np.zeros(TELEMETRY_FEATURES, dtype=np.float32)

        # Ensure exact telemetry feature size
        if len(telemetry) < TELEMETRY_FEATURES:
            telemetry = np.pad(telemetry, (0, TELEMETRY_FEATURES - len(telemetry)))
        elif len(telemetry) > TELEMETRY_FEATURES:
            telemetry = telemetry[:TELEMETRY_FEATURES]

        return {
            "screen": screen,
            "telemetry": telemetry,
        }

    # fallback: single image array with zero-padded telemetry.
    arr = np.asarray(obs)
    if arr.ndim >= 3 and arr.shape[0] == IMG_HIST_LEN:
        screen = arr.astype(np.uint8)
    else:
        screen = np.zeros((IMG_HIST_LEN, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    return {
        "screen": screen,
        "telemetry": np.zeros(TELEMETRY_FEATURES, dtype=np.float32),
    }


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""

    def __init__(
        self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.05
    ):
        self.mu = mu * np.ones(size, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self) -> None:
        self.state = np.copy(self.mu)

    def __call__(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state.astype(np.float32)
