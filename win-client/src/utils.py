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


class AdaptiveActionFilter:
    """
    Adaptive action filter for smoothing SAC outputs during data collection.

    Applies a deadzone around zero and an Adaptive Exponential Moving Average (EMA)
    to the steering channel (channel 0), while leaving gas (channel 1) and brake
    (channel 2) untouched for crisp pedal control.

    If filtering is disabled in config, __call__ is a no-op returning the input action.
    """

    def __init__(
        self,
        enabled: bool = False,
        steer_deadzone: float = 0.015,
        min_alpha: float = 0.5,
        max_alpha: float = 0.85,
        delta_scale: float = 0.3,
    ) -> None:
        self.enabled = bool(enabled)
        self.deadzone = float(steer_deadzone)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.alpha_range = self.max_alpha - self.min_alpha
        self.inv_delta_scale = 1.0 / float(delta_scale) if delta_scale > 0 else 1.0
        self.prev_steer = 0.0

    def reset(self) -> None:
        self.prev_steer = 0.0

    def __call__(self, action: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return action

        raw_steer = float(action[0])
        if -self.deadzone < raw_steer < self.deadzone:
            raw_steer = 0.0

        delta = abs(raw_steer - self.prev_steer)
        if delta * self.inv_delta_scale >= 1.0:
            alpha = self.max_alpha
        else:
            alpha = self.min_alpha + self.alpha_range * (delta * self.inv_delta_scale)

        smooth_steer = alpha * raw_steer + (1.0 - alpha) * self.prev_steer
        if abs(smooth_steer) < 1e-4:
            smooth_steer = 0.0

        self.prev_steer = smooth_steer

        out_action = np.copy(action)
        out_action[0] = smooth_steer
        return out_action
