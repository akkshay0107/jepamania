import platform

import numpy as np
from core.config import TELEMETRY_FEATURES

# Spatial resolution expected by the encoder.
IMG_SIZE: int = 64


def obs_to_dict(obs, obs_type: str = "screen") -> dict[str, np.ndarray]:
    """
    Normalise the raw observation returned by the tmrl / rtgym environment
    into a dict: {screen, telemetry} or {lidar, telemetry}.
    """
    if obs_type not in ("screen", "lidar"):
        raise ValueError("obs_type must be either 'screen' or 'lidar'.")

    if isinstance(obs, dict):
        telemetry = np.asarray(obs["telemetry"], dtype=np.float32)
        if len(telemetry) < TELEMETRY_FEATURES:
            pad_len = TELEMETRY_FEATURES - len(telemetry)
            telemetry = np.pad(telemetry, (0, pad_len), mode="constant")
        elif len(telemetry) > TELEMETRY_FEATURES:
            raise ValueError(
                f"Environment produced {len(telemetry)} telemetry floats, "
                f"expected at most TELEMETRY_FEATURES={TELEMETRY_FEATURES}."
            )
        if obs_type == "lidar":
            lidar = np.asarray(obs["lidar"], dtype=np.float32)
            if lidar.ndim == 2:
                lidar = lidar[-1:]
            elif lidar.ndim == 1:
                lidar = lidar[np.newaxis, ...]
            return {"lidar": lidar, "telemetry": telemetry}
        else:
            screen = np.asarray(obs["screen"], dtype=np.uint8)
            if screen.ndim == 3:
                screen = screen[-1:]
            elif screen.ndim == 2:
                screen = screen[np.newaxis, ...]
            return {"screen": screen, "telemetry": telemetry}

    if isinstance(obs, (tuple, list)) and len(obs) >= 1:
        lidar = None
        screen = None
        if obs_type == "lidar":
            telem_parts = []
            for item in obs:
                arr = np.asarray(item)
                if lidar is None and (arr.ndim == 2 or arr.size >= 19):
                    lidar = arr.astype(np.float32)
                else:
                    telem_parts.append(arr.flatten().astype(np.float32))

            if lidar is None:
                from core.config import LIDAR_BEAMS

                lidar = np.zeros((1, LIDAR_BEAMS), dtype=np.float32)
            else:
                if lidar.ndim == 2:
                    lidar = lidar[-1:]
                elif lidar.ndim == 1:
                    lidar = lidar[np.newaxis, ...]
        else:
            telem_parts = []
            for item in obs:
                arr = np.asarray(item)
                if screen is None and (arr.ndim >= 3 or arr.dtype == np.uint8):
                    screen = arr.astype(np.uint8)
                else:
                    telem_parts.append(arr.flatten().astype(np.float32))

            if screen is None:
                screen = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            else:
                if screen.ndim == 3:
                    screen = screen[-1:]
                elif screen.ndim == 2:
                    screen = screen[np.newaxis, ...]

        if telem_parts:
            telemetry = np.concatenate(telem_parts, axis=0)
        else:
            telemetry = np.zeros(TELEMETRY_FEATURES, dtype=np.float32)

        if len(telemetry) < TELEMETRY_FEATURES:
            pad_len = TELEMETRY_FEATURES - len(telemetry)
            telemetry = np.pad(telemetry, (0, pad_len), mode="constant")
        elif len(telemetry) > TELEMETRY_FEATURES:
            raise ValueError(
                f"Environment produced {len(telemetry)} telemetry floats, "
                f"expected at most TELEMETRY_FEATURES={TELEMETRY_FEATURES}. "
                "Check the env observation layout against core.config."
            )

        if obs_type == "lidar":
            assert lidar is not None
            return {"lidar": lidar, "telemetry": telemetry}
        assert screen is not None
        return {"screen": screen, "telemetry": telemetry}

    arr = np.asarray(obs)
    if obs_type == "lidar":
        from core.config import LIDAR_BEAMS

        if arr.ndim >= 2:
            lidar = arr[-1:].astype(np.float32)
        elif arr.ndim == 1:
            lidar = arr[np.newaxis, ...].astype(np.float32)
        else:
            lidar = np.zeros((1, LIDAR_BEAMS), dtype=np.float32)
        return {
            "lidar": lidar,
            "telemetry": np.zeros(TELEMETRY_FEATURES, dtype=np.float32),
        }

    if arr.ndim >= 3:
        screen = arr[-1:].astype(np.uint8)
    elif arr.ndim == 2:
        screen = arr[np.newaxis, ...].astype(np.uint8)
    else:
        screen = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

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
    to the steering channel (channel 2 in TMRL), leaving gas / brake untouched.

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

        raw_steer = float(action[2])
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
        out_action[2] = smooth_steer
        return out_action


def get_tmrl_env():
    """Return the TMRL environment if running on Windows."""
    if platform.system() != "Windows":
        raise RuntimeError(
            "TMRL gymnasium environment can only be instantiated on Windows systems."
        )
    import tmrl  # type: ignore[import-not-found]

    return tmrl.get_environment()


def get_tmrl_obs_preprocessor():
    """Return the default TMRL observation preprocessor if running on Windows."""
    if platform.system() != "Windows":
        raise RuntimeError(
            "TMRL observation preprocessor can only be accessed on Windows systems."
        )
    import tmrl.config.config_objects as cfg_obj  # type: ignore[import-not-found]

    return cfg_obj.OBS_PREPROCESSOR


def get_tmrl_policy_class():
    """Return the default TMRL POLICY class if running on Windows."""
    if platform.system() != "Windows":
        raise RuntimeError("TMRL policy class can only be accessed on Windows systems.")
    import tmrl.config.config_objects as cfg_obj  # type: ignore[import-not-found]

    return cfg_obj.POLICY
