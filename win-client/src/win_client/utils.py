import platform

import numpy as np
from core.config import IMG_SIZE, TELEMETRY_FEATURES


def obs_to_dict(obs) -> dict[str, np.ndarray]:
    """Read one grayscale frame and telemetry from TMRL or a simulator."""
    if isinstance(obs, dict):
        screen = np.asarray(obs["screen"], dtype=np.uint8)
        telemetry = np.asarray(obs["telemetry"], dtype=np.float32).reshape(-1)
        if telemetry.size == 3:
            telemetry = np.pad(telemetry, (0, TELEMETRY_FEATURES - 3))
    elif isinstance(obs, (tuple, list)):
        parts = [np.asarray(item) for item in obs]
        image = next(
            (
                index
                for index, part in enumerate(parts)
                if part.shape[-2:] == (IMG_SIZE, IMG_SIZE)
            ),
            None,
        )
        if image is None:
            raise ValueError("TMRL observation is missing a grayscale frame.")
        screen = parts.pop(image).astype(np.uint8, copy=False)
        telemetry = (
            np.concatenate([part.reshape(-1) for part in parts], dtype=np.float32)
            if parts
            else np.empty(0, dtype=np.float32)
        )
    else:
        raise ValueError("Expected a TMRL observation or simulation dictionary.")

    if screen.ndim == 3:
        screen = screen[-1:]
    elif screen.ndim == 2:
        screen = screen[None]
    if screen.shape != (1, IMG_SIZE, IMG_SIZE):
        raise ValueError("The game must return a grayscale 64×64 observation.")
    if telemetry.size != TELEMETRY_FEATURES:
        raise ValueError(
            f"Expected {TELEMETRY_FEATURES} telemetry values including action history."
        )
    return {"screen": screen, "telemetry": telemetry}


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
    from win_client.env_patches import configure_rtgym_action_history

    configure_rtgym_action_history()
    import tmrl  # type: ignore[import-not-found]

    env = tmrl.get_environment()
    base_env = env.unwrapped
    if (
        not getattr(base_env, "act_in_obs", False)
        or getattr(base_env, "act_buf_len", 0) != 2
    ):
        raise RuntimeError(
            "TMRL environment was created without the required two-action "
            "history in observations."
        )
    return env


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
