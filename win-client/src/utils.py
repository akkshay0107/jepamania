import platform

import numpy as np
from core.config import TELEMETRY_FEATURES

# Spatial resolution expected by the encoder.
IMG_SIZE: int = 64


def obs_to_dict(obs) -> dict[str, np.ndarray]:
    """
    Normalise the raw observation returned by the tmrl / rtgym environment
    into a dict: {screen, telemetry}, keeping only the single latest screen frame.

    The full TM20 environment (TM2020FULL) returns a tuple of two arrays:
    obs[0] — screen stack  (IMG_HIST_LEN, H, W) uint8
    obs[1] — telemetry     (TELEMETRY_FEATURES,) float32
    """
    if isinstance(obs, dict):
        screen = np.asarray(obs["screen"], dtype=np.uint8)
        if screen.ndim == 3:
            screen = screen[-1:]
        elif screen.ndim == 2:
            screen = screen[np.newaxis, ...]
        return {
            "screen": screen,
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

        # Fail loudly on layout mismatch: silently padding/truncating here
        # previously masked a wrong TELEMETRY_FEATURES constant and produced
        # shards that were mostly zero padding.
        if len(telemetry) != TELEMETRY_FEATURES:
            raise ValueError(
                f"Environment produced {len(telemetry)} telemetry floats, "
                f"expected TELEMETRY_FEATURES={TELEMETRY_FEATURES}. "
                "Check the env observation layout against core.config."
            )

        return {
            "screen": screen,
            "telemetry": telemetry,
        }

    # Shape normalization supports raw numpy arrays directly passed to the environment
    arr = np.asarray(obs)
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


# Runtime monkey-patching
# Runs at import time as a side effect
# TODO: fix this later if sticking with the monkey patch approach
# to be more explicit and not as a import side effect.
if platform.system() == "Windows":
    try:
        import time

        import tmrl.custom.tm.utils.control_keyboard as ck
        import win32gui
        from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface

        # The default TM2020Interface get_obs_rew_terminated_info discards player input.
        # Extract them here to preserve the actions required for model training.
        def patched_get_obs_rew_terminated_info(self):
            data, img = self.grab_data_and_img()
            speed = np.array([data[0]], dtype="float32")
            gear = np.array([data[9]], dtype="float32")
            rpm = np.array([data[10]], dtype="float32")
            rew, terminated = self.reward_function.compute_reward(
                pos=np.array([data[2], data[3], data[4]])
            )
            self.img_hist.append(img)
            imgs = np.array(list(self.img_hist))
            obs = [speed, gear, rpm, imgs]
            end_of_track = bool(data[8])

            info = {"action": np.array([data[6], data[7], data[5]], dtype="float32")}

            if end_of_track:
                terminated = True
                rew += self.finish_reward
            rew += self.constant_penalty
            rew = np.float32(rew)
            return obs, rew, terminated, info

        TM2020Interface.get_obs_rew_terminated_info = (
            patched_get_obs_rew_terminated_info
        )

        # SendInput sends simulated keyboard events globally
        # to the active focused window. We assert game window focus before running
        # keyres to prevent key event leaks to the console host.
        original_keyres = ck.keyres

        def patched_keyres():
            hwnd = win32gui.FindWindow(None, "Trackmania")
            if hwnd != 0:
                try:
                    win32gui.SetForegroundWindow(hwnd)
                except Exception:
                    pass
                time.sleep(0.05)
                if win32gui.GetForegroundWindow() == hwnd:
                    original_keyres()
                else:
                    import logging

                    logging.warning(
                        "Trackmania window not focused. Reset key skipped."
                    )

        ck.keyres = patched_keyres

    except Exception as e:
        import logging

        logging.warning(f"Failed to apply Windows TMRL patches: {e}")
