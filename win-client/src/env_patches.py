"""Explicit environment patches for Trackmania (TMRL / rtgym) interface on Windows."""

import logging
import platform

import numpy as np


def _patched_safe_keyres() -> None:
    if platform.system() != "Windows":
        return
    import time

    import tmrl.custom.tm.utils.control_keyboard as ck  # type: ignore[import-not-found]
    import win32gui  # type: ignore[import-not-found,import-untyped]

    hwnd = win32gui.FindWindow(None, "Trackmania")
    if hwnd != 0:
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass
        time.sleep(0.05)
        if win32gui.GetForegroundWindow() == hwnd:
            if hasattr(ck, "_original_keyres"):
                ck._original_keyres()
            else:
                ck.keyres()
        else:
            logging.warning("Trackmania window not focused. Reset key skipped.")


def _collection_get_obs_rew_terminated_info(self):
    data, img = self.grab_data_and_img()
    speed = np.array([data[0]], dtype="float32")
    gear = np.array([data[9]], dtype="float32")
    rpm = np.array([data[10]], dtype="float32")

    self.img_hist.append(img)
    imgs = np.array(list(self.img_hist))
    obs = [speed, gear, rpm, imgs]

    end_of_track = bool(data[8])
    curr_pos = np.array([data[2], data[3], data[4]], dtype=np.float32)

    teleported = False
    if hasattr(self, "prev_pos") and self.prev_pos is not None:
        dist = float(np.linalg.norm(curr_pos - self.prev_pos))
        if dist > 25.0:
            teleported = True
    self.prev_pos = curr_pos

    info = {
        "action": np.array([data[5], data[6], data[7]], dtype="float32"),
        "end_of_track": end_of_track,
        "teleported": teleported,
    }

    terminated = end_of_track or teleported
    if terminated:
        self.prev_pos = None
    rew = np.float32(1.0 if end_of_track else 0.0)
    return obs, rew, terminated, info


def apply_data_collection_patches() -> None:
    """Apply TMRL patches for data collection and map-agnostic evaluation.

    Ignores TMRL's reference trajectory reward function and only marks
    an episode terminated when crossing the finish line (end_of_track)
    or when an in-game manual respawn/teleport is detected.
    """
    if platform.system() != "Windows":
        return

    try:
        import tmrl.custom.tm.utils.control_keyboard as ck  # type: ignore[import-not-found]
        from tmrl.custom.tm.tm_gym_interfaces import (  # type: ignore[import-not-found]
            TM2020Interface,
        )

        TM2020Interface.get_obs_rew_terminated_info = (
            _collection_get_obs_rew_terminated_info
        )

        if not hasattr(TM2020Interface, "_original_reset"):
            TM2020Interface._original_reset = TM2020Interface.reset

        def _patched_reset(self, *args, **kwargs):
            self.prev_pos = None
            return self._original_reset(*args, **kwargs)

        TM2020Interface.reset = _patched_reset

        if not hasattr(ck, "_original_keyres"):
            ck._original_keyres = ck.keyres
        ck.keyres = _patched_safe_keyres

        logging.info("Applied TMRL Data Collection patches cleanly.")
    except Exception as e:
        logging.warning(f"Failed to apply TMRL Data Collection patches: {e}")


def apply_online_rl_patches() -> None:
    """Apply TMRL patches for online RL training phase.

    Preserves shaped reward computation and online progress evaluation.
    """
    if platform.system() != "Windows":
        return

    try:
        import tmrl.custom.tm.utils.control_keyboard as ck  # type: ignore[import-not-found]
        from tmrl.custom.tm.tm_gym_interfaces import (  # type: ignore[import-not-found]
            TM2020Interface,
        )

        def _online_rl_get_obs_rew_terminated_info(self):
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

            info = {"action": np.array([data[5], data[6], data[7]], dtype="float32")}

            if end_of_track:
                terminated = True
                rew += self.finish_reward
            rew += self.constant_penalty
            rew = np.float32(rew)
            return obs, rew, terminated, info

        TM2020Interface.get_obs_rew_terminated_info = (
            _online_rl_get_obs_rew_terminated_info
        )

        if not hasattr(ck, "_original_keyres"):
            ck._original_keyres = ck.keyres
        ck.keyres = _patched_safe_keyres

        logging.info("Applied TMRL Online RL patches cleanly.")
    except Exception as e:
        logging.warning(f"Failed to apply TMRL Online RL patches: {e}")
