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


def _monitor_step(
    self, speed_val: float, end_of_track: bool, teleported: bool, base_rew: float
):
    from core.config import FAILURE_PENALTY

    from win_client.settings import cfg

    if not hasattr(self, "_ep_frame_count"):
        self._ep_frame_count = 0
    if not hasattr(self, "_stuck_counter"):
        self._stuck_counter = 0
    if not hasattr(self, "_warmup_counter"):
        self._warmup_counter = 0

    self._ep_frame_count += 1
    if self._warmup_counter < cfg.episode_monitor.warmup_frames:
        self._warmup_counter += 1

    reason = None
    if end_of_track:
        reason = "done"
    elif teleported:
        reason = "respawn"
    elif self._warmup_counter >= cfg.episode_monitor.warmup_frames:
        if speed_val < cfg.episode_monitor.stuck_speed_kmh:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        if self._stuck_counter >= cfg.episode_monitor.stuck_window_frames:
            reason = "stuck"
        elif self._ep_frame_count >= cfg.episode_monitor.max_frames_per_episode:
            reason = "frame_budget"

    terminated = reason is not None
    if terminated:
        self.prev_pos = None
        self._ep_frame_count = 0
        self._stuck_counter = 0
        self._warmup_counter = 0

    if reason == "done":
        rew = base_rew + getattr(self, "finish_reward", 0.0)
    elif reason in ("stuck", "frame_budget", "respawn"):
        rew = FAILURE_PENALTY
    else:
        rew = base_rew + getattr(self, "constant_penalty", 0.0)

    rew = np.float32(rew)
    return rew, terminated, reason


def _collection_get_obs_rew_terminated_info(self):
    data, img = self.grab_data_and_img()
    speed_val = float(data[0])
    speed = np.array([speed_val], dtype="float32")
    gear = np.array([data[9]], dtype="float32")
    rpm = np.array([data[10]], dtype="float32")

    curr_pos = np.array([data[2], data[3], data[4]], dtype=np.float32)
    _rew, _tmrl_terminated = self.reward_function.compute_reward(pos=curr_pos)

    self.img_hist.append(img)
    imgs = np.array(list(self.img_hist))
    obs = [speed, gear, rpm, imgs]

    end_of_track = bool(data[8])

    teleported = False
    if hasattr(self, "prev_pos") and self.prev_pos is not None:
        dist = float(np.linalg.norm(curr_pos - self.prev_pos))
        if dist > 25.0:
            teleported = True
    self.prev_pos = curr_pos

    rew, terminated, reason = _monitor_step(
        self, speed_val, end_of_track, teleported, base_rew=0.0
    )

    info = {
        "action": np.array([data[5], data[6], data[7]], dtype="float32"),
        "end_of_track": end_of_track,
        "teleported": teleported,
        "termination_reason": reason,
    }
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
            self._ep_frame_count = 0
            self._stuck_counter = 0
            self._warmup_counter = 0
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
            speed_val = float(data[0])
            speed = np.array([speed_val], dtype="float32")
            gear = np.array([data[9]], dtype="float32")
            rpm = np.array([data[10]], dtype="float32")

            curr_pos = np.array([data[2], data[3], data[4]], dtype=np.float32)
            base_rew, _ = self.reward_function.compute_reward(pos=curr_pos)

            self.img_hist.append(img)
            imgs = np.array(list(self.img_hist))
            obs = [speed, gear, rpm, imgs]

            end_of_track = bool(data[8])

            teleported = False
            if hasattr(self, "prev_pos") and self.prev_pos is not None:
                dist = float(np.linalg.norm(curr_pos - self.prev_pos))
                if dist > 25.0:
                    teleported = True
            self.prev_pos = curr_pos

            rew, terminated, reason = _monitor_step(
                self, speed_val, end_of_track, teleported, base_rew=float(base_rew)
            )

            info = {
                "action": np.array([data[5], data[6], data[7]], dtype="float32"),
                "end_of_track": end_of_track,
                "teleported": teleported,
                "termination_reason": reason,
            }
            return obs, rew, terminated, info

        TM2020Interface.get_obs_rew_terminated_info = (
            _online_rl_get_obs_rew_terminated_info
        )

        if not hasattr(TM2020Interface, "_original_reset"):
            TM2020Interface._original_reset = TM2020Interface.reset

        def _patched_reset(self, *args, **kwargs):
            self.prev_pos = None
            self._ep_frame_count = 0
            self._stuck_counter = 0
            self._warmup_counter = 0
            return self._original_reset(*args, **kwargs)

        TM2020Interface.reset = _patched_reset

        if not hasattr(ck, "_original_keyres"):
            ck._original_keyres = ck.keyres
        ck.keyres = _patched_safe_keyres

        logging.info("Applied TMRL Online RL patches cleanly.")
    except Exception as e:
        logging.warning(f"Failed to apply TMRL Online RL patches: {e}")
