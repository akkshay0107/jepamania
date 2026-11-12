"""Explicit environment patches for Trackmania (TMRL / rtgym) interface on Windows."""

import logging
import platform
import sys

import numpy as np

ACTION_HISTORY_LEN = 2


def configure_rtgym_action_history(*, online: bool = False) -> None:
    """
    Configure TMRL's native chronological action buffer.
    """
    if platform.system() != "Windows":
        return

    import tmrl.config.config_constants as tmrl_config  # type: ignore[import-not-found]

    env_config = getattr(tmrl_config, "ENV_CONFIG", None)
    if not isinstance(env_config, dict):
        raise RuntimeError("TMRL ENV_CONFIG is unavailable or has an invalid type.")

    rtgym_config = env_config.get("RTGYM_CONFIG")
    if not isinstance(rtgym_config, dict):
        raise RuntimeError("TMRL RTGYM_CONFIG is unavailable or has an invalid type.")

    rtgym_config.update(
        {
            "act_in_obs": True,
            "act_buf_len": ACTION_HISTORY_LEN,
            "reset_act_buf": True,
        }
    )
    if online:
        from win_client.game_session import GAME_HZ, MAX_EPISODE_STEPS

        if not tmrl_config.PRAGMA_GAMEPAD:
            raise RuntimeError("Online driving requires TMRL's virtual gamepad.")
        rtgym_config["time_step_duration"] = 1 / GAME_HZ
        rtgym_config["ep_max_length"] = MAX_EPISODE_STEPS
    tmrl_config.ACT_BUF_LEN = ACTION_HISTORY_LEN

    # TMRL 0.7.1 materializes CONFIG_DICT at import time.  Keep it aligned if
    # another module imported config_objects before this helper ran.
    config_objects = sys.modules.get("tmrl.config.config_objects")
    if config_objects is not None:
        config_dict = getattr(config_objects, "CONFIG_DICT", None)
        if isinstance(config_dict, dict):
            config_dict.update(rtgym_config)


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
            # Windows can deny foreground activation while another app owns focus.
            logging.debug("Trackmania foreground activation denied.")
        time.sleep(0.05)
        if win32gui.GetForegroundWindow() == hwnd:
            if hasattr(ck, "_original_keyres"):
                ck._original_keyres()
            else:
                ck.keyres()
        else:
            logging.warning("Trackmania window not focused. Reset key skipped.")


def _collection_end(
    self, speed_val: float, end_of_track: bool, teleported: bool
) -> str | None:
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

    if reason is not None:
        self.prev_pos = None
        self._ep_frame_count = 0
        self._stuck_counter = 0
        self._warmup_counter = 0

    return reason


def _capture_trackmania(self):
    data, image = self.grab_data_and_img()
    speed = np.array([data[0]], dtype=np.float32)
    gear = np.array([data[9]], dtype=np.float32)
    rpm = np.array([data[10]], dtype=np.float32)
    position = np.asarray(data[2:5], dtype=np.float32)
    previous = getattr(self, "prev_pos", None)
    teleported = previous is not None and np.linalg.norm(position - previous) > 25.0
    self.prev_pos = position
    self.img_hist.append(image)
    observation = [speed, gear, rpm, np.asarray(self.img_hist)]
    return observation, {
        "action": np.asarray(data[5:8], dtype=np.float32),
        "end_of_track": bool(data[8]),
        "teleported": bool(teleported),
    }


def _collection_get_obs_rew_terminated_info(self):
    obs, info = _capture_trackmania(self)
    reason = _collection_end(
        self, float(obs[0][0]), info["end_of_track"], info["teleported"]
    )
    info["termination_reason"] = reason
    return obs, np.float32(0.0), reason is not None, info


def _online_get_obs_rew_terminated_info(self):
    obs, info = _capture_trackmania(self)
    finished = info["end_of_track"]
    teleported = info["teleported"]
    info["termination_reason"] = (
        "done" if finished else "respawn" if teleported else None
    )
    # The driving adapter owns the fixed replay reward and terminal semantics.
    return obs, np.float32(0.0), finished or teleported, info


def apply_data_collection_patches() -> None:
    """Apply recording cutoffs and expose controls for SSL data collection."""
    _apply_patches(online=False)


def apply_online_rl_patches() -> None:
    """Expose game events at the ordinary 20 Hz cadence."""
    _apply_patches(online=True)


def _apply_patches(*, online: bool) -> None:
    if platform.system() != "Windows":
        return

    configure_rtgym_action_history(online=online)
    import tmrl.custom.tm.utils.control_keyboard as ck  # type: ignore[import-not-found]
    from tmrl.custom.tm.tm_gym_interfaces import (  # type: ignore[import-not-found]
        TM2020Interface,
    )

    TM2020Interface.get_obs_rew_terminated_info = (
        _online_get_obs_rew_terminated_info
        if online
        else _collection_get_obs_rew_terminated_info
    )
    if not hasattr(TM2020Interface, "_original_reset"):
        TM2020Interface._original_reset = TM2020Interface.reset

    def reset(self, *args, **kwargs):
        self.prev_pos = None
        self._ep_frame_count = 0
        self._stuck_counter = 0
        self._warmup_counter = 0
        return self._original_reset(*args, **kwargs)

    TM2020Interface.reset = reset
    if not hasattr(ck, "_original_keyres"):
        ck._original_keyres = ck.keyres
    ck.keyres = _patched_safe_keyres
