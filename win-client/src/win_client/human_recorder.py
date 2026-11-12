import datetime
import logging
from collections import deque
from contextlib import closing
from pathlib import Path

import numpy as np

from win_client.data_writer import HDF5Writer
from win_client.game_session import IDLE_ACTION
from win_client.settings import cfg
from win_client.utils import get_tmrl_env, obs_to_dict


class HumanRecorder:
    """Record the player's executed controls and observations."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path(cfg.data_output_dir)

    def _make_session_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"human_session_{timestamp}.h5"

    def _default_metadata(self) -> dict:
        return {
            "source": "human",
            "map_name": cfg.agent.map_name,
            "map_uid": cfg.agent.map_uid,
            "policy_name": "human",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def run(self) -> None:
        with closing(get_tmrl_env()) as env:
            raw_obs, _ = env.reset()
            writer = HDF5Writer(self._make_session_path())
            action_history = deque((IDLE_ACTION, IDLE_ACTION), maxlen=2)
            try:
                writer.new_episode(self._default_metadata())
                logging.info("Recording started. Press Ctrl-C to stop.")
                while True:
                    recorded_obs = obs_to_dict(raw_obs)
                    recorded_obs["telemetry"][3:] = np.concatenate(action_history)
                    raw_obs, _, terminated, truncated, info = env.step(
                        IDLE_ACTION.copy()
                    )
                    if "action" not in info:
                        raise RuntimeError(
                            "Openplanet did not provide the player's controls."
                        )
                    action = np.array(info["action"], dtype=np.float32)
                    if (
                        action.shape != IDLE_ACTION.shape
                        or not np.isfinite(action).all()
                    ):
                        raise ValueError("Openplanet returned invalid player controls.")
                    writer.append(recorded_obs, action)
                    action_history.append(action)

                    if terminated or truncated:
                        reason = info.get("termination_reason") or (
                            "truncated" if truncated else "done"
                        )
                        writer.end_episode(reason)
                        logging.info(
                            "Episode ended (%s). Starting a new episode.", reason
                        )
                        raw_obs, _ = env.reset()
                        action_history.extend((IDLE_ACTION, IDLE_ACTION))
                        writer.new_episode(self._default_metadata())
            except KeyboardInterrupt:
                logging.info("Stopping human recording.")
            finally:
                writer.end_episode("manual")
                writer.close()
