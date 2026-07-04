import datetime
import logging
from pathlib import Path

import numpy as np
import tmrl
from src.data_writer import HDF5Writer
from src.settings import cfg
from src.utils import obs_to_dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class HumanRecorder:
    """
    Map-agnostic human play recorder.

    Immediately starts recording frames on startup. Stepping the environment
    exposes the human's actual inputs via the environment's info dictionary.
    Exiting via Ctrl+C closes the session file cleanly.
    """

    def __init__(self) -> None:
        self.writer: HDF5Writer | None = None

    def _make_session_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(cfg.data_output_dir) / f"human_session_{timestamp}.h5"

    def _default_metadata(self) -> dict:
        map_name = getattr(cfg.agent, "map_name", "unknown")
        map_uid = getattr(cfg.agent, "map_uid", "unknown")
        return {
            "source": "human",
            "map_name": map_name,
            "map_uid": map_uid,
            "policy_name": "human",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def run(self) -> None:
        env = tmrl.get_environment()

        # Initialize writer and start recording immediately
        self.writer = HDF5Writer(self._make_session_path())
        self.writer.new_episode(self._default_metadata())

        raw_obs, _info = env.reset()
        obs_dict = obs_to_dict(raw_obs)

        warmup_counter = 0
        frame_count = 0
        dummy_action = np.zeros(cfg.action_dim, dtype=np.float32)

        logging.info("Recording STARTED. Make inputs in-game. Press Ctrl-C to stop.")

        try:
            while True:
                raw_next, _reward, terminated, truncated, info = env.step(dummy_action)
                next_obs_dict = obs_to_dict(raw_next)
                done = terminated or truncated

                # Validate Openplanet input streaming on the first captured step
                if frame_count == 0 and warmup_counter == 0:
                    if "action" not in info:
                        logging.warning(
                            "\n=====================================================\n"
                            "WARNING: 'action' not found in environment info!\n"
                            "Openplanet is NOT streaming your human inputs.\n"
                            "Every frame will log dummy [0, 0, 0] actions.\n"
                            "Please check your Openplanet TMRL plugin status.\n"
                            "=====================================================\n"
                        )

                reason = None
                if done:
                    reason = "done"

                if reason:
                    if self.writer is not None:
                        # Episode ended -> record boundary
                        # and continue in same session file.
                        self.writer.end_episode(termination=reason)
                        self.writer.new_episode(self._default_metadata())
                        logging.info(
                            "Episode ended. Starting new episode in same session."
                        )

                    raw_obs, _info = env.reset()
                    obs_dict = obs_to_dict(raw_obs)
                    warmup_counter = 0
                    frame_count = 0
                    continue

                if warmup_counter < cfg.episode_monitor.warmup_frames:
                    warmup_counter += 1
                    obs_dict = next_obs_dict
                    continue

                if self.writer is not None:
                    # The game sends the human's actual inputs back in info.
                    actual_action = np.asarray(
                        info.get("action", dummy_action), dtype=np.float32
                    )
                    self.writer.append(obs_dict, actual_action)
                    frame_count += 1

                obs_dict = next_obs_dict

        except KeyboardInterrupt:
            logging.info("Exiting HumanRecorder. Stopping collection.")
        finally:
            if self.writer is not None:
                self.writer.end_episode(termination="manual")
                self.writer.close()
            env.close()


if __name__ == "__main__":
    recorder = HumanRecorder()
    recorder.run()
