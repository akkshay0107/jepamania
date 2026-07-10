import datetime
import logging
from pathlib import Path

import numpy as np
from src.data_writer import HDF5Writer
from src.settings import cfg
from src.utils import get_tmrl_env, obs_to_dict

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

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path(cfg.data_output_dir)
        self.writer: HDF5Writer | None = None

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
        env = get_tmrl_env()

        self.writer = HDF5Writer(self._make_session_path())
        self.writer.new_episode(self._default_metadata())

        _raw_obs, _info = env.reset()

        frame_count = 0
        completed_episodes = 0
        dummy_action = np.zeros(cfg.action_dim, dtype=np.float32)

        logging.info("Recording STARTED. Make inputs in-game. Press Ctrl-C to stop.")

        try:
            while True:
                raw_next, reward, terminated, truncated, info = env.step(dummy_action)
                next_obs_dict = obs_to_dict(raw_next)
                done = terminated or truncated

                # Validate Openplanet input streaming on the first captured step
                if frame_count == 0:
                    if "action" not in info:
                        logging.warning(
                            "\n=====================================================\n"
                            "WARNING: 'action' not found in environment info!\n"
                            "Openplanet is NOT streaming your human inputs.\n"
                            "Every frame will log dummy [0, 0, 0] actions.\n"
                            "Please check your Openplanet TMRL plugin status.\n"
                            "=====================================================\n"
                        )

                actual_action = np.asarray(
                    info.get("action", dummy_action), dtype=np.float32
                )
                if self.writer is not None:
                    self.writer.append(
                        next_obs_dict, actual_action, reward=float(reward)
                    )

                if done:
                    completed_episodes += 1
                    reason = info.get(
                        "termination_reason", "truncated" if truncated else "done"
                    )
                    if self.writer is not None:
                        self.writer.end_episode(termination=reason)
                        self.writer.new_episode(self._default_metadata())
                    logging.info(
                        f"Episode {completed_episodes} completed ({reason}). "
                        "Starting new episode."
                    )

                    _raw_obs, _info = env.reset()
                    frame_count = 0
                    continue

                frame_count += 1

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
