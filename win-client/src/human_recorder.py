import datetime
import logging
from pathlib import Path

import keyboard
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
    Hotkey-driven human play recorder.

    The recorder runs a continuous loop, stepping the environment at the
    real-time control frequency. Recording is toggled on/off by the configured
    hotkey; each ON→OFF cycle produces one HDF5 session file.
    """

    def __init__(self) -> None:
        self.recording_requested: bool = False
        self.recording_active: bool = False
        self.writer: HDF5Writer | None = None
        self.session_count: int = 0

        try:
            keyboard.on_press_key(cfg.human.record_hotkey, self._toggle_recording)
            logging.info(
                f"Hotkey '{cfg.human.record_hotkey}' registered — "
                "press it to start / stop recording."
            )
        except Exception as exc:
            logging.error(f"Failed to register hotkey: {exc}")

    def _toggle_recording(self, _event) -> None:
        self.recording_requested = not self.recording_requested

    def _make_session_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(cfg.data_output_dir) / f"human_session_{timestamp}.h5"

    def _default_metadata(self) -> dict:
        return {
            "source": "human",
            "map_name": "unknown",
            "map_uid": "unknown",
            "policy_name": "human",
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def run(self) -> None:
        env = tmrl.get_environment()
        raw_obs, _info = env.reset()
        obs_dict = obs_to_dict(raw_obs)

        dummy_action = np.zeros(cfg.action_dim, dtype=np.float32)

        logging.info(
            f"Ready. Press '{cfg.human.record_hotkey}' to start/stop recording. "
            "Press Ctrl-C to quit."
        )

        try:
            while True:
                if self.recording_requested and not self.recording_active:
                    self.session_count += 1
                    self.writer = HDF5Writer(self._make_session_path())
                    self.writer.new_episode(self._default_metadata())
                    self.recording_active = True
                    logging.info(f"STARTED recording session {self.session_count}.")

                elif not self.recording_requested and self.recording_active:
                    assert self.writer is not None
                    self.writer.end_episode(termination="manual")
                    self.writer.close()
                    self.writer = None
                    self.recording_active = False
                    logging.info(f"STOPPED recording session {self.session_count}.")

                raw_next, _reward, terminated, truncated, info = env.step(dummy_action)
                next_obs_dict = obs_to_dict(raw_next)
                done = terminated or truncated

                if self.recording_active and self.writer is not None:
                    # The game sends the human's actual inputs back in info.
                    actual_action = np.asarray(
                        info.get("action", dummy_action), dtype=np.float32
                    )
                    self.writer.append(obs_dict, actual_action)

                obs_dict = next_obs_dict

                if done:
                    if self.recording_active and self.writer is not None:
                        # Episode ended mid-recording -> record boundary and continue
                        # in the same session file.
                        self.writer.end_episode(termination="done")
                        self.writer.new_episode(self._default_metadata())
                        logging.info(
                            "Episode ended. Started new episode in same session."
                        )

                    raw_obs, _info = env.reset()
                    obs_dict = obs_to_dict(raw_obs)

        except KeyboardInterrupt:
            logging.info("Exiting HumanRecorder.")
        finally:
            if self.writer is not None:
                self.writer.end_episode(termination="manual")
                self.writer.close()
            env.close()


if __name__ == "__main__":
    recorder = HumanRecorder()
    recorder.run()
