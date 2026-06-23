import datetime
import logging
from pathlib import Path

import gym
import keyboard
import numpy as np
from src.data_writer import HDF5Writer
from src.settings import cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class HumanRecorder:
    def __init__(self):
        self.recording_requested = False
        self.recording_active = False
        self.writer = None
        self.session_count = 0

        try:
            keyboard.on_press_key(cfg.human.record_hotkey, self.toggle_recording)
            logging.info(
                f"Hotkey '{cfg.human.record_hotkey}' registered to toggle recording."
            )
        except Exception as e:
            logging.error(f"Failed to register hotkey. Error: {e}")

    def toggle_recording(self, event):
        self.recording_requested = not self.recording_requested

    def run(self):
        env_id = "rtgym:real-time-gym-v0"
        logging.info(f"Initializing environment: {env_id}")

        try:
            env = gym.make(env_id)
        except Exception as e:
            logging.error(f"Failed to initialize environment. Error: {e}")
            return

        obs = env.reset()
        logging.info(
            f"Ready. Press {cfg.human.record_hotkey} to start/stop recording."
            "Press Ctrl+C to quit."
        )

        dummy_action = np.zeros(cfg.action_dim, dtype=np.float32)

        try:
            while True:
                if self.recording_requested and not self.recording_active:
                    self.session_count += 1
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = (
                        Path(cfg.data_output_dir) / f"human_session_{timestamp}.h5"
                    )
                    self.writer = HDF5Writer(filepath=filepath)
                    self.recording_active = True
                    logging.info(
                        f"STARTED recording session {self.session_count} to {filepath}"
                    )

                elif not self.recording_requested and self.recording_active:
                    if self.writer is not None:
                        self.writer.close()
                        self.writer = None
                    self.recording_active = False
                    logging.info(f"STOPPED recording session {self.session_count}.")

                next_obs, reward, done, info = env.step(dummy_action)

                if self.recording_active and self.writer is not None:
                    image = obs if isinstance(obs, np.ndarray) else obs[0]

                    telemetry = {
                        "speed": info.get("speed", 0.0),
                        "gear": info.get("gear", 0),
                        "rpm": info.get("rpm", 0.0),
                    }

                    actual_action = info.get("action", dummy_action)

                    self.writer.append(image, telemetry, actual_action)

                obs = next_obs

                if done:
                    obs = env.reset()

        except KeyboardInterrupt:
            logging.info("Exiting Human Recorder.")
        finally:
            if self.writer is not None:
                self.writer.close()
            env.close()


if __name__ == "__main__":
    recorder = HumanRecorder()
    recorder.run()
