import datetime
import logging
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import tmrl
from src.data_writer import HDF5Writer
from src.settings import cfg
from src.terrain_scheduler import MapConfig, TerrainScheduler
from src.utils import obs_to_dict
from tmrl.actor import TorchActorModule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_policy(path_str: str | None) -> Callable[[dict], np.ndarray]:
    """Load a PyTorch SAC actor. Raises an error if path is None or invalid."""
    if not path_str:
        raise ValueError(
            "A policy checkpoint must be configured for all tracks. "
            "Random OUNoise exploration is disabled."
        )

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {path}")

    actor = TorchActorModule()
    actor.load(str(path), device="cpu")
    logging.info(f"Policy: loaded SAC actor from {path.name}")

    def _actor(obs_dict: dict) -> np.ndarray:
        return np.asarray(actor.act(obs_dict, test=True), dtype=np.float32)

    return _actor


def switch_map(uid: str, env) -> dict:
    """Trigger a map switch via Play Map Extended and reset the env."""
    logging.info(f"Switching map to UID: {uid}")
    uri = f"trackmania://openplanet/playmapextended/open?uid={uid}"
    webbrowser.open(uri)
    time.sleep(cfg.map_cycler.map_load_wait_s)
    raw_obs, _info = env.reset()
    return obs_to_dict(raw_obs)


class AgentCollector:
    """
    Drives the full data collection loop:
    Schedules maps, runs the environment, records data, and handles episode resets.
    """

    def __init__(self) -> None:

        registry_path = Path(__file__).parent.parent / cfg.terrain_maps_path
        self._scheduler = TerrainScheduler(registry_path)

    def _session_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(cfg.data_output_dir) / f"agent_{timestamp}.h5"

    def run(self) -> None:
        """Start the automated data collection loop."""
        env = tmrl.get_environment()
        writer = HDF5Writer(self._session_path())

        map_config: MapConfig = next(self._scheduler)
        policy = load_policy(map_config.policy)

        logging.info(f"Starting on terrain={map_config.terrain!r}")
        obs_dict = switch_map(map_config.uid, env)

        writer.new_episode(
            {
                "source": "agent",
                "terrain_type": map_config.terrain,
                "map_name": map_config.name,
                "map_uid": map_config.uid,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

        warmup_counter = 0
        speed_window = deque(maxlen=cfg.episode_monitor.stuck_window_frames)
        frame_count = 0
        completed_episodes = 0
        terrain_episodes = 0

        try:
            while True:
                action = policy(obs_dict)
                raw_next, _reward, terminated, truncated, info = env.step(action)
                next_obs_dict = obs_to_dict(raw_next)
                done = terminated or truncated
                speed = float(info.get("speed", 0.0))

                if warmup_counter < cfg.episode_monitor.warmup_frames:
                    warmup_counter += 1
                    obs_dict = next_obs_dict
                    continue

                writer.append(obs_dict, action)
                obs_dict = next_obs_dict
                frame_count += 1
                speed_window.append(speed)

                reason = None
                if done:
                    reason = "done"
                elif (
                    len(speed_window) == speed_window.maxlen
                    and max(speed_window) < cfg.episode_monitor.stuck_speed_kmh
                ):
                    reason = "stuck"
                elif frame_count >= cfg.episode_monitor.max_frames_per_episode:
                    reason = "frame_budget"

                if not reason:
                    continue

                writer.end_episode(termination=reason)
                completed_episodes += 1
                terrain_episodes += 1
                frame_count = 0
                speed_window.clear()
                warmup_counter = 0

                if terrain_episodes >= cfg.episode_monitor.terrain_episode_budget:
                    terrain_episodes = 0
                    map_config = next(self._scheduler)
                    policy = load_policy(map_config.policy)
                    obs_dict = switch_map(map_config.uid, env)
                else:
                    raw_obs, _info = env.reset()
                    obs_dict = obs_to_dict(raw_obs)

                writer.new_episode(
                    {
                        "source": "agent",
                        "terrain_type": map_config.terrain,
                        "map_name": map_config.name,
                        "map_uid": map_config.uid,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                )

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received.")
        finally:
            writer.end_episode(termination="manual")
            writer.close()
            logging.info("AgentCollector shut down cleanly.")


def main() -> None:
    collector = AgentCollector()
    collector.run()


if __name__ == "__main__":
    main()
