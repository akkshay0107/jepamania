import datetime
import logging
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import tmrl
from src.data_writer import HDF5Writer
from src.settings import cfg
from src.utils import OUNoise, obs_to_dict
from tmrl.actor import TorchActorModule

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_policy(path_str: str | None) -> Callable[[dict], np.ndarray]:
    """Load a PyTorch SAC actor. Raises an error if path is None or invalid."""
    if not path_str:
        raise ValueError(
            "A policy checkpoint must be configured. "
            "Please specify agent.policy_path in settings.yaml."
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


class AgentCollector:
    """
    Drives the data collection loop on the currently loaded map:
    Runs the environment, records data, and handles episode resets.
    """

    def __init__(self) -> None:
        pass

    def _session_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(cfg.data_output_dir) / f"agent_{timestamp}.h5"

    def run(self) -> None:
        """Start the map-agnostic data collection loop."""
        env = tmrl.get_environment()
        writer = HDF5Writer(self._session_path())

        policy = load_policy(cfg.agent.policy_path)

        logging.info("Starting collection. Make sure you are loaded into the map.")
        raw_obs, _info = env.reset()
        obs_dict = obs_to_dict(raw_obs)

        noise = OUNoise(
            size=cfg.action_dim,
            mu=cfg.agent.exploration.ou_noise_mu,
            theta=cfg.agent.exploration.ou_noise_theta,
            sigma=cfg.agent.exploration.ou_noise_sigma,
        )
        noise.reset()

        # Load metadata from config overrides or default to unknown
        map_name = getattr(cfg.agent, "map_name", "unknown")
        map_uid = getattr(cfg.agent, "map_uid", "unknown")

        writer.new_episode(
            {
                "source": "agent",
                "map_name": map_name,
                "map_uid": map_uid,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

        warmup_counter = 0
        speed_window = deque(maxlen=cfg.episode_monitor.stuck_window_frames)
        frame_count = 0
        completed_episodes = 0

        try:
            while True:
                action = policy(obs_dict)
                action = action + noise()
                action[0] = np.clip(action[0], -1.0, 1.0)
                action[1] = np.clip(action[1], 0.0, 1.0)
                action[2] = np.clip(action[2], 0.0, 1.0)
                action = action.astype(np.float32)
                raw_next, _reward, terminated, truncated, info = env.step(action)
                next_obs_dict = obs_to_dict(raw_next)
                done = terminated or truncated
                speed = float(info.get("speed", 0.0))

                reason = None
                if done:
                    reason = "done"
                elif (
                    warmup_counter >= cfg.episode_monitor.warmup_frames
                    and len(speed_window) == speed_window.maxlen
                    and max(speed_window) < cfg.episode_monitor.stuck_speed_kmh
                ):
                    reason = "stuck"
                elif (
                    warmup_counter >= cfg.episode_monitor.warmup_frames
                    and frame_count >= cfg.episode_monitor.max_frames_per_episode
                ):
                    reason = "frame_budget"

                if reason:
                    writer.end_episode(termination=reason)
                    completed_episodes += 1
                    logging.info(
                        f"Episode {completed_episodes} ended via reason: {reason}"
                    )

                    frame_count = 0
                    speed_window.clear()
                    warmup_counter = 0

                    raw_obs, _info = env.reset()
                    obs_dict = obs_to_dict(raw_obs)
                    noise.reset()

                    if completed_episodes % cfg.episode_monitor.episodes_per_shard == 0:
                        logging.info(
                            "Sharding HDF5 file: "
                            "closing current shard and starting a new one."
                        )
                        writer.close()
                        writer = HDF5Writer(self._session_path())

                    writer.new_episode(
                        {
                            "source": "agent",
                            "map_name": map_name,
                            "map_uid": map_uid,
                            "timestamp": datetime.datetime.now().isoformat(),
                        }
                    )
                    continue

                if warmup_counter < cfg.episode_monitor.warmup_frames:
                    warmup_counter += 1
                    obs_dict = next_obs_dict
                    continue

                writer.append(obs_dict, action)
                obs_dict = next_obs_dict
                frame_count += 1
                speed_window.append(speed)

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
