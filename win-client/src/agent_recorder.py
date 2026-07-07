import datetime
import logging
from collections import deque
from pathlib import Path

import numpy as np
import tmrl
import tmrl.config.config_objects as cfg_obj
from src.data_writer import HDF5Writer
from src.settings import cfg
from src.utils import AdaptiveActionFilter, OUNoise, obs_to_dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_actor(path_str: str | None, observation_space, action_space):
    """
    Load a pretrained SAC actor like how tmrl does it for their
    pretrained model
    """
    if not path_str:
        raise ValueError(
            "A policy checkpoint must be configured. "
            "Please specify agent.policy_path in settings.yaml."
        )

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {path}")

    # cfg_obj.POLICY is set at import time based on config.json:
    #   - grayscale images  → SquashedGaussianVanillaCNNActor
    #   - color images      → SquashedGaussianVanillaColorCNNActor
    #   - lidar             → SquashedGaussianMLPActor
    policy_cls = cfg_obj.POLICY
    actor = policy_cls(observation_space=observation_space, action_space=action_space)
    actor = actor.load(str(path), device="cpu")
    logging.info(f"Policy: loaded {policy_cls.__name__} from {path.name}")
    return actor


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
        env = tmrl.get_environment()
        writer = HDF5Writer(self._session_path())

        actor = load_actor(
            cfg.agent.policy_path,
            env.observation_space,
            env.action_space,
        )

        # tmrl --test uses cfg_obj.OBS_PREPROCESSOR, which is
        # obs_preprocessor_tm_act_in_obs for the full (non-lidar) env.
        obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

        logging.info("Starting collection. Make sure you are loaded into the map.")
        raw_obs, _info = env.reset()

        noise = OUNoise(
            size=cfg.action_dim,
            mu=cfg.agent.exploration.ou_noise_mu,
            theta=cfg.agent.exploration.ou_noise_theta,
            sigma=cfg.agent.exploration.ou_noise_sigma,
        )
        noise.reset()

        action_filter = AdaptiveActionFilter(
            enabled=cfg.agent.filter.enabled,
            steer_deadzone=cfg.agent.filter.steer_deadzone,
            min_alpha=cfg.agent.filter.min_alpha,
            max_alpha=cfg.agent.filter.max_alpha,
            delta_scale=cfg.agent.filter.delta_scale,
        )
        action_filter.reset()

        map_name = cfg.agent.map_name
        map_uid = cfg.agent.map_uid

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
                preprocessed_obs = obs_preprocessor(raw_obs)
                action = actor.act_(preprocessed_obs, test=True)
                action = np.asarray(action, dtype=np.float32)
                action[2] = action[2] + noise()[2]
                action = action_filter(action)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

                raw_next, _reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                speed = float(np.asarray(raw_next[0]).flat[0])

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
                    noise.reset()
                    action_filter.reset()

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
                    raw_obs = raw_next
                    continue

                obs_dict = obs_to_dict(raw_next)
                writer.append(obs_dict, action)
                raw_obs = raw_next
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
