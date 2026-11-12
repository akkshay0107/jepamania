import datetime
import logging
from contextlib import closing
from pathlib import Path

import numpy as np

from win_client.data_writer import HDF5Writer
from win_client.settings import cfg
from win_client.utils import (
    AdaptiveActionFilter,
    OUNoise,
    get_tmrl_env,
    get_tmrl_obs_preprocessor,
    get_tmrl_policy_class,
    obs_to_dict,
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
    policy_cls = get_tmrl_policy_class()
    actor = policy_cls(observation_space=observation_space, action_space=action_space)
    actor = actor.load(str(path), device="cpu")
    logging.info(f"Policy: loaded {policy_cls.__name__} from {path.name}")
    return actor


class AgentCollector:
    """
    Drives the data collection loop on the currently loaded map:
    Runs the environment, records data, and handles episode resets.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or Path(cfg.data_output_dir)

    def _session_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"agent_{timestamp}.h5"

    def run(self) -> None:
        with closing(get_tmrl_env()) as env:
            actor = load_actor(
                cfg.agent.policy_path, env.observation_space, env.action_space
            )
            obs_preprocessor = get_tmrl_obs_preprocessor()
            raw_obs, _ = env.reset()
            noise = OUNoise(
                size=cfg.action_dim,
                mu=cfg.agent.exploration.ou_noise_mu,
                theta=cfg.agent.exploration.ou_noise_theta,
                sigma=cfg.agent.exploration.ou_noise_sigma,
            )
            action_filter = AdaptiveActionFilter(
                enabled=cfg.agent.filter.enabled,
                steer_deadzone=cfg.agent.filter.steer_deadzone,
                min_alpha=cfg.agent.filter.min_alpha,
                max_alpha=cfg.agent.filter.max_alpha,
                delta_scale=cfg.agent.filter.delta_scale,
            )
            writer = HDF5Writer(self._session_path())
            completed_episodes = 0
            try:
                writer.new_episode(self._metadata())
                while True:
                    preprocessed_obs = obs_preprocessor(raw_obs)
                    action = np.asarray(
                        actor.act_(preprocessed_obs, test=True), dtype=np.float32
                    )
                    action[2] += noise()[2]
                    action = action_filter(action)
                    np.clip(action, [0.0, 0.0, -1.0], [1.0, 1.0, 1.0], out=action)

                    recorded_obs = obs_to_dict(raw_obs)
                    raw_obs, _, terminated, truncated, info = env.step(action)
                    writer.append(recorded_obs, action)
                    if terminated or truncated:
                        reason = info.get("termination_reason") or (
                            "truncated" if truncated else "done"
                        )
                        writer.end_episode(reason)
                        completed_episodes += 1
                        logging.info(
                            "Episode %s ended (%s).", completed_episodes, reason
                        )

                        raw_obs, _ = env.reset()
                        noise.reset()
                        action_filter.reset()
                        if (
                            completed_episodes % cfg.episode_monitor.episodes_per_shard
                            == 0
                        ):
                            writer.close()
                            writer = HDF5Writer(self._session_path())
                        writer.new_episode(self._metadata())
            except KeyboardInterrupt:
                logging.info("Stopping agent recording.")
            finally:
                writer.end_episode("manual")
                writer.close()

    def _metadata(self) -> dict:
        return {
            "source": "agent",
            "map_name": cfg.agent.map_name,
            "map_uid": cfg.agent.map_uid,
            "timestamp": datetime.datetime.now().isoformat(),
        }
