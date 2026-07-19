"""Real-time latent MPC control and rollout collection for Trackmania."""

import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax
import numpy as np
from core.actions import to_continuous_action_np
from core.async_planner import AsyncPlannerWrapper
from core.config import SubJepaConfig, load_config
from core.dynamics import MLPValueHead
from core.encoders import load_models_auto
from core.interfaces import Encoder, Predictor
from core.planners import create_planner

from win_client.data_writer import HDF5Writer, validate_hdf5_schema
from win_client.env_patches import apply_online_rl_patches
from win_client.settings import cfg
from win_client.utils import get_tmrl_env, obs_to_dict

WIN_CLIENT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_subjepa_checkpoint(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path] = None,
    seed: int = 42,
) -> tuple[Encoder, Predictor, str, SubJepaConfig]:
    """Load a combined encoder/predictor checkpoint with encoder detection."""
    config_path = config_path or WIN_CLIENT_ROOT.parent / "core" / "config.yaml"
    model_cfg = load_config(str(config_path))
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    models, detected_type = load_models_auto(
        path,
        jax.random.PRNGKey(seed),
        model_cfg.encoder,
        model_cfg.predictor,
    )
    return models[0], models[1], detected_type, model_cfg


def load_value_head_checkpoint(
    value_head_path: str | Path,
    config_path: Optional[str | Path] = None,
    seed: int = 42,
) -> MLPValueHead:
    """Load a serialized value head."""
    path = Path(value_head_path)
    if not path.exists():
        raise FileNotFoundError(f"Value head checkpoint not found: {path}")
    config_path = config_path or WIN_CLIENT_ROOT.parent / "core" / "config.yaml"
    model_cfg = load_config(str(config_path))
    template = MLPValueHead(model_cfg.value_head, jax.random.PRNGKey(seed))
    return eqx.tree_deserialise_leaves(path, template)


class MPCDriver:
    """Run asynchronous MPC against an environment supplied by its owner."""

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        value_head_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        max_episodes: Optional[int] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path or cfg.mpc.checkpoint_path
        self.value_head_path = value_head_path or cfg.mpc.value_head_path
        if not self.checkpoint_path or not self.value_head_path:
            raise ValueError("Both model and value-head checkpoints are required")

        self.output_dir = Path(output_dir) if output_dir else None
        self.max_episodes = max_episodes
        self.encoder, self.predictor, self.encoder_type, self.model_cfg = (
            load_subjepa_checkpoint(
                self.checkpoint_path, config_path=config_path, seed=cfg.mpc.seed
            )
        )
        self.value_head = load_value_head_checkpoint(
            self.value_head_path, config_path=config_path, seed=cfg.mpc.seed
        )
        planner = create_planner(
            self.model_cfg.planner, self.predictor, self.value_head
        )
        self.async_wrapper = AsyncPlannerWrapper(
            encoder=self.encoder,
            predictor=self.predictor,
            planner=planner,
            default_action=0,
            seed=cfg.mpc.seed,
        )
        self.obs_type = "lidar" if self.encoder_type == "lidar" else "screen"

    def _planner_observation(self, raw_obs: Any) -> dict[str, np.ndarray]:
        observation = obs_to_dict(raw_obs, obs_type=self.obs_type, keep_history=True)
        if "screen" in observation:
            screen = observation["screen"]
            observation["screen"] = screen.astype(np.float32)
            if np.issubdtype(screen.dtype, np.integer):
                observation["screen"] /= 255.0
        for name in ("telemetry", "lidar"):
            if name in observation:
                observation[name] = observation[name].astype(np.float32)
        return observation

    def collect(
        self,
        env: Any,
        rollout_file: Optional[str | Path],
        num_episodes: Optional[int],
        iteration: int,
    ) -> tuple[int, ...]:
        """Collect episodes without taking ownership of the environment."""
        writer: Optional[HDF5Writer] = None
        if rollout_file is not None:
            path = Path(rollout_file)
            if path.exists() and validate_hdf5_schema(path) != self.obs_type:
                raise ValueError(
                    f"Rollout observation type does not match {self.obs_type} encoder"
                )
            writer = HDF5Writer(path, obs_type=self.obs_type, append=True)

        raw_obs, _ = env.reset()
        completed_ids: list[int] = []
        active_episode_id: Optional[int] = None
        self.async_wrapper.reset()
        self.async_wrapper.start()

        def start_episode() -> None:
            nonlocal active_episode_id
            if writer is None:
                return
            active_episode_id = writer.new_episode(
                {
                    "source": "mpc_driver",
                    "iteration": iteration,
                    "map_name": cfg.agent.map_name,
                    "map_uid": cfg.agent.map_uid,
                    "encoder_type": self.encoder_type,
                    "planner_type": self.model_cfg.planner.type,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

        start_episode()
        try:
            while num_episodes is None or len(completed_ids) < num_episodes:
                planner_obs = self._planner_observation(raw_obs)
                recorded_obs = obs_to_dict(
                    raw_obs, obs_type=self.obs_type, keep_history=False
                )
                action_index = self.async_wrapper.step(planner_obs)
                action = to_continuous_action_np(action_index).astype(np.float32)
                raw_obs, reward, terminated, truncated, info = env.step(action)
                if writer is not None:
                    writer.append(recorded_obs, action, float(reward))

                if not (terminated or truncated):
                    continue
                reason = info.get(
                    "termination_reason", "truncated" if truncated else "done"
                )
                if writer is not None:
                    writer.end_episode(termination=reason)
                    if active_episode_id is None:
                        raise RuntimeError("Writer lost the active episode ID")
                    completed_ids.append(active_episode_id)
                    active_episode_id = None
                elif num_episodes is not None:
                    completed_ids.append(len(completed_ids))
                if num_episodes is not None and len(completed_ids) >= num_episodes:
                    break
                raw_obs, _ = env.reset()
                self.async_wrapper.reset()
                start_episode()
        finally:
            self.async_wrapper.stop()
            if writer is not None:
                # Missing completion metadata keeps interrupted data out of replay.
                writer.close()
        return tuple(completed_ids)

    def run(self) -> None:
        """Compatibility entry point for standalone autonomous driving."""
        env = get_tmrl_env()
        try:
            rollout_file = None
            if cfg.mpc.record_rollouts or self.output_dir is not None:
                output_dir = self.output_dir or Path("win-client/data/rl/rollouts")
                if output_dir.suffix == ".h5":
                    rollout_file = output_dir
                else:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    rollout_file = output_dir / "online_rollouts.h5"
            self.collect(env, rollout_file, self.max_episodes, iteration=0)
        finally:
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect real-time MPC rollouts")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--value-head-path", type=Path, required=True)
    parser.add_argument(
        "--rollout-file",
        type=Path,
        default=Path("win-client/data/rl/rollouts/online_rollouts.h5"),
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_online_rl_patches()
    env = get_tmrl_env()
    try:
        driver = MPCDriver(args.checkpoint_path, args.value_head_path)
        driver.collect(env, args.rollout_file, args.num_episodes, iteration=0)
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
