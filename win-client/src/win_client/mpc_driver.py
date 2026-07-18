"""
Windows Real-Time MPC Driver for JEPAMania.

Bridges the trained pure JAX/Equinox Sub-JEPA latent world model and value head
with the live Trackmania (tmrl / rtgym) environment on Windows. Leverages
AsyncPlannerWrapper for delay-compensated asynchronous trajectory optimization.
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Optional

# ruff: noqa: E402
WIN_CLIENT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WIN_CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(WIN_CLIENT_ROOT))

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

from win_client.data_writer import HDF5Writer
from win_client.env_patches import apply_online_rl_patches
from win_client.settings import cfg
from win_client.utils import get_tmrl_env, obs_to_dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_subjepa_checkpoint(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path] = None,
    seed: int = 42,
) -> tuple[Encoder, Predictor, str, SubJepaConfig]:
    """Loads a combined (encoder, predictor) Equinox checkpoint with auto-detection."""
    if config_path is None:
        config_path = WIN_CLIENT_ROOT.parent / "core" / "config.yaml"

    model_cfg = load_config(str(config_path))
    key = jax.random.PRNGKey(seed)

    path = Path(checkpoint_path)
    (loaded_encoder, loaded_predictor), detected_type = load_models_auto(
        path, key, model_cfg.encoder, model_cfg.predictor
    )
    logging.info(f"Loaded Sub-JEPA models ({detected_type} encoder) from {path.name}")
    return loaded_encoder, loaded_predictor, detected_type, model_cfg


def load_value_head_checkpoint(
    value_head_path: str | Path,
    config_path: Optional[str | Path] = None,
    seed: int = 42,
) -> MLPValueHead:
    """Loads a pretrained MLPValueHead from an Equinox .eqx checkpoint."""
    path = Path(value_head_path)
    if not path.exists():
        raise FileNotFoundError(f"Value head checkpoint not found: {path}")

    if config_path is None:
        config_path = WIN_CLIENT_ROOT.parent / "core" / "config.yaml"

    model_cfg = load_config(str(config_path))
    key = jax.random.PRNGKey(seed)
    template = MLPValueHead(model_cfg.value_head, key)
    loaded_value_head = eqx.tree_deserialise_leaves(path, template)
    logging.info(f"Loaded MLPValueHead from {path.name}")
    return loaded_value_head


class MPCDriver:
    """
    Real-time Windows MPC Driver executing latent-space rollouts in Trackmania.

    Performs asynchronous background trajectory planning and sends raw planned
    continuous actions directly to the environment (no adaptive filter applied).
    Trajectory scoring relies solely on the learned MLPValueHead.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        value_head_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        max_episodes: Optional[int] = None,
    ) -> None:
        """Initializes real-time MPC driver with Sub-JEPA models and value head.

        Arguments:
          checkpoint_path: Path to combined (encoder, predictor) Equinox checkpoint
          value_head_path: Path to learned MLPValueHead Equinox checkpoint
          config_path: Optional custom path to model configuration yaml
          output_dir: Output directory for recorded rollout HDF5 files
          max_episodes: Maximum number of rollout episodes to complete before stopping
        """
        self.output_dir = (
            Path(output_dir) if output_dir else Path("win-client/data/rl/rollouts")
        )
        self.max_episodes = max_episodes
        self.checkpoint_path = checkpoint_path or cfg.mpc.checkpoint_path
        if not self.checkpoint_path:
            raise ValueError(
                "A combined model checkpoint path must be provided via arguments "
                "or settings.yaml (mpc.checkpoint_path)."
            )

        self.value_head_path = value_head_path or cfg.mpc.value_head_path
        if not self.value_head_path:
            raise ValueError(
                "A value head checkpoint path must be provided via arguments "
                "or settings.yaml (mpc.value_head_path)."
            )

        (
            self.encoder,
            self.predictor,
            self.encoder_type,
            self.model_cfg,
        ) = load_subjepa_checkpoint(
            checkpoint_path=self.checkpoint_path,
            config_path=config_path,
            seed=cfg.mpc.seed,
        )
        self.value_head = load_value_head_checkpoint(
            self.value_head_path,
            config_path=config_path,
            seed=cfg.mpc.seed,
        )

        self.objective_fn = self.value_head

        self.planner = self._build_planner()
        self.async_wrapper = AsyncPlannerWrapper(
            encoder=self.encoder,
            predictor=self.predictor,
            planner=self.planner,
            default_action=0,
            seed=cfg.mpc.seed,
        )

        self.obs_type = "lidar" if self.encoder_type == "lidar" else "screen"
        self.writer: Optional[HDF5Writer] = None

    def _build_planner(self):
        return create_planner(self.model_cfg.planner, self.predictor, self.objective_fn)

    def _make_session_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_dir or Path("win-client/data/rl/rollouts")
        return out_dir / f"mpc_rollouts_{timestamp}.h5"

    def _start_episode_record(self) -> None:
        if self.writer is not None:
            self.writer.new_episode(
                {
                    "source": "mpc_driver",
                    "map_name": cfg.agent.map_name,
                    "map_uid": cfg.agent.map_uid,
                    "encoder_type": self.encoder_type,
                    "planner_type": self.model_cfg.planner.type,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

    def run(self) -> None:
        env = get_tmrl_env()

        if cfg.mpc.record_rollouts or self.output_dir is not None:
            self.writer = HDF5Writer(self._make_session_path(), obs_type=self.obs_type)
            self._start_episode_record()

        logging.info("Starting real-time MPC control loop...")
        raw_obs, _info = env.reset()
        self.async_wrapper.start()

        completed_episodes = 0

        try:
            while True:
                planner_obs = obs_to_dict(
                    raw_obs, obs_type=self.obs_type, keep_history=True
                )
                if "screen" in planner_obs and np.issubdtype(
                    planner_obs["screen"].dtype, np.integer
                ):
                    planner_obs["screen"] = (
                        planner_obs["screen"].astype(np.float32) / 255.0
                    )
                elif "screen" in planner_obs and np.issubdtype(
                    planner_obs["screen"].dtype, np.floating
                ):
                    planner_obs["screen"] = planner_obs["screen"].astype(
                        np.float32
                    )
                for k in ("telemetry", "lidar"):
                    if k in planner_obs and planner_obs[k] is not None:
                        planner_obs[k] = planner_obs[k].astype(np.float32)

                rec_obs = obs_to_dict(
                    raw_obs, obs_type=self.obs_type, keep_history=False
                )
                discrete_action = self.async_wrapper.step(planner_obs)
                continuous_action = to_continuous_action_np(discrete_action).astype(
                    np.float32
                )

                raw_next, reward, terminated, truncated, info = env.step(
                    continuous_action
                )
                done = terminated or truncated

                if self.writer is not None:
                    self.writer.append(rec_obs, continuous_action, reward=float(reward))

                raw_obs = raw_next

                if done:
                    completed_episodes += 1
                    reason = info.get(
                        "termination_reason", "truncated" if truncated else "done"
                    )
                    logging.info(
                        f"Episode {completed_episodes} ended via reason: {reason} "
                        f"(terminal reward: {float(reward):.2f})"
                    )
                    if self.writer is not None:
                        self.writer.end_episode(termination=reason)

                    if (
                        self.max_episodes is not None
                        and completed_episodes >= self.max_episodes
                    ):
                        logging.info(
                            f"Target of {self.max_episodes} rollout episodes reached."
                        )
                        break

                    if (
                        self.writer is not None
                        and completed_episodes % cfg.episode_monitor.episodes_per_shard
                        == 0
                    ):
                        logging.info("Sharding HDF5 file: closing current shard")
                        self.writer.close()
                        self.writer = HDF5Writer(
                            self._make_session_path(), obs_type=self.obs_type
                        )

                    self._start_episode_record()
                    raw_obs, _info = env.reset()
                    self.async_wrapper.reset()

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Shutting down MPC driver.")
        finally:
            self.async_wrapper.stop()
            if self.writer is not None and not getattr(self.writer, "_closed", False):
                self.writer.end_episode(termination="manual")
                self.writer.close()
            env.close()
            logging.info("MPCDriver shut down cleanly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MPC trajectories and collect rollout data for online RL"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to combined Sub-JEPA Equinox checkpoint (.eqx)",
    )
    parser.add_argument(
        "--value-head-path",
        type=Path,
        required=True,
        help="Path to MLPValueHead Equinox checkpoint (.eqx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("win-client/data/rl/rollouts"),
        help="Directory to record rollout HDF5 files",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of rollout episodes to record before stopping",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.info(f"Rollout collector for {args.num_episodes} eps -> {args.output_dir}")

    apply_online_rl_patches()

    driver = MPCDriver(
        checkpoint_path=args.checkpoint_path,
        value_head_path=args.value_head_path,
        output_dir=args.output_dir,
        max_episodes=args.num_episodes,
    )
    driver.run()


if __name__ == "__main__":
    main()
