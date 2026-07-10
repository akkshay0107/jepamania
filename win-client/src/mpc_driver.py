"""
Windows Real-Time MPC Driver for JEPAMania.

Bridges the trained pure JAX/Equinox Sub-JEPA latent world model and value head
with the live Trackmania (tmrl / rtgym) environment on Windows. Leverages
AsyncPlannerWrapper for delay-compensated asynchronous trajectory optimization.
"""

import argparse
import datetime
import logging
from collections import deque
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import numpy as np
from core.actions import to_continuous_action_np
from core.async_planner import AsyncPlannerWrapper
from core.config import load_config
from core.dynamics import MLPPredictor, MLPValueHead
from core.encoders import ConvEncoder, LidarEncoder, ViTEncoder
from core.interfaces import Encoder, Predictor
from core.planners import BeamSearchPlanner, CEMPlanner, RandomShootingPlanner
from src.data_writer import HDF5Writer
from src.env_patches import apply_data_collection_patches
from src.settings import cfg
from src.utils import get_tmrl_env, obs_to_dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_subjepa_checkpoint(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path] = None,
    encoder_type: str = "vit",
    seed: int = 42,
) -> tuple[Encoder, Predictor]:
    """Loads pretrained Encoder and MLPPredictor weights from an Equinox .eqx file."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Sub-JEPA checkpoint not found: {path}")

    if config_path is None:
        config_path = (
            Path(__file__).resolve().parent.parent.parent / "core" / "config.yaml"
        )

    model_cfg = load_config(str(config_path))
    key = jax.random.PRNGKey(seed)
    key_enc, key_pred = jax.random.split(key)

    if encoder_type == "vit":
        encoder = ViTEncoder(model_cfg.encoder, key_enc)
    elif encoder_type == "lidar":
        encoder = LidarEncoder(model_cfg.encoder, key_enc)
    elif encoder_type == "conv":
        encoder = ConvEncoder(model_cfg.encoder, key_enc)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    predictor = MLPPredictor(model_cfg.predictor, key_pred)
    loaded_encoder, loaded_predictor = eqx.tree_deserialise_leaves(
        path, (encoder, predictor)
    )
    logging.info(f"Loaded Sub-JEPA models ({encoder_type} encoder) from {path.name}")
    return loaded_encoder, loaded_predictor


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
        config_path = (
            Path(__file__).resolve().parent.parent.parent / "core" / "config.yaml"
        )

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
        encoder_type: Optional[str] = None,
        planner_type: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        max_episodes: Optional[int] = None,
    ) -> None:
        """Initializes real-time MPC driver with Sub-JEPA models and value head.

        Arguments:
          checkpoint_path: Path to Sub-JEPA pretrained Equinox checkpoint
          value_head_path: Path to learned MLPValueHead Equinox checkpoint
          config_path: Optional custom path to model configuration yaml
          encoder_type: Encoder modality ('screen', 'lidar', or 'conv')
          planner_type: Trajectory planning algorithm ('cem', 'beam', or 'random')
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
                "A Sub-JEPA checkpoint path must be provided via arguments "
                "or settings.yaml (mpc.checkpoint_path)."
            )

        self.value_head_path = value_head_path or cfg.mpc.value_head_path
        if not self.value_head_path:
            raise ValueError(
                "A value head checkpoint path must be provided via arguments "
                "or settings.yaml (mpc.value_head_path)."
            )

        self.encoder_type = encoder_type or cfg.mpc.encoder_type
        self.planner_type = planner_type or cfg.mpc.planner_type

        self.encoder, self.predictor = load_subjepa_checkpoint(
            self.checkpoint_path,
            config_path=config_path,
            encoder_type=self.encoder_type,
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
        seq_len = int(cfg.mpc.sequence_len)
        if self.planner_type == "cem":
            return CEMPlanner(
                predictor=self.predictor,
                objective_fn=self.objective_fn,
                sequence_len=seq_len,
                num_iters=int(cfg.mpc.num_iters),
                num_samples=int(cfg.mpc.num_samples),
                num_elites=int(cfg.mpc.num_elites),
                alpha=0.25,
            )
        elif self.planner_type == "beam":
            return BeamSearchPlanner(
                predictor=self.predictor,
                objective_fn=self.objective_fn,
                sequence_len=seq_len,
                beam_width=int(cfg.mpc.beam_width),
            )
        elif self.planner_type == "random":
            return RandomShootingPlanner(
                predictor=self.predictor,
                objective_fn=self.objective_fn,
                sequence_len=seq_len,
                num_samples=int(cfg.mpc.num_samples),
            )
        else:
            raise ValueError(
                f"Unsupported planner '{self.planner_type}'; "
                "choose from 'cem', 'beam', 'random'."
            )

    def _make_session_path(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_dir or Path("win-client/data/rl/rollouts")
        return out_dir / f"mpc_rollouts_{timestamp}.h5"

    def run(self) -> None:
        env = get_tmrl_env()

        if cfg.mpc.record_rollouts or self.output_dir is not None:
            self.writer = HDF5Writer(self._make_session_path(), obs_type=self.obs_type)
            self.writer.new_episode(
                {
                    "source": "mpc_driver",
                    "map_name": cfg.agent.map_name,
                    "map_uid": cfg.agent.map_uid,
                    "encoder_type": self.encoder_type,
                    "planner_type": self.planner_type,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

        logging.info("Starting real-time MPC control loop...")
        raw_obs, _info = env.reset()
        self.async_wrapper.start()

        warmup_counter = 0
        speed_window = deque(maxlen=cfg.episode_monitor.stuck_window_frames)
        frame_count = 0
        completed_episodes = 0

        try:
            while True:
                obs_dict = obs_to_dict(raw_obs, obs_type=self.obs_type)
                discrete_action = self.async_wrapper.step(obs_dict)
                continuous_action = to_continuous_action_np(discrete_action).astype(
                    np.float32
                )

                raw_next, _reward, terminated, truncated, _info = env.step(
                    continuous_action
                )
                done = terminated or truncated
                speed = float(np.asarray(raw_next[0]).flat[0])
                if warmup_counter >= cfg.episode_monitor.warmup_frames:
                    speed_window.append(speed)

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
                    completed_episodes += 1
                    logging.info(
                        f"Episode {completed_episodes} ended via reason: {reason}"
                    )
                    if self.writer is not None and frame_count > 0:
                        self.writer.end_episode(termination=reason)

                    if (
                        self.max_episodes is not None
                        and completed_episodes >= self.max_episodes
                    ):
                        logging.info(
                            f"Target of {self.max_episodes} rollout episodes reached. "
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

                    if self.writer is not None:
                        self.writer.new_episode(
                            {
                                "source": "mpc_driver",
                                "map_name": cfg.agent.map_name,
                                "map_uid": cfg.agent.map_uid,
                                "timestamp": datetime.datetime.now().isoformat(),
                            }
                        )

                    warmup_counter = 0
                    frame_count = 0
                    speed_window.clear()
                    raw_obs, _info = env.reset()
                    self.async_wrapper.reset()
                    continue

                if warmup_counter < cfg.episode_monitor.warmup_frames:
                    warmup_counter += 1
                    raw_obs = raw_next
                    continue

                if self.writer is not None:
                    self.writer.append(
                        obs_dict, continuous_action, reward=float(_reward)
                    )

                raw_obs = raw_next
                frame_count += 1

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Shutting down MPC driver.")
        finally:
            self.async_wrapper.stop()
            if self.writer is not None:
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
        required=True,
        help="Path to Sub-JEPA Equinox checkpoint (.eqx)",
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
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="screen",
        choices=["screen", "lidar"],
        help="Encoder modality",
    )
    parser.add_argument(
        "--planner-type",
        type=str,
        default="cem",
        choices=["cem", "beam", "random"],
        help="Trajectory planner algorithm",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.info(f"Rollout collector for {args.num_episodes} eps -> {args.output_dir}")

    apply_data_collection_patches()

    driver = MPCDriver(
        checkpoint_path=args.checkpoint_path,
        value_head_path=args.value_head_path,
        encoder_type=args.encoder_type,
        planner_type=args.planner_type,
        output_dir=args.output_dir,
        max_episodes=args.num_episodes,
    )
    driver.run()


if __name__ == "__main__":
    main()
