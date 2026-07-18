"""Cyclic online RL orchestrator: alternate rollout collection and fine-tuning.

Pure orchestration — all hyperparameters live in train/config.yaml and
win-client/settings.yaml. Runs cleanly across Option B workspace packages
without namespace collisions or subprocess overhead.
"""

import argparse
import gc
import logging
from pathlib import Path
from typing import Optional

import jax
from core.config import load_config
from train.config import load_train_config
from train.finetune import train_rl
from win_client.env_patches import apply_online_rl_patches
from win_client.mpc_driver import MPCDriver

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_finetune_phase(
    data_dir: Path,
    checkpoint_path: Path,
    value_head_path: Optional[Path],
    output_dir: Path,
    root_dir: Path,
    dry_run: bool = False,
    step_offset: int = 0,
    use_importance_sampling: bool = False,
) -> int:
    logging.info(
        f"Fine-tuning phase on data={data_dir} -> output={output_dir} "
        f"(importance_sampling={use_importance_sampling})"
    )
    if dry_run:
        logging.info("[Dry Run] Fine-tuning skipped.")
        return step_offset

    model_cfg = load_config(str(root_dir / "core" / "config.yaml"))
    train_cfg = load_train_config(str(root_dir / "train" / "config.yaml"))
    _, next_step = train_rl(
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint=checkpoint_path,
        value_head_path=value_head_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        step_offset=step_offset,
        use_importance_sampling=use_importance_sampling,
    )
    gc.collect()
    jax.clear_caches()
    return next_step


def run_mpc_phase(
    checkpoint_path: Path,
    value_head_path: Path,
    output_dir: Path,
    num_episodes: int,
    dry_run: bool = False,
) -> None:
    logging.info(f"MPC rollout collection ({num_episodes} eps) -> {output_dir}")
    if dry_run:
        logging.info("[Dry Run] Rollout collection skipped.")
        return

    apply_online_rl_patches()
    driver = MPCDriver(
        checkpoint_path=checkpoint_path,
        value_head_path=value_head_path,
        output_dir=output_dir,
        max_episodes=num_episodes,
    )
    driver.run()
    gc.collect()
    jax.clear_caches()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stateless cyclic RL training orchestrator."
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=Path("checkpoints/pretrain/pretrain_model_latest.eqx"),
    )
    parser.add_argument(
        "--bootstrap-dir",
        type=Path,
        default=Path("win-client/data/rl/bootstrap"),
    )
    parser.add_argument(
        "--rollouts-dir",
        type=Path,
        default=Path("win-client/data/rl/rollouts"),
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints/rl"),
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iter", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _seed_rollouts_from_bootstrap(
    bootstrap_dir: Path, rollouts_dir: Path, num_episodes: int = 5
) -> None:
    if list(rollouts_dir.rglob("*.h5")):
        return
    bootstrap_files = sorted(bootstrap_dir.rglob("*.h5"))
    if not bootstrap_files:
        logging.warning(
            f"No bootstrap HDF5 files found in {bootstrap_dir} to seed rollouts."
        )
        return
    logging.info(
        f"Seeding {rollouts_dir} with up to {num_episodes} "
        "episodes from bootstrap data..."
    )
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    import h5py
    import numpy as np
    from win_client.data_writer import HDF5Writer

    seed_path = rollouts_dir / "online_rollouts.h5"
    writer = None
    episodes_copied = 0

    for bf in bootstrap_files:
        if episodes_copied >= num_episodes:
            break
        try:
            with h5py.File(bf, "r") as f:
                if "episode_id" not in f or "observations" not in f:
                    continue
                obs_type = "lidar" if "lidar" in f["observations"] else "screen"  # type: ignore
                if writer is None:
                    writer = HDF5Writer(seed_path, obs_type=obs_type, append=False)

                ep_ids = np.asarray(f["episode_id"], dtype=np.int32)
                unique_eps = np.unique(ep_ids)
                for ep in unique_eps:
                    if episodes_copied >= num_episodes:
                        break
                    mask = ep_ids == ep
                    indices = np.where(mask)[0]
                    writer.new_episode(
                        {"source": "bootstrap_seed", "original_ep": int(ep)}
                    )
                    telemetry_slice = f["observations/telemetry"][indices]  # type: ignore
                    if obs_type == "lidar":
                        main_obs_slice = f["observations/lidar"][indices]  # type: ignore
                    else:
                        main_obs_slice = f["observations/screen"][indices]  # type: ignore
                    acts_slice = f["actions"][indices]  # type: ignore
                    if "rewards" in f:
                        rews_slice = f["rewards"][indices]  # type: ignore
                    else:
                        rews_slice = np.zeros(len(indices), dtype=np.float32)

                    for idx_loc in range(len(indices)):
                        obs = {"telemetry": telemetry_slice[idx_loc]}
                        if obs_type == "lidar":
                            obs["lidar"] = main_obs_slice[idx_loc]
                        else:
                            obs["screen"] = main_obs_slice[idx_loc]
                        writer.append(
                            obs, acts_slice[idx_loc], float(rews_slice[idx_loc])
                        )
                    writer.end_episode("done")
                    episodes_copied += 1
        except Exception as e:
            logging.warning(f"Error while reading bootstrap shard {bf}: {e}")

    if writer is not None:
        writer.close()
        logging.info(f"Successfully seeded {episodes_copied} episodes into {seed_path}")


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.rollouts_dir.mkdir(parents=True, exist_ok=True)

    model_ckpt = args.checkpoints_dir / "ft_model_latest.eqx"
    valhead_ckpt = args.checkpoints_dir / "ft_value_head_latest.eqx"

    import wandb

    if not args.dry_run:
        wandb.init(
            project="jepamania",
            name="online_rl_cyclic",
            mode="offline",
            config={
                "num_iterations": args.num_iterations,
                "episodes_per_iter": args.episodes_per_iter,
            },
        )

    try:
        _seed_rollouts_from_bootstrap(
            args.bootstrap_dir, args.rollouts_dir, num_episodes=args.episodes_per_iter
        )
        current_step = 0
        for iter_idx in range(args.num_iterations):
            logging.info(f"=== Starting RL Cycle Iteration {iter_idx} ===")
            if not valhead_ckpt.exists():
                logging.info(
                    f"[Iter {iter_idx}] Bootstrapping on {args.bootstrap_dir}..."
                )
                current_step = run_finetune_phase(
                    data_dir=args.bootstrap_dir,
                    checkpoint_path=args.pretrain_checkpoint,
                    value_head_path=None,
                    output_dir=args.checkpoints_dir,
                    root_dir=root_dir,
                    dry_run=args.dry_run,
                    step_offset=current_step,
                    use_importance_sampling=False,
                )
            else:
                logging.info(
                    f"[Iter {iter_idx}] Collecting {args.episodes_per_iter}"
                    "rollouts directly into {args.rollouts_dir}..."
                )
                run_mpc_phase(
                    checkpoint_path=model_ckpt,
                    value_head_path=valhead_ckpt,
                    output_dir=args.rollouts_dir,
                    num_episodes=args.episodes_per_iter,
                    dry_run=args.dry_run,
                )
                logging.info(
                    f"[Iter {iter_idx}] Fine-tuning on collected rollouts"
                    " in {args.rollouts_dir}..."
                )
                current_step = run_finetune_phase(
                    data_dir=args.rollouts_dir,
                    checkpoint_path=model_ckpt,
                    value_head_path=valhead_ckpt,
                    output_dir=args.checkpoints_dir,
                    root_dir=root_dir,
                    dry_run=args.dry_run,
                    step_offset=current_step,
                    use_importance_sampling=True,
                )

        logging.info("=== RL Cyclic Orchestration Complete ===")
    finally:
        if not args.dry_run and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
