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
) -> None:
    logging.info(f"Fine-tuning phase on data={data_dir} -> output={output_dir}")
    if dry_run:
        logging.info("[Dry Run] Fine-tuning skipped.")
        return

    model_cfg = load_config(str(root_dir / "core" / "config.yaml"))
    train_cfg = load_train_config(str(root_dir / "train" / "config.yaml"))
    train_rl(
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint=checkpoint_path,
        value_head_path=value_head_path,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    gc.collect()
    jax.clear_caches()


def run_mpc_phase(
    checkpoint_path: Path,
    value_head_path: Path,
    output_dir: Path,
    num_episodes: int,
    planner_type: str,
    dry_run: bool = False,
) -> None:
    logging.info(
        f"MPC rollout collection ({num_episodes} eps, "
        f"planner={planner_type}) -> {output_dir}"
    )
    if dry_run:
        logging.info("[Dry Run] Rollout collection skipped.")
        return

    apply_online_rl_patches()
    driver = MPCDriver(
        checkpoint_path=checkpoint_path,
        value_head_path=value_head_path,
        planner_type=planner_type,
        output_dir=output_dir,
        max_episodes=num_episodes,
    )
    driver.run()
    gc.collect()
    jax.clear_caches()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cyclic Online RL Orchestrator")
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
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iter", type=int, default=5)
    parser.add_argument(
        "--planner-type",
        type=str,
        default="cem",
        choices=["cem", "beam", "random"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.rollouts_dir.mkdir(parents=True, exist_ok=True)

    model_ckpt = args.checkpoints_dir / "ft_model_latest.eqx"
    valhead_ckpt = args.checkpoints_dir / "ft_value_head_latest.eqx"

    for iter_idx in range(args.num_iterations):
        logging.info(f"=== Starting RL Cycle Iteration {iter_idx} ===")
        if not valhead_ckpt.exists():
            logging.info(f"[Iter {iter_idx}] Bootstrapping on {args.bootstrap_dir}...")
            run_finetune_phase(
                data_dir=args.bootstrap_dir,
                checkpoint_path=args.pretrain_checkpoint,
                value_head_path=None,
                output_dir=args.checkpoints_dir,
                root_dir=root_dir,
                dry_run=args.dry_run,
            )
        else:
            iter_rollout_dir = args.rollouts_dir / f"iter_{iter_idx}"
            iter_rollout_dir.mkdir(parents=True, exist_ok=True)
            logging.info(
                f"[Iter {iter_idx}] Collecting {args.episodes_per_iter} rollouts..."
            )
            run_mpc_phase(
                checkpoint_path=model_ckpt,
                value_head_path=valhead_ckpt,
                output_dir=iter_rollout_dir,
                num_episodes=args.episodes_per_iter,
                planner_type=args.planner_type,
                dry_run=args.dry_run,
            )
            logging.info(f"[Iter {iter_idx}] Fine-tuning on collected rollouts...")
            run_finetune_phase(
                data_dir=iter_rollout_dir,
                checkpoint_path=model_ckpt,
                value_head_path=valhead_ckpt,
                output_dir=args.checkpoints_dir,
                root_dir=root_dir,
                dry_run=args.dry_run,
            )

    logging.info("=== RL Cyclic Orchestration Complete ===")


if __name__ == "__main__":
    main()
