"""Cyclic online RL orchestrator: alternate rollout collection and fine-tuning.

Pure orchestration — all training hyperparameters live in train/config.yaml
and are read by the fine-tuning script itself.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    cmd_str = " ".join(cmd)
    logging.info(f"Executing: {cmd_str}")
    if dry_run:
        logging.info("[Dry Run] Command skipped.")
        return
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cyclic Online RL Orchestrator")
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=Path("checkpoints/pretrain/pretrain_model_latest.eqx"),
        help="Combined (encoder, predictor) Sub-JEPA checkpoint to start from",
    )
    parser.add_argument(
        "--bootstrap-dir",
        type=Path,
        default=Path("win-client/data/rl/bootstrap"),
        help="Path to initial reward-labeled bootstrap HDF5 data",
    )
    parser.add_argument(
        "--rollouts-dir",
        type=Path,
        default=Path("win-client/data/rl/rollouts"),
        help="Base directory for online rollout HDF5 files",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints/rl"),
        help="Shared fine-tune output directory for RL checkpoints",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of cyclic RL iterations to run",
    )
    parser.add_argument(
        "--episodes-per-iter",
        type=int,
        default=5,
        help="Number of rollout episodes to collect per iteration",
    )
    parser.add_argument(
        "--python-train",
        type=str,
        default=sys.executable,
        help="Python executable for training script",
    )
    parser.add_argument(
        "--python-client",
        type=str,
        default=sys.executable,
        help="Python executable for Windows client rollout script",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(__file__).resolve().parent
    finetune_script = str(root_dir / "train" / "src" / "finetune.py")
    mpc_script = str(root_dir / "win-client" / "src" / "mpc_driver.py")

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.rollouts_dir.mkdir(parents=True, exist_ok=True)

    model_ckpt = args.checkpoints_dir / "ft_model_latest.eqx"
    valhead_ckpt = args.checkpoints_dir / "ft_value_head_latest.eqx"

    for iter_idx in range(args.num_iterations):
        logging.info(f"=== Starting RL Cycle Iteration {iter_idx} ===")

        # A missing value head means no fine-tuning has completed yet: start
        # fresh from the pretrained model on bootstrap data. Otherwise collect
        # rollouts with the current models and continue fine-tuning from them.
        fresh = not valhead_ckpt.exists()

        if fresh:
            logging.info(
                f"[Iter {iter_idx}] Bootstrapping on {args.bootstrap_dir}..."
            )
            data_dir = args.bootstrap_dir
            train_cmd = [
                args.python_train,
                finetune_script,
                "--data-dir",
                str(data_dir),
                "--checkpoint",
                str(args.pretrain_checkpoint),
                "--output-dir",
                str(args.checkpoints_dir),
            ]
        else:
            iter_rollout_dir = args.rollouts_dir / f"iter_{iter_idx}"
            iter_rollout_dir.mkdir(parents=True, exist_ok=True)

            logging.info(
                f"[Iter {iter_idx}] Collecting {args.episodes_per_iter} rollout eps..."
            )
            rollout_cmd = [
                args.python_client,
                mpc_script,
                "--checkpoint-path",
                str(model_ckpt),
                "--value-head-path",
                str(valhead_ckpt),
                "--output-dir",
                str(iter_rollout_dir),
                "--num-episodes",
                str(args.episodes_per_iter),
            ]
            run_cmd(rollout_cmd, dry_run=args.dry_run)

            logging.info(f"[Iter {iter_idx}] Fine-tuning on collected rollouts...")
            train_cmd = [
                args.python_train,
                finetune_script,
                "--data-dir",
                str(iter_rollout_dir),
                "--checkpoint",
                str(model_ckpt),
                "--value-head",
                str(valhead_ckpt),
                "--output-dir",
                str(args.checkpoints_dir),
            ]

        run_cmd(train_cmd, dry_run=args.dry_run)

    logging.info("=== RL Cyclic Orchestration Complete ===")


if __name__ == "__main__":
    main()
