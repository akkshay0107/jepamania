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
    res = subprocess.run(cmd, check=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cyclic Online RL Orchestrator")
    parser.add_argument(
        "--ssl-checkpoint",
        type=Path,
        required=True,
        help="Pre-trained SSL Sub-JEPA Equinox checkpoint",
    )
    parser.add_argument(
        "--bootstrap-dir",
        type=Path,
        default=Path("data/rl/bootstrap"),
        help="Path to initial reward-labeled bootstrap HDF5 data",
    )
    parser.add_argument(
        "--rollouts-dir",
        type=Path,
        default=Path("data/rl/rollouts"),
        help="Base directory for online rollout HDF5 files",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints/rl"),
        help="Base directory for RL checkpoints",
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
        "--warmup-epochs",
        type=int,
        default=5,
        help="Value head warmup epochs on initial bootstrap training",
    )
    parser.add_argument(
        "--joint-epochs",
        type=int,
        default=5,
        help="Joint fine-tuning epochs per iteration",
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
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrap iteration 0 if already trained",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.rollouts_dir.mkdir(parents=True, exist_ok=True)

    current_subjepa = args.ssl_checkpoint
    current_valhead: Path | None = None

    for iter_idx in range(args.num_iterations):
        logging.info(f"=== Starting RL Cycle Iteration {iter_idx} ===")

        iter_ckpt_dir = args.checkpoints_dir / f"iter_{iter_idx}"

        if iter_idx == 0:
            if not args.skip_bootstrap:
                logging.info(
                    f"[Iter 0] Fine-tuning on bootstrap data ({args.bootstrap_dir})..."
                )
                train_cmd = [
                    args.python_train,
                    "train/src/finetune.py",
                    "--data-dir",
                    str(args.bootstrap_dir),
                    "--ssl-checkpoint",
                    str(args.ssl_checkpoint),
                    "--output-dir",
                    str(iter_ckpt_dir),
                    "--warmup-epochs",
                    str(args.warmup_epochs),
                    "--joint-epochs",
                    str(args.joint_epochs),
                ]
                run_cmd(train_cmd, dry_run=args.dry_run)
            else:
                logging.info("[Iteration 0] Skipping bootstrap training step.")
                current_subjepa = iter_ckpt_dir / "rl_joint_latest_subjepa.eqx"
                current_valhead = iter_ckpt_dir / "rl_joint_latest_value_head.eqx"
                assert current_valhead is not None
                if not args.dry_run and (
                    not current_subjepa.exists() or not current_valhead.exists()
                ):
                    raise FileNotFoundError(
                        f"Cannot skip bootstrap: checkpoints missing in {iter_ckpt_dir}"
                    )
            current_subjepa = iter_ckpt_dir / "rl_joint_latest_subjepa.eqx"
            current_valhead = iter_ckpt_dir / "rl_joint_latest_value_head.eqx"
            continue

        # --- Iteration > 0: Rollout + Train ---
        assert current_valhead is not None, "Value head checkpoint is not set."

        iter_rollout_dir = args.rollouts_dir / f"iter_{iter_idx}"
        iter_rollout_dir.mkdir(parents=True, exist_ok=True)

        msg = f"[Iter {iter_idx}] Collecting {args.episodes_per_iter} rollout eps..."
        logging.info(msg)
        rollout_cmd = [
            args.python_client,
            "win-client/src/mpc_driver.py",
            "--checkpoint-path",
            str(current_subjepa),
            "--value-head-path",
            str(current_valhead),
            "--output-dir",
            str(iter_rollout_dir),
            "--num-episodes",
            str(args.episodes_per_iter),
        ]
        run_cmd(rollout_cmd, dry_run=args.dry_run)

        msg2 = f"[Iter {iter_idx}] Fine-tuning Sub-JEPA + Val Head on rollouts..."
        logging.info(msg2)
        train_cmd = [
            args.python_train,
            "train/src/finetune.py",
            "--data-dir",
            str(iter_rollout_dir),
            "--ssl-checkpoint",
            str(current_subjepa),
            "--value-head-checkpoint",
            str(current_valhead),
            "--output-dir",
            str(iter_ckpt_dir),
            "--warmup-epochs",
            "0",  # No warmup needed after iteration 0
            "--joint-epochs",
            str(args.joint_epochs),
        ]
        run_cmd(train_cmd, dry_run=args.dry_run)

        current_subjepa = iter_ckpt_dir / "rl_joint_latest_subjepa.eqx"
        current_valhead = iter_ckpt_dir / "rl_joint_latest_value_head.eqx"

    logging.info("=== RL Cyclic Orchestration Complete ===")


if __name__ == "__main__":
    main()
