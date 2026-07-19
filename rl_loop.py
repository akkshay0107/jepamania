"""Resumable online RL loop alternating persistent MPC collection and updates."""

import argparse
import gc
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import jax
from core.config import load_config
from train.config import load_train_config
from train.finetune import train_rl
from train.priority_dataloader import (
    read_completed_episode_summaries,
    select_replay_episodes,
)
from win_client.env_patches import apply_online_rl_patches
from win_client.game_session import GameSessionWorker
from win_client.mpc_driver import MPCDriver

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunState:
    iteration: int
    pending_episode_ids: tuple[int, ...]
    global_step: int
    model_checkpoint: str
    value_head_checkpoint: str


def save_run_state(checkpoints_dir: Path, state: RunState) -> None:
    """Atomically persist orchestration state after a durable phase boundary."""
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    temporary = checkpoints_dir / ".run_state.json.tmp"
    destination = checkpoints_dir / "run_state.json"
    temporary.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def load_run_state(checkpoints_dir: Path) -> Optional[RunState]:
    state_path = checkpoints_dir / "run_state.json"
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["pending_episode_ids"] = tuple(payload["pending_episode_ids"])
    return RunState(**payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable cyclic online RL")
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=Path("checkpoints/finetune/ft_model_latest.eqx"),
    )
    parser.add_argument(
        "--initial-value-head",
        type=Path,
        default=Path("checkpoints/finetune/ft_value_head_latest.eqx"),
    )
    parser.add_argument(
        "--rollout-file",
        type=Path,
        default=Path("win-client/data/rl/rollouts/online_rollouts.h5"),
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/rl"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iteration", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_checkpoint_bundle(model: Path, value_head: Path) -> None:
    candidates = (model, value_head, model.parent / "projectors.eqx")
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Online RL requires a complete fine-tuned checkpoint bundle; missing: "
            + ", ".join(str(path) for path in missing)
        )


def _initial_state(args: argparse.Namespace) -> RunState:
    existing = load_run_state(args.checkpoints_dir)
    if existing is not None:
        return existing
    return RunState(
        iteration=0,
        pending_episode_ids=(),
        global_step=0,
        model_checkpoint=str(args.initial_model),
        value_head_checkpoint=str(args.initial_value_head),
    )


def _collect_iteration(
    session: GameSessionWorker,
    state: RunState,
    rollout_file: Path,
    episode_target: int,
) -> tuple[int, ...]:
    summaries = read_completed_episode_summaries(rollout_file)
    completed = tuple(
        item.episode_id for item in summaries if item.iteration == state.iteration
    )
    missing = max(0, episode_target - len(completed))
    if missing:
        driver = MPCDriver(
            checkpoint_path=state.model_checkpoint,
            value_head_path=state.value_head_checkpoint,
        )
        collected = session.collect(driver, rollout_file, missing, state.iteration)
        completed += tuple(collected)
        del driver
        gc.collect()
        jax.clear_caches()
    return tuple(dict.fromkeys(completed))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.iterations < 0 or args.episodes_per_iteration <= 0:
        raise ValueError(
            "iterations must be non-negative and episodes must be positive"
        )

    state = _initial_state(args)
    _validate_checkpoint_bundle(
        Path(state.model_checkpoint), Path(state.value_head_checkpoint)
    )
    if args.dry_run:
        LOGGER.info(
            "Dry run: iteration %s -> %s, %s episodes per iteration",
            state.iteration,
            args.iterations,
            args.episodes_per_iteration,
        )
        return
    if state.iteration >= args.iterations:
        LOGGER.info("Online RL already completed %s iterations", state.iteration)
        return

    root_dir = Path(__file__).resolve().parent
    model_cfg = load_config(str(root_dir / "core" / "config.yaml"))
    train_cfg = load_train_config(str(root_dir / "train" / "config.yaml"))
    import wandb

    apply_online_rl_patches()
    session = GameSessionWorker()
    session.start()
    try:
        wandb.init(
            project="jepamania",
            name="online_rl_cyclic",
            mode="offline",
            config={
                "iterations": args.iterations,
                "episodes_per_iteration": args.episodes_per_iteration,
            },
        )
        while state.iteration < args.iterations:
            pending = state.pending_episode_ids
            if not pending:
                pending = _collect_iteration(
                    session,
                    state,
                    args.rollout_file,
                    args.episodes_per_iteration,
                )
                state = RunState(
                    state.iteration,
                    pending,
                    state.global_step,
                    state.model_checkpoint,
                    state.value_head_checkpoint,
                )
                save_run_state(args.checkpoints_dir, state)

            summaries = read_completed_episode_summaries(args.rollout_file)
            selection = select_replay_episodes(
                summaries,
                new_episode_ids=pending,
                current_iteration=state.iteration,
                historical_limit=train_cfg.finetune.replay_history_limit,
                recency_decay=train_cfg.finetune.replay_recency_decay,
                seed=train_cfg.finetune.seed + state.iteration,
            )
            output_dir = args.checkpoints_dir / f"iteration_{state.iteration:04d}"
            _, next_step = train_rl(
                data_dir=args.rollout_file,
                output_dir=output_dir,
                checkpoint=Path(state.model_checkpoint),
                value_head_path=Path(state.value_head_checkpoint),
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                episode_ids=selection.episode_ids,
                step_offset=state.global_step,
            )
            state = RunState(
                iteration=state.iteration + 1,
                pending_episode_ids=(),
                global_step=next_step,
                model_checkpoint=str(output_dir / "ft_model_latest.eqx"),
                value_head_checkpoint=str(output_dir / "ft_value_head_latest.eqx"),
            )
            save_run_state(args.checkpoints_dir, state)
            gc.collect()
            jax.clear_caches()
    finally:
        session.close()
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
