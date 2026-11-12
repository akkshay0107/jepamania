"""Collect, train, publish, and evaluate Trackmania RL in fixed epochs."""

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, cast

import h5py
import jax
import numpy as np
from core.actions import ACTION_GRID_NP, discretize_action_np
from core.config import SubJepaConfig, load_config
from core.encoders import DrivingModels
from core.planners import CEMPlanner
from omegaconf import OmegaConf
from train.config import FinetuneConfig, TrainConfig, load_train_config
from train.dataloader import SlidingWindowDataset
from train.finetune import OnlineState, build_joint_learner
from train.pretrain import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_TRAIN_CONFIG,
    Batch,
    legacy_splits,
    load_checkpoint,
    log_json,
    new_state,
    sample_legacy,
    save_checkpoint,
)
from train.rollout_episodes import publish_episode
from win_client.env_patches import apply_online_rl_patches
from win_client.game_session import REWARD_DEFINITION, DrivingEnvironment, Observation
from win_client.mpc_driver import (
    DRIVER_CHECKPOINT_VERSION,
    PREPROCESSING,
    Episode,
    MPCDriver,
    drive_episode,
    make_driving_models,
)
from win_client.settings import cfg as game_config
from win_client.utils import get_tmrl_env

HEURISTIC_ACTIONS = discretize_action_np(
    np.array(
        [(1, 0, steer) for steer in (-0.5, -0.2, 0.0, 0.2, 0.5)]
        + [(0, 0, 0), (0, 1, 0)],
        dtype=np.float32,
    )
)


class RunState(NamedTuple):
    learner: OnlineState
    completed_epoch: int
    attempt: int
    replay_cursor: int


def heuristic_driver(rng: np.random.Generator) -> Callable[[Observation], int]:
    tick = 0
    held_action = 0

    def select(_observation: Observation) -> int:
        nonlocal tick, held_action
        if tick % 4 == 0:
            held_action = int(
                rng.choice(HEURISTIC_ACTIONS, p=(0.1, 0.15, 0.3, 0.15, 0.1, 0.1, 0.1))
            )
        tick += 1
        return held_action

    return select


def collect_drive(
    env: DrivingEnvironment,
    driver: MPCDriver,
    models: DrivingModels,
    use_cem: bool,
    seed: int,
    number: int,
) -> Episode:
    for retry in range(3):
        if use_cem:
            driver.reset(2 * (3 * number + retry))
            select = partial(driver.select, models)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, number, retry]))
            select = heuristic_driver(rng)
        episode = drive_episode(env, select)
        if episode.actions:
            return episode
    raise RuntimeError("Three game resets produced no complete transition.")


def include_drive(
    path: Path,
    epoch: int,
    transitions: SlidingWindowDataset,
    windows: SlidingWindowDataset,
) -> None:
    with h5py.File(path, "r") as file:
        episode = cast(h5py.Group, file["metadata/episode_0"])
        if (
            file.attrs.get("reward_convention") != REWARD_DEFINITION
            or episode.attrs.get("epoch") != epoch
            or episode.attrs.get("checkpoint_version") != epoch - 1
        ):
            raise ValueError(f"Replay reward, epoch, or checkpoint differs in {path}.")
    before = len(transitions)
    transitions.include_online_shard(path)
    if len(transitions) == before:
        raise ValueError(f"No complete transitions in {path}.")
    windows.include_online_shard(path)


def sample_replay(
    data: SlidingWindowDataset, size: int, rng: np.random.Generator, recent_start: int
) -> Batch:
    recent_size = (size + 1) // 2 if recent_start < len(data) else 0
    indices = rng.integers(len(data), size=size)
    if recent_size:
        indices[:recent_size] = rng.integers(recent_start, len(data), size=recent_size)
    return data.read_batch(indices)


def write_text_atomic(path: Path, contents: str) -> None:
    temp = path.with_suffix(".part")
    temp.write_text(contents, encoding="utf-8")
    temp.replace(path)


def evaluate(
    env: DrivingEnvironment, driver: MPCDriver, state: RunState, cfg: FinetuneConfig
) -> dict[str, int | float | None]:
    epoch = state.completed_epoch
    models = state.learner.models
    episodes = []
    for race in range(cfg.evaluation_races):
        driver.reset(2 * (epoch * cfg.evaluation_races + race) + 1)
        episodes.append(
            drive_episode(env, partial(driver.select, models), keep_observations=False)
        )

    finishes = [ep.elapsed_seconds for ep in episodes if ep.reason == "done"]
    latencies = [latency for ep in episodes for latency in ep.inference_ms]
    return {
        "epoch": epoch,
        "checkpoint_version": epoch,
        "finish_rate": len(finishes) / len(episodes),
        "median_finish_seconds": float(np.median(finishes)) if finishes else None,
        "mean_discounted_return": float(
            np.mean([ep.discounted_return(cfg.gamma) for ep in episodes])
        ),
        "inference_p99_ms": float(np.percentile(latencies, 99)),
    }


def evaluate_if_due(
    root: Path,
    env: DrivingEnvironment,
    driver: MPCDriver,
    state: RunState,
    cfg: FinetuneConfig,
) -> None:
    epoch, interval = state.completed_epoch, cfg.evaluation_interval
    if epoch == 0 or interval is None or epoch % interval:
        return
    path = root / "evaluation.jsonl"
    history = path.read_text(encoding="utf-8") if path.is_file() else ""
    if not history or json.loads(history.splitlines()[-1])["epoch"] != epoch:
        row = json.dumps(evaluate(env, driver, state, cfg), allow_nan=False)
        write_text_atomic(path, f"{history}{row}\n")


def publish_driver(path: Path, state: RunState, inference: dict[str, Any]) -> None:
    save_checkpoint(
        path,
        state.learner.models,
        metadata={
            **inference,
            "checkpoint_version": state.completed_epoch,
            "updates": int(state.learner.updates),
        },
    )


def run_epochs(
    root: Path,
    state: RunState,
    update: Callable,
    legacy: list[SlidingWindowDataset],
    config: TrainConfig,
    driver: MPCDriver,
    env: DrivingEnvironment,
    inference: dict[str, Any],
) -> None:
    """Restore replay, then run collection, updates, and publication in order.

    Arguments:
      root: Directory for replay, checkpoints, and logs.
      state: Learner and counters at the last completed epoch.
      update: One optimizer attempt.
      legacy: Historical datasets for the auxiliary SSL objective.
      config: Replay and epoch settings.
      driver: CEM driver used for collection and evaluation.
      env: Shared game adapter.
      inference: Metadata stored with the driver weights.

    Returns:
      None. Progress is saved at completed epochs.
    """
    cfg = config.finetune
    replay = root / "replay"
    transitions = SlidingWindowDataset(replay, data_files=[], view="rl")
    windows = SlidingWindowDataset(
        replay,
        data_files=[],
        view="online_ssl",
        rollout_len=config.pretrain.rollout_len,
    )
    drives = cfg.drives_per_epoch
    for number in range(state.completed_epoch * drives):
        path = replay / f"episode-{number:08d}.h5"
        include_drive(path, number // drives + 1, transitions, windows)
    if len(transitions) != state.replay_cursor:
        raise ValueError("Replay transition count differs from the learner checkpoint.")

    driver_path = root / "driver.eqx"
    if state.completed_epoch:
        published = None
        if driver_path.is_file():
            with driver_path.open("rb") as file:
                published = json.loads(file.readline()).get("checkpoint_version")
        if published != state.completed_epoch:
            publish_driver(driver_path, state, inference)
        evaluate_if_due(root, env, driver, state, cfg)

    learner, attempt = state.learner, state.attempt
    for epoch in range(state.completed_epoch + 1, cfg.total_epochs + 1):
        use_cem = epoch > cfg.exploration_epochs
        snapshot = learner.models
        first_transition, first_window = len(transitions), len(windows)
        for number in range((epoch - 1) * drives, epoch * drives):
            path = replay / f"episode-{number:08d}.h5"
            if not path.is_file():
                episode = collect_drive(
                    env, driver, snapshot, use_cem, cfg.seed, number
                )
                metadata = {
                    "epoch": epoch,
                    "checkpoint_version": epoch - 1,
                    "drive": number,
                    "controller": "cem" if use_cem else "heuristic",
                }
                publish_episode(path, episode, metadata)
                log_json(
                    root / "collection.jsonl",
                    {**metadata, **episode.summary(cfg.gamma)},
                )
            include_drive(path, epoch, transitions, windows)

        recent_transition = first_transition if use_cem else len(transitions)
        recent_window = first_window if use_cem else len(windows)
        reports: list[dict[str, object]] = []
        for _ in range(cfg.optimizer_attempts_per_epoch):
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, attempt]))
            real = sample_replay(transitions, cfg.batch_size, rng, recent_transition)
            historical = (
                sample_legacy(legacy, cfg.batch_size, rng)[0] if legacy else None
            )
            online = (
                sample_replay(windows, cfg.batch_size, rng, recent_window)
                if len(windows)
                else None
            )
            learner, metrics = update(learner, real, historical, online)
            attempt += 1
            accepted = bool(metrics.pop("accepted"))
            if not accepted or attempt == 1 or attempt % cfg.log_every == 0:
                report: dict[str, object] = {
                    "epoch": epoch,
                    "attempt": attempt,
                    "accepted": accepted,
                }
                if accepted:
                    report.update(
                        {name: float(value) for name, value in metrics.items()}
                    )
                reports.append(report)

        state = RunState(learner, epoch, attempt, len(transitions))
        save_checkpoint(root / "learner.eqx", state)
        for report in reports:
            log_json(root / "updates.jsonl", report)
        publish_driver(driver_path, state, inference)
        evaluate_if_due(root, env, driver, state, cfg)
        print(
            json.dumps(
                {"epoch": epoch, "attempts": attempt, "updates": int(learner.updates)}
            ),
            flush=True,
        )


def run_online(
    args: argparse.Namespace,
    model: SubJepaConfig,
    config: TrainConfig,
    env_factory: Callable[[], Any] = get_tmrl_env,
) -> None:
    """Initialize or resume an epoch run and own the game session.

    Arguments:
      args: Run paths and initialization options.
      model: Encoder, predictor, and planner settings.
      config: Replay, loss, and epoch settings.
      env_factory: Factory for the reset/step game interface.

    Returns:
      None. The run writes learner state, driver weights, and metrics.
    """
    cfg = config.finetune
    if (
        min(
            cfg.batch_size,
            cfg.log_every,
            cfg.total_epochs,
            cfg.drives_per_epoch,
            cfg.optimizer_attempts_per_epoch,
            cfg.evaluation_races,
        )
        <= 0
    ):
        raise ValueError("Epoch, batch, update, and logging counts must be positive.")
    if not 0 <= cfg.exploration_epochs <= cfg.total_epochs:
        raise ValueError("Exploration epochs must fit the run.")
    if cfg.evaluation_interval is not None and cfg.evaluation_interval <= 0:
        raise ValueError("Evaluation interval must be positive when set.")
    if model.planner.sequence_len > config.pretrain.rollout_len:
        raise ValueError("Planning horizon must fit the SSL rollout horizon.")

    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run.json"
    if not args.resume and (
        manifest_path.exists() or any((root / "replay").glob("episode-*.h5"))
    ):
        raise FileExistsError(f"Use --resume or a new run directory: {root}")

    key, init_key = jax.random.split(jax.random.key(cfg.seed))
    models = make_driving_models(model, init_key)
    if not args.resume:
        models = (*load_checkpoint(args.initialize, models[:2]), models[2])
    optimizer, update = build_joint_learner(models, cfg, config.loss)
    learner = new_state(
        models,
        optimizer,
        (models[0], models[2]),
        key,
        model.encoder.latent_dim,
        config.loss,
    )
    legacy, validation, sources = legacy_splits(
        args.data_dir,
        config.pretrain.rollout_len,
        config.pretrain.val_ratio,
        config.pretrain.seed,
    )
    inference = {
        "version": DRIVER_CHECKPOINT_VERSION,
        "model": OmegaConf.to_container(model, resolve=True),
        "action_table": ACTION_GRID_NP.tolist(),
        "preprocessing": PREPROCESSING,
        "gamma": cfg.gamma,
    }
    manifest = json.loads(
        json.dumps(
            {
                "inference": inference,
                "reward": REWARD_DEFINITION,
                "map_name": game_config.agent.map_name,
                "map_uid": game_config.agent.map_uid,
                "training": OmegaConf.to_container(config, resolve=True),
                "legacy_sources": sources,
                "legacy_train": [group.episode_keys for group in legacy],
                "legacy_validation": [group.episode_keys for group in validation],
            }
        )
    )
    state = RunState(learner, 0, 0, 0)
    if args.resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["inference"]["version"] != DRIVER_CHECKPOINT_VERSION:
            raise ValueError("Start a scalar-value run from a pretrained world model.")
        if any(saved.get(name) != value for name, value in manifest.items()):
            raise ValueError(
                "Replay configuration, reward, map or legacy split differs."
            )
        state = load_checkpoint(root / "learner.eqx", state)
    else:
        projector_path = args.projectors or args.initialize.with_name("projectors.eqx")
        if args.projectors or projector_path.is_file():
            state = state._replace(
                learner=learner._replace(
                    bases=load_checkpoint(projector_path, learner.bases)
                )
            )
        manifest["initial_world_model"] = str(args.initialize.resolve())
        manifest["projectors"] = (
            str(projector_path.resolve()) if projector_path.is_file() else None
        )
        save_checkpoint(root / "learner.eqx", state)
        write_text_atomic(manifest_path, json.dumps(manifest, indent=2))

    driver = MPCDriver(CEMPlanner(model.planner, cfg.gamma), cfg.seed)
    driver.warmup(state.learner.models)
    apply_online_rl_patches()
    env = DrivingEnvironment(env_factory())
    try:
        run_epochs(
            root,
            state,
            update,
            [] if cfg.ablation == "rl-only" else legacy,
            config,
            driver,
            env,
            inference,
        )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--initialize", type=Path)
    initial.add_argument("--resume", action="store_true")
    parser.add_argument("--projectors", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_online(
        args, load_config(str(args.model_config)), load_train_config(args.config)
    )


if __name__ == "__main__":
    main()
