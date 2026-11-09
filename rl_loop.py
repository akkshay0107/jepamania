"""Collect on the game thread and learn from completed HDF5 episodes in parallel."""

import argparse
import json
import queue
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from core.actions import ACTION_GRID_NP
from core.config import IMG_HIST_LEN, SubJepaConfig, load_config
from core.encoders import DrivingModels, load_world_model
from core.planners import CEMPlanner, rollout_reward
from omegaconf import OmegaConf
from train.config import TrainConfig, load_train_config
from train.dataloader import SlidingWindowDataset
from train.finetune import OnlineState, build_joint_learner, make_joint_objective
from train.pretrain import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_TRAIN_CONFIG,
    alpha_for,
    encode,
    legacy_splits,
    load_checkpoint,
    log_json,
    make_gradient_probe,
    new_state,
    sample_legacy,
    save_checkpoint,
)
from train.rollout_episodes import publish_episode
from win_client.env_patches import apply_online_rl_patches
from win_client.game_session import (
    MAX_EPISODE_STEPS,
    REWARD_DEFINITION,
    DrivingEnvironment,
)
from win_client.mpc_driver import (
    PREPROCESSING,
    Controller,
    Episode,
    MPCDriver,
    drive_episode,
    make_driving_models,
)
from win_client.settings import cfg as game_config
from win_client.utils import get_tmrl_env


class Snapshot(NamedTuple):
    models: DrivingModels
    updates: int


class RunState(NamedTuple):
    learner: OnlineState
    attempt: int
    replay_cursor: int
    update_credit: int


def replay_shards(root: Path) -> list[Path]:
    return sorted(root.glob("episode-*.h5"))


def replay_cursor(path: Path) -> int:
    return int(path.stem.removeprefix("episode-"))


def exploration_action(decision: int, fill: int, seed: int) -> int | None:
    filling = decision < fill
    phase_decision = decision if filling else decision - fill
    block = phase_decision // 4
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(filling), block]))
    epsilon = 1.0 if filling else 0.2 - 0.15 * min(block * 4 / 100_000, 1.0)
    return int(rng.integers(len(ACTION_GRID_NP))) if rng.random() < epsilon else None


class OnlineLearner:
    """Own the learner state and publish whole model versions at update boundaries."""

    def __init__(
        self,
        root: Path,
        config: TrainConfig,
        manifest: dict[str, Any],
        state: RunState,
        update: Callable,
        legacy: list[SlidingWindowDataset],
    ) -> None:
        self.root = root
        self.config = config
        self.manifest = manifest
        self.state = state
        self.update = update
        self.legacy = [] if config.finetune.ablation == "rl-only" else legacy
        self.probe = make_gradient_probe(
            make_joint_objective(config.finetune, config.loss),
            ("prediction", "weighted_regularization", "reward", "td"),
        )
        probe_rng = np.random.default_rng(config.finetune.seed + 1)
        self.probe_legacy = (
            sample_legacy(self.legacy, min(4, config.finetune.batch_size), probe_rng)[0]
            if self.legacy
            else None
        )
        self.incoming: queue.Queue[Path | None] = queue.Queue()
        self.latest = Snapshot(state.learner.models, int(state.learner.updates))

    def _save(self, state: RunState) -> None:
        save_checkpoint(self.root / "learner.eqx", state)
        save_checkpoint(
            self.root / "driver.eqx",
            state.learner.models,
            metadata={
                **self.manifest["inference"],
                "updates": int(state.learner.updates),
            },
        )
        self.latest = Snapshot(state.learner.models, int(state.learner.updates))

    def run(self) -> None:
        cfg = self.config.finetune
        rollout_len = self.config.pretrain.rollout_len
        replay = self.root / "replay"
        transitions = SlidingWindowDataset(replay, data_files=[], view="rl")
        windows = SlidingWindowDataset(
            replay, data_files=[], view="online_ssl", rollout_len=rollout_len
        )
        state = self.state
        self._save(state)
        seen = set()
        closing = False
        rejected = 0

        def include(path: Path, *, restore: bool = False) -> None:
            nonlocal state
            if path in seen:
                return
            with h5py.File(path, "r") as file:
                if file.attrs.get("reward_convention") != REWARD_DEFINITION:
                    raise ValueError(f"Reward definition differs in {path}.")
            before = len(transitions)
            transitions.include_online_shard(path)
            windows.include_online_shard(path)
            after = len(transitions)
            if after != replay_cursor(path):
                raise ValueError(
                    f"Replay sequence or transition count differs in {path}."
                )
            seen.add(path)
            if restore:
                return
            gained = max(0, after - cfg.fill_decisions) - max(
                0, before - cfg.fill_decisions
            )
            if before < cfg.fill_decisions <= after:
                gained += 1
            state = state._replace(
                replay_cursor=after,
                update_credit=min(MAX_EPISODE_STEPS, state.update_credit + gained),
            )

        for path in replay_shards(replay):
            include(path, restore=replay_cursor(path) <= state.replay_cursor)
        if len(transitions) < state.replay_cursor:
            raise ValueError("Online replay is missing episodes used by the learner.")

        while not closing or state.update_credit > 0:
            # Drain metadata before sampling so a slow learner sees fresh replay.
            while not closing:
                try:
                    incoming = self.incoming.get(block=state.update_credit == 0)
                except queue.Empty:
                    break
                if incoming is None:
                    closing = True
                else:
                    include(incoming)

            if state.update_credit == 0:
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence([cfg.seed, state.attempt])
            )
            indices = rng.integers(len(transitions), size=cfg.batch_size)
            real = transitions.read_batch(indices)
            legacy, legacy_ids = (
                sample_legacy(self.legacy, cfg.batch_size, rng)
                if self.legacy
                else (None, [])
            )
            online_ids = (
                rng.integers(len(windows), size=cfg.batch_size)
                if len(windows)
                else None
            )
            online = None if online_ids is None else windows.read_batch(online_ids)
            next_state, metrics = self.update(state.learner, real, legacy, online)
            state = RunState(
                next_state,
                state.attempt + 1,
                state.replay_cursor,
                state.update_credit - 1,
            )
            if not bool(metrics["accepted"]):
                rejected += 1
                log_json(
                    self.root / "updates.jsonl",
                    {
                        "attempt": state.attempt,
                        "replay_cursor": state.replay_cursor,
                        "rejected_transitions": indices.tolist(),
                        "legacy_indices": legacy_ids,
                        "online_indices": None
                        if online_ids is None
                        else online_ids.tolist(),
                    },
                )
                if rejected >= 8:
                    self._save(state)
                    raise FloatingPointError(
                        "Eight consecutive joint updates were rejected."
                    )
                continue
            rejected = 0
            updates = int(next_state.updates)
            if updates in (1, 2) or updates % 250 == 0:
                self._save(state)
            self._log_update(state, metrics, legacy, online, transitions, windows)
        self._save(state)
        self.state = state

    def _log_update(self, state, metrics, legacy, online, transitions, windows) -> None:
        cfg = self.config.finetune
        updates = int(state.learner.updates)
        sources = {}
        for name, batch in (("legacy", legacy), ("online", online)):
            if cfg.ablation == "rl-only":
                continue
            if batch is not None:
                size, steps = batch["actions_seq"].shape[:2]
                sources[name] = {
                    "B": size,
                    "K": steps,
                    "N": size * steps,
                    "alpha": alpha_for(batch, self.config.loss.reg_weight),
                }
        report = {
            "attempt": state.attempt,
            "updates": updates,
            "replay_cursor": state.replay_cursor,
            "joint_decisions": state.replay_cursor - cfg.fill_decisions,
            "updates_per_joint_decision": updates
            / max(1, state.replay_cursor - cfg.fill_decisions),
            "ssl": sources,
            "rl_B": cfg.batch_size,
        }
        if updates == 1 or updates % cfg.log_every == 0:
            report.update({name: float(value) for name, value in metrics.items()})
        if updates % cfg.log_every == 0:
            probe_size = min(4, cfg.batch_size)
            real_probe = transitions.read_batch(
                np.arange(min(probe_size, len(transitions)))
            )
            online_probe = (
                windows.read_batch(np.arange(min(probe_size, len(windows))))
                if len(windows)
                else None
            )
            gradients = self.probe(
                state.learner,
                jax.random.key(cfg.seed + 1),
                real_probe,
                self.probe_legacy,
                online_probe,
            )
            report["encoder_gradients"] = {
                name: float(value) for name, value in gradients.items()
            }
        log_json(self.root / "updates.jsonl", report)


@eqx.filter_jit
def forecast_reward(models: DrivingModels, frames, telemetry, actions, gamma: float):
    encoder, predictor, head = models
    roots = encode(encoder, frames, telemetry)
    returns, _ = jax.vmap(rollout_reward, in_axes=(None, None, 0, 0, None))(
        predictor, head, roots, jnp.asarray(actions), gamma
    )
    return returns


def check_reward_forecasts(models, episode: Episode, horizon: int, gamma: float):
    if len(episode.actions) < horizon:
        return None
    roots = np.linspace(0, len(episode.actions) - horizon, 24, dtype=np.intp)
    rows = roots[:, None] + np.arange(horizon)
    history = np.maximum(roots[:, None] + np.arange(-IMG_HIST_LEN + 1, 1), 0)
    frames = np.stack(
        [
            np.stack([episode.observations[index]["screen"][-1] for index in row])
            for row in history
        ]
    )
    telemetry = np.stack([episode.observations[index]["telemetry"] for index in roots])
    actions = np.asarray(episode.actions, dtype=np.int32)[rows]
    actual = np.asarray(episode.rewards)[rows] @ (gamma ** np.arange(horizon))
    predicted = np.asarray(forecast_reward(models, frames, telemetry, actions, gamma))
    if not np.isfinite(predicted).all():
        return None
    return {
        "mae": float(np.mean(np.abs(predicted - actual))),
        "actual_mean": float(np.mean(actual)),
        "predicted_mean": float(np.mean(predicted)),
    }


def evaluate(
    driver: MPCDriver,
    env: DrivingEnvironment,
    snapshot: Snapshot,
    decisions: int,
    config: TrainConfig,
) -> tuple[dict[str, Any], bool]:
    cfg = config.finetune
    results: dict[str, list[dict[str, Any]]] = {"q": [], "mpc": []}
    checks = []
    can_search = False
    for pair in range(cfg.evaluation_episodes):
        order: tuple[Controller, Controller] = (
            ("mpc", "q")
            if can_search and (decisions // cfg.evaluation_every + pair) % 2
            else ("q", "mpc")
        )
        for controller in order:
            if controller == "mpc" and not can_search:
                continue
            driver.reset(2 * (decisions + pair) + 1)

            def select(obs, mode: Controller = controller):
                return driver.select(snapshot.models, obs, mode)

            episode = drive_episode(
                env,
                select,
                keep_observations=controller == "q",
            )
            results[controller].append(episode.summary(cfg.gamma))
            if controller == "q":
                check = check_reward_forecasts(
                    snapshot.models, episode, driver.planner.horizon, cfg.gamma
                )
                checks.append(check)
                # Keep the initial search trial bounded by a held-out reward check.
                can_search = check is not None and check["mae"] <= 0.1
    q, mpc = results["q"], results["mpc"]
    comparable = len(q) == len(mpc) and bool(q) and sum(e["finished"] for e in mpc) > 0
    improved = comparable and (
        sum(e["finished"] for e in mpc),
        np.mean([e["return"] for e in mpc]),
    ) > (sum(e["finished"] for e in q), np.mean([e["return"] for e in q]))
    return {
        "decisions": decisions,
        "updates": snapshot.updates,
        "episodes": results,
        "reward_forecast_checks": checks,
    }, bool(improved)


def collect_trial(
    env: DrivingEnvironment,
    driver: MPCDriver,
    learner: OnlineLearner,
    storage: ThreadPoolExecutor,
    worker: Future,
    decisions: int,
    root: Path,
) -> int:
    cfg = learner.config.finetune
    evaluation_boundary = (decisions // cfg.evaluation_every + 1) * cfg.evaluation_every
    pending: list[Future] = []
    use_mpc = False
    empty = 0

    def store(path, episode, metadata):
        learner.incoming.put(publish_episode(path, episode, metadata))
        log_json(root / "collection.jsonl", {**metadata, **episode.summary(cfg.gamma)})

    while decisions < cfg.decisions:
        if worker.done():
            worker.result()
            raise RuntimeError("The online learner stopped unexpectedly.")
        while pending and (pending[0].done() or len(pending) >= 4):
            pending.pop(0).result()
        snapshot = learner.latest
        driver.reset(2 * decisions)
        controller: Controller = "mpc" if use_mpc else "q"
        use_mpc = False
        start = decisions
        tick = start
        block_end = start
        held_action = None

        def select(obs):
            nonlocal tick, block_end, held_action
            if tick == block_end:
                held_action = exploration_action(tick, cfg.fill_decisions, cfg.seed)
                phase_tick = (
                    tick if tick < cfg.fill_decisions else tick - cfg.fill_decisions
                )
                block_end = tick + 4 - phase_tick % 4
                if held_action is not None:
                    driver.reset()
            tick += 1
            if held_action is not None:
                return held_action
            return driver.select(snapshot.models, obs, controller)

        limit = min(
            MAX_EPISODE_STEPS,
            cfg.decisions - decisions,
            evaluation_boundary - decisions,
        )
        if decisions < cfg.fill_decisions:
            limit = min(limit, cfg.fill_decisions - decisions)
        episode = drive_episode(env, select, max_steps=limit)
        if not episode.actions:
            empty += 1
            if empty >= 3:
                raise RuntimeError("Three game resets produced no complete transition.")
            continue
        empty = 0
        decisions += len(episode.actions)
        metadata = {
            "start_decision": start,
            "end_decision": decisions,
            "joint_decisions": max(0, decisions - cfg.fill_decisions),
            "held_action": exploration_action(
                decisions - 1, cfg.fill_decisions, cfg.seed
            ),
            "block_offset": (
                decisions
                if decisions <= cfg.fill_decisions
                else decisions - cfg.fill_decisions
            )
            % 4,
            "updates": snapshot.updates,
            "controller": "fill" if start < cfg.fill_decisions else controller,
        }
        pending.append(
            storage.submit(
                store,
                root / "replay" / f"episode-{decisions:06d}.h5",
                episode,
                metadata,
            )
        )
        if decisions == evaluation_boundary or decisions == cfg.decisions:
            if learner.latest.updates:
                report, use_mpc = evaluate(
                    driver, env, learner.latest, decisions, learner.config
                )
                log_json(root / "evaluation.jsonl", report)
            elif decisions < cfg.decisions:
                log_json(
                    root / "evaluation.jsonl",
                    {
                        "decisions": decisions,
                        "updates": 0,
                        "reason": "awaiting_joint_update",
                    },
                )
            evaluation_boundary += cfg.evaluation_every

    for item in pending:
        item.result()
    return decisions


def run_online(
    args: argparse.Namespace,
    model: SubJepaConfig,
    config: TrainConfig,
    env_factory: Callable[[], Any] = get_tmrl_env,
) -> None:
    """Run one decision-budgeted trial, restoring durable state when requested.

    Arguments:
      args: Run directory, recording source, and checkpoint options.
      model: Encoder, predictor, and planner settings.
      config: Offline split, loss, and online training settings.
      env_factory: Factory for the ordinary reset/step game interface.

    Returns:
      None. Progress, model snapshots, and evaluation summaries are saved to disk.
    """
    cfg = config.finetune
    if not 0 < cfg.fill_decisions < cfg.decisions:
        raise ValueError(
            "Fill must be positive and smaller than the online decision budget."
        )
    if (
        min(
            cfg.batch_size, cfg.log_every, cfg.evaluation_every, cfg.evaluation_episodes
        )
        <= 0
    ):
        raise ValueError(
            "Batch size, logging and evaluation intervals must be positive."
        )
    if model.planner.sequence_len > config.pretrain.rollout_len:
        raise ValueError("Planning horizon must fit the SSL rollout horizon.")

    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    key, init_key, world_key = jax.random.split(jax.random.key(cfg.seed), 3)
    models = make_driving_models(model, init_key)
    if not args.resume:
        world = load_world_model(
            args.initialize, world_key, model.encoder, model.predictor
        )
        models = (*world, models[2])
    optimizer, update = build_joint_learner(models, cfg, config.loss)
    state = new_state(
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
    manifest = json.loads(
        json.dumps(
            {
                "inference": {
                    "version": 1,
                    "model": OmegaConf.to_container(model, resolve=True),
                    "action_table": ACTION_GRID_NP.tolist(),
                    "preprocessing": PREPROCESSING,
                    "gamma": cfg.gamma,
                },
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
    manifest_path = root / "run.json"
    if args.resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {name: saved[name] for name in manifest} != manifest:
            raise ValueError(
                "Replay configuration, reward, map or legacy split differs."
            )
        restored = load_checkpoint(root / "learner.eqx", RunState(state, 0, 0, 0))
    else:
        if manifest_path.exists():
            raise FileExistsError(f"Use --resume or a new run directory: {root}")
        projector_path = args.projectors or args.initialize.with_name("projectors.eqx")
        if args.projectors or projector_path.is_file():
            state = state._replace(bases=load_checkpoint(projector_path, state.bases))
        manifest["initial_world_model"] = str(args.initialize.resolve())
        manifest["projectors"] = (
            str(projector_path.resolve()) if projector_path.is_file() else None
        )
        restored = RunState(state, 0, 0, 0)
        save_checkpoint(root / "learner.eqx", restored)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    shards = replay_shards(root / "replay")
    decisions = replay_cursor(shards[-1]) if shards else 0
    learner = OnlineLearner(root, config, manifest, restored, update, legacy)
    driver = MPCDriver(CEMPlanner(model.planner, cfg.gamma), cfg.seed)
    driver.warmup(learner.latest.models)
    apply_online_rl_patches()
    env = DrivingEnvironment(env_factory())
    try:
        with (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="learner") as learning,
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay") as storage,
        ):
            future = learning.submit(learner.run)
            try:
                decisions = collect_trial(
                    env, driver, learner, storage, future, decisions, root
                )
            finally:
                storage.shutdown(wait=True)
                learner.incoming.put(None)
                future.result()

        history = root / "evaluation.jsonl"
        evaluations = (
            []
            if not history.exists()
            else [
                json.loads(line)["decisions"]
                for line in history.read_text().splitlines()
            ]
        )
        if learner.latest.updates and decisions not in evaluations:
            report, _ = evaluate(driver, env, learner.latest, decisions, config)
            log_json(history, report)
        print(
            json.dumps(
                {
                    "decisions": decisions,
                    "updates": learner.latest.updates,
                    "updates_per_joint_decision": learner.latest.updates
                    / max(1, decisions - cfg.fill_decisions),
                }
            ),
            flush=True,
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
