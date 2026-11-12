"""Run synchronous CEM inference and record Trackmania drives."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from core.actions import ACTION_GRID_NP
from core.config import (
    ACTION_HISTORY_FEATURES,
    IMG_HIST_LEN,
    IMG_SIZE,
    PIXEL_SCALE,
    TELEMETRY_SCALE,
    SubJepaConfig,
)
from core.dynamics import MLPPredictor, RewardValueHead
from core.encoders import DrivingModels, ViTEncoder
from core.planners import CEMPlanner, Plan
from omegaconf import OmegaConf

from win_client.game_session import MAX_EPISODE_STEPS, DrivingEnvironment, Observation

DRIVER_CHECKPOINT_VERSION = 2
PREPROCESSING = {
    "frames": IMG_HIST_LEN,
    "image_size": IMG_SIZE,
    "pixel_scale": PIXEL_SCALE,
    "telemetry_scale": list(TELEMETRY_SCALE),
    "past_controls": ACTION_HISTORY_FEATURES // 3,
}


def make_driving_models(config: SubJepaConfig, key) -> DrivingModels:
    if config.encoder.latent_dim != config.predictor.latent_dim:
        raise ValueError("The encoder and predictor must use the same learned width.")
    encoder_key, predictor_key, head_key = jax.random.split(key, 3)
    return (
        ViTEncoder(config.encoder, encoder_key),
        MLPPredictor(config.predictor, predictor_key),
        RewardValueHead(config.encoder.latent_dim, head_key),
    )


def load_driver(path: Path, seed: int):
    with path.open("rb") as file:
        metadata = json.loads(file.readline())
        if (
            metadata["version"] != DRIVER_CHECKPOINT_VERSION
            or metadata["action_table"] != ACTION_GRID_NP.tolist()
        ):
            raise ValueError("Unsupported driving checkpoint action table or version.")
        if metadata["preprocessing"] != PREPROCESSING:
            raise ValueError("Unsupported driving checkpoint preprocessing.")
        config = cast(
            SubJepaConfig,
            OmegaConf.merge(OmegaConf.structured(SubJepaConfig), metadata["model"]),
        )
        template = make_driving_models(config, jax.random.key(seed))
        models = eqx.tree_deserialise_leaves(file, template)
    return MPCDriver(CEMPlanner(config.planner, metadata["gamma"]), seed), models


def _encode(encoder: ViTEncoder, observation: Observation):
    return encoder(
        {
            "screen": jnp.asarray(observation["screen"], dtype=jnp.float32)
            / PIXEL_SCALE,
            "telemetry": jnp.asarray(observation["telemetry"], dtype=jnp.float32),
        }
    )


@eqx.filter_jit
def search_action(
    planner: CEMPlanner, models: DrivingModels, obs: Observation, plan: Plan, key
):
    encoder, predictor, head = models
    key, search_key = jax.random.split(key)
    next_plan = planner(predictor, head, _encode(encoder, obs), plan, search_key)
    return next_plan.actions[0], next_plan, key


class MPCDriver:
    def __init__(self, planner: CEMPlanner, seed: int) -> None:
        self.planner = planner
        self.seed = seed
        self.key = jax.random.key(seed)
        self.empty = planner.empty_plan()
        self.plan = self.empty

    def reset(self, episode_seed: int | None = None) -> None:
        self.plan = self.empty
        if episode_seed is not None:
            self.key = jax.random.fold_in(jax.random.key(self.seed), episode_seed)

    def select(self, models: DrivingModels, observation: Observation) -> int:
        action, self.plan, self.key = search_action(
            self.planner, models, observation, self.plan, self.key
        )
        return int(action)

    def warmup(self, models: DrivingModels) -> None:
        observation = {
            "screen": np.zeros((IMG_HIST_LEN, IMG_SIZE, IMG_SIZE), dtype=np.uint8),
            "telemetry": np.zeros(3 + ACTION_HISTORY_FEATURES, dtype=np.float32),
        }
        jax.block_until_ready(
            search_action(
                self.planner, models, observation, self.empty, jax.random.key(0)
            )
        )


@dataclass
class Episode:
    observations: list[Observation]
    actions: list[int]
    rewards: list[float]
    done: bool
    reason: str
    inference_ms: list[float]
    action_errors: list[float]
    elapsed_seconds: float

    def discounted_return(self, gamma: float) -> float:
        return float(np.dot(self.rewards, gamma ** np.arange(len(self.rewards))))

    def summary(self, gamma: float) -> dict[str, object]:
        steps = len(self.actions)
        return {
            "steps": steps,
            "reason": self.reason,
            "finished": self.reason == "done",
            "finish_seconds": self.elapsed_seconds if self.reason == "done" else None,
            "elapsed_seconds": self.elapsed_seconds,
            "done": self.done,
            "return": self.discounted_return(gamma),
            "inference_p99_ms": (
                float(np.percentile(self.inference_ms, 99))
                if self.inference_ms
                else None
            ),
            "action_mismatch_rate": (
                float(np.mean(np.asarray(self.action_errors) > 0.05))
                if self.action_errors
                else None
            ),
        }


def drive_episode(
    env: DrivingEnvironment,
    select: Callable[[Observation], int],
    *,
    max_steps: int = MAX_EPISODE_STEPS,
    keep_observations: bool = True,
) -> Episode:
    observation = env.reset()
    started = time.perf_counter()
    result = Episode(
        [_record_observation(observation)] if keep_observations else [],
        [],
        [],
        False,
        "budget",
        [],
        [],
        0.0,
    )
    while len(result.actions) < max_steps:
        inference_start = time.perf_counter()
        action = select(observation)
        result.inference_ms.append(1000 * (time.perf_counter() - inference_start))
        step = env.step(action)
        if step.observation is None:
            result.reason = "missing_observation"
            break
        observation = step.observation
        if step.action_id is None:
            if keep_observations:
                result.observations[0] = _record_observation(observation)
        else:
            if keep_observations:
                result.observations.append(_record_observation(observation))
            result.actions.append(step.action_id)
            result.rewards.append(step.reward)
        if step.action_error is not None:
            result.action_errors.append(step.action_error)
        if step.reason is not None:
            result.done = step.done
            result.reason = step.reason
            break
    result.elapsed_seconds = time.perf_counter() - started
    return result


def _record_observation(observation: Observation) -> Observation:
    return {
        "screen": observation["screen"][-1:].copy(),
        "telemetry": observation["telemetry"],
    }
