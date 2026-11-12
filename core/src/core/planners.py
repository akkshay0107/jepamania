"""Categorical CEM with survival-weighted rewards and continuation value."""

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from core.actions import ACTION_GRID_JAX
from core.config import NUM_ACTIONS, PlannerConfig
from core.dynamics import MLPPredictor, RewardValueHead

if TYPE_CHECKING:
    from typing import Annotated as Bool
    from typing import Annotated as Float
    from typing import Annotated as Int
else:
    from jaxtyping import Bool, Float, Int

UNIFORM_MIX = 0.05


class Plan(NamedTuple):
    actions: Int[Array, "horizon"]
    probabilities: Float[Array, "horizon actions"]
    valid: Bool[Array, ""]


def uniform_mix(probs: Float[Array, "... actions"]) -> Float[Array, "... actions"]:
    return (1 - UNIFORM_MIX) * probs + UNIFORM_MIX / NUM_ACTIONS


def sample_actions(probs: Float[Array, "... actions"], key: PRNGKeyArray):
    # One uniform per draw suffices for a positive categorical distribution.
    draws = jax.random.uniform(key, probs.shape[:-1])
    cumulative = jnp.cumsum(probs[..., :-1], axis=-1)
    return jnp.sum(draws[..., None] >= cumulative, axis=-1, dtype=jnp.int32)


class CEMPlanner(eqx.Module):
    horizon: int = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    samples: int = eqx.field(static=True)
    elites: int = eqx.field(static=True)
    elite_weight: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)

    def __init__(self, config: PlannerConfig, gamma: float) -> None:
        self.horizon = config.sequence_len
        self.iterations = config.cem_iters
        self.samples = config.cem_samples
        self.elites = config.cem_elites
        self.elite_weight = config.cem_alpha
        self.gamma = gamma

    def __check_init__(self) -> None:
        if min(self.horizon, self.iterations, self.elites) <= 0:
            raise ValueError(
                "CEM horizon, iterations, and elite count must be positive."
            )
        if self.samples <= self.elites:
            raise ValueError("CEM needs more samples than retained elites.")
        if not 0 < self.elite_weight <= 1 or not 0 < self.gamma <= 1:
            raise ValueError("CEM elite weight and discount must be in (0, 1].")

    def empty_plan(self) -> Plan:
        return Plan(
            jnp.zeros(self.horizon, dtype=jnp.int32),
            jnp.full((self.horizon, NUM_ACTIONS), 1 / NUM_ACTIONS, dtype=jnp.float32),
            jnp.asarray(False),
        )

    def score(
        self,
        predictor: MLPPredictor,
        head: RewardValueHead,
        root: Float[Array, "state_dim"],
        actions: Int[Array, "horizon"],
    ) -> Float[Array, ""]:
        def step(carry, action):
            state, weight, score = carry
            rewards, logits, _ = head(state)
            score = score + weight * rewards[action]
            weight = weight * self.gamma * jax.nn.sigmoid(-logits[action])
            return (predictor(state, ACTION_GRID_JAX[action]), weight, score), None

        (tail, weight, score), _ = jax.lax.scan(
            step,
            (root, jnp.ones((), dtype=root.dtype), jnp.zeros((), dtype=root.dtype)),
            actions,
        )
        return score + weight * head.state_value(tail)

    def __call__(
        self,
        predictor: MLPPredictor,
        head: RewardValueHead,
        root: Float[Array, "state_dim"],
        previous: Plan,
        key: PRNGKeyArray,
    ) -> Plan:
        shifted_probs = jnp.concatenate(
            (previous.probabilities[1:], jnp.full((1, NUM_ACTIONS), 1 / NUM_ACTIONS))
        )
        starting_elites = jnp.zeros((self.elites, self.horizon), dtype=jnp.int32)

        def iteration(carry, item):
            probs, old_elites = carry
            index, search_key = item
            candidates = sample_actions(
                jnp.broadcast_to(probs, (self.samples, *probs.shape)), search_key
            )
            shifted_best = jnp.concatenate((previous.actions[1:], candidates[0, -1:]))
            candidates = candidates.at[0].set(
                jnp.where(previous.valid, shifted_best, candidates[0])
            )
            candidates = candidates.at[: self.elites].set(
                jnp.where(index > 0, old_elites, candidates[: self.elites])
            )
            scores = eqx.filter_vmap(self.score, in_axes=(None, None, None, 0))(
                predictor, head, root, candidates
            )
            _, elite_indices = jax.lax.top_k(scores, self.elites)
            elites = candidates[elite_indices]
            elite_frequency = jax.nn.one_hot(elites, NUM_ACTIONS).mean(axis=0)
            probs = uniform_mix(
                (1 - self.elite_weight) * probs + self.elite_weight * elite_frequency
            )
            return (probs, elites), None

        (probs, elites), _ = jax.lax.scan(
            iteration,
            (shifted_probs, starting_elites),
            (jnp.arange(self.iterations), jax.random.split(key, self.iterations)),
        )
        return Plan(elites[0], probs, jnp.asarray(True))
