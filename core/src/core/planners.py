"""Categorical CEM with survival-weighted rewards and continuation value."""

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from core.actions import ACTION_GRID_JAX, NUM_ACTIONS
from core.config import PlannerConfig
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
        score_candidates = eqx.filter_vmap(self.score, in_axes=(None, None, None, 0))

        def sample(probs, draw_key):
            return sample_actions(
                jnp.broadcast_to(probs, (self.samples, *probs.shape)), draw_key
            )

        def select_elites(probs, candidates, scores):
            elite_scores, indices = jax.lax.top_k(scores, self.elites)
            elites = candidates[indices]
            frequency = jax.nn.one_hot(elites, NUM_ACTIONS).mean(axis=0)
            probs = uniform_mix(
                (1 - self.elite_weight) * probs + self.elite_weight * frequency
            )
            return probs, elites, elite_scores

        keys = jax.random.split(key, self.iterations)
        candidates = sample(shifted_probs, keys[0])
        shifted_best = jnp.concatenate((previous.actions[1:], candidates[0, -1:]))
        candidates = candidates.at[0].set(
            jnp.where(previous.valid, shifted_best, candidates[0])
        )
        initial = select_elites(
            shifted_probs,
            candidates,
            score_candidates(predictor, head, root, candidates),
        )

        def iteration(carry, draw_key):
            probs, old_elites, old_scores = carry
            # A full draw preserves the previous planner's seeded proposals.
            fresh = sample(probs, draw_key)[self.elites :]
            scores = score_candidates(predictor, head, root, fresh)
            return select_elites(
                probs,
                jnp.concatenate((old_elites, fresh)),
                jnp.concatenate((old_scores, scores)),
            ), None

        (probs, elites, _), _ = jax.lax.scan(iteration, initial, keys[1:])
        return Plan(elites[0], probs, jnp.asarray(True))
