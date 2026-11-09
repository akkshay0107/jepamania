"""Categorical CEM with learned interval rewards and a terminal Q estimate."""

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from core.actions import ACTION_GRID_JAX
from core.config import NUM_ACTIONS, PlannerConfig
from core.dynamics import MLPPredictor, RewardQHead, search_transition

if TYPE_CHECKING:
    from typing import Annotated as Bool
    from typing import Annotated as Float
    from typing import Annotated as Int
else:
    from jaxtyping import Bool, Float, Int

GUIDED_SAMPLES = 32
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


def _discounted_sum(rewards: Float[Array, "horizon ..."], gamma: float):
    discounts = gamma ** jnp.arange(rewards.shape[0], dtype=jnp.float32)
    return discounts @ rewards


def rollout_reward(
    predictor: MLPPredictor,
    head: RewardQHead,
    root: Float[Array, "state_dim"],
    actions: Int[Array, "horizon"],
    gamma: float,
) -> tuple[Float[Array, ""], Float[Array, "state_dim"]]:
    def step(state, action):
        rewards, _ = head.for_search(state)
        return search_transition(predictor, state, ACTION_GRID_JAX[action]), rewards[
            action
        ]

    tail, rewards = jax.lax.scan(step, root, actions)
    return _discounted_sum(rewards, gamma), tail


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
        if self.samples < GUIDED_SAMPLES + 2 + self.elites:
            raise ValueError(
                "CEM needs slots for guided sequences and retained elites."
            )
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
        head: RewardQHead,
        root: Float[Array, "state_dim"],
        actions: Int[Array, "horizon"],
    ) -> Float[Array, ""]:
        reward, tail = rollout_reward(predictor, head, root, actions, self.gamma)
        _, tail_q = head.for_search(tail)
        return reward + self.gamma**self.horizon * jnp.max(tail_q)

    def _guided(self, predictor, head, root, key):
        states = jnp.broadcast_to(root, (GUIDED_SAMPLES + 1, root.shape[0]))

        def step(states, step_key):
            rewards, q = jax.vmap(head.for_search)(states)
            scale = jnp.maximum(jnp.std(q, axis=-1, keepdims=True), 0.1)
            probs = uniform_mix(jax.nn.softmax((q - q.mean(-1, keepdims=True)) / scale))
            actions = sample_actions(probs, step_key)
            actions = actions.at[0].set(jnp.argmax(q[0]))
            next_states = jax.vmap(search_transition, in_axes=(None, 0, 0))(
                predictor, states, ACTION_GRID_JAX[actions]
            )
            selected_rewards = jnp.take_along_axis(rewards, actions[:, None], axis=-1)[
                :, 0
            ]
            return next_states, (actions, selected_rewards)

        tail, (actions, rewards) = jax.lax.scan(
            step, states, jax.random.split(key, self.horizon)
        )
        _, tail_q = jax.vmap(head.for_search)(tail)
        scores = _discounted_sum(
            rewards, self.gamma
        ) + self.gamma**self.horizon * jnp.max(tail_q, axis=-1)
        return actions.T, scores

    def __call__(
        self,
        predictor: MLPPredictor,
        head: RewardQHead,
        root: Float[Array, "state_dim"],
        previous: Plan,
        key: PRNGKeyArray,
    ) -> Plan:
        shifted_probs = jnp.concatenate(
            (previous.probabilities[1:], jnp.full((1, NUM_ACTIONS), 1 / NUM_ACTIONS))
        )
        random_count = self.samples - GUIDED_SAMPLES - 1
        starting_elites = jnp.zeros((self.elites, self.horizon), dtype=jnp.int32)

        def iteration(carry, item):
            probs, best, best_score, old_elites = carry
            index, iter_key = item
            guided_key, search_key = jax.random.split(iter_key)
            guided, guided_scores = self._guided(predictor, head, root, guided_key)
            frequencies = jax.nn.one_hot(guided[1:], NUM_ACTIONS).mean(axis=0)
            initial = uniform_mix(
                jnp.where(
                    previous.valid, 0.5 * (frequencies + shifted_probs), frequencies
                )
            )
            probs = jnp.where(index == 0, initial, probs)
            random = sample_actions(
                jnp.broadcast_to(probs, (random_count, *probs.shape)), search_key
            )
            shifted_best = jnp.concatenate((previous.actions[1:], guided[0, -1:]))
            reserved = jnp.where(previous.valid, shifted_best, random[0])
            retained = jnp.where(index > 0, old_elites, random[1 : self.elites + 1])
            search_candidates = jnp.concatenate(
                (reserved[None], retained, random[self.elites + 1 :])
            )
            search_scores = jax.vmap(self.score, in_axes=(None, None, None, 0))(
                predictor, head, root, search_candidates
            )
            candidates = jnp.concatenate((guided, search_candidates))
            scores = jnp.concatenate((guided_scores, search_scores))
            elite_scores, elite_indices = jax.lax.top_k(scores, self.elites)
            elites = candidates[elite_indices]
            elite_frequency = jax.nn.one_hot(elites, NUM_ACTIONS).mean(axis=0)
            probs = uniform_mix(
                (1 - self.elite_weight) * probs + self.elite_weight * elite_frequency
            )
            best = jnp.where(elite_scores[0] > best_score, elites[0], best)
            return (probs, best, jnp.maximum(best_score, elite_scores[0]), elites), None

        (probs, best, _, _), _ = jax.lax.scan(
            iteration,
            (
                previous.probabilities,
                previous.actions,
                jnp.asarray(-jnp.inf),
                starting_elites,
            ),
            (jnp.arange(self.iterations), jax.random.split(key, self.iterations)),
        )
        return Plan(best, probs, jnp.asarray(True))
