from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int, PRNGKeyArray

from core.config import NUM_ACTIONS
from core.interfaces import Predictor


class RandomShootingPlanner(eqx.Module):
    """
    Randomly samples N action sequences, unrolls them through the predictor, and
    returns the sequence with the highest objective score.
    """

    predictor: Predictor
    objective_fn: Callable[
        [Float[Array, ""], Float[Array, "latent_dim"]], Float[Array, ""]
    ]
    sequence_len: int
    num_samples: int

    def _step_fn(
        self,
        carry: Tuple[Float[Array, ""], Float[Array, "latent_dim"]],
        action: Int[Array, ""],
    ) -> Tuple[Tuple[Float[Array, ""], Float[Array, "latent_dim"]], None]:
        # The carry maintains the necessary state and the per step
        # output is not needed. None returned in that slot to prevent
        # memory alloc for per step scores.
        score, latent = carry
        next_state = self.predictor(latent, action)
        next_score = self.objective_fn(score, next_state)
        return (next_score, next_state), None

    def _rollout_fn(
        self,
        actions: Int[Array, "sequence_len"],
        current_latent_state: Float[Array, "latent_dim"],
    ) -> Float[Array, ""]:
        (final_score, _), _ = jax.lax.scan(
            self._step_fn, (jnp.array(0.0), current_latent_state), actions
        )
        return final_score

    def __call__(
        self,
        current_latent_state: Float[Array, "latent_dim"],
        **kwargs: Any,
    ) -> Int[Array, "sequence_len"]:
        key: Optional[PRNGKeyArray] = kwargs.get("key")
        if key is None:
            raise ValueError(
                "RandomShootingPlanner requires a PRNGKey passed as 'key' in kwargs."
            )

        action_seqs = jax.random.randint(
            key, (self.num_samples, self.sequence_len), minval=0, maxval=NUM_ACTIONS
        )
        rollout_wrapper = Partial(
            self._rollout_fn, current_latent_state=current_latent_state
        )

        scores = jax.vmap(rollout_wrapper)(action_seqs)
        best_idx = jnp.argmax(scores)
        return action_seqs[best_idx]


class CEMPlanner(eqx.Module):
    """
    Iteratively samples action sequences from a parameterized distribution,
    evaluates them, and updates the distribution towards high-reward trajectories.
    """

    predictor: Predictor
    objective_fn: Callable[
        [Float[Array, ""], Float[Array, "latent_dim"]], Float[Array, ""]
    ]
    sequence_len: int

    def __call__(
        self,
        current_latent_state: Float[Array, "latent_dim"],
        **kwargs: Any,
    ) -> Int[Array, "sequence_len"]:
        raise NotImplementedError("CEMPlanner is not yet implemented.")


class BeamSearchPlanner(eqx.Module):
    """
    Maintains a set of the top-K highest scoring nodes of the search tree.
    Only expands them in the next iteration.
    """

    predictor: Predictor
    objective_fn: Callable[
        [Float[Array, ""], Float[Array, "latent_dim"]], Float[Array, ""]
    ]
    sequence_len: int
    beam_width: int

    def _expand_beam(
        self,
        state: Float[Array, "latent_dim"],
        current_score: Float[Array, ""],
        actions_to_try: Int[Array, "num_actions"],
    ) -> Tuple[Float[Array, "num_actions latent_dim"], Float[Array, "num_actions"]]:
        next_states = jax.vmap(self.predictor, in_axes=(None, 0))(state, actions_to_try)
        new_scores = jax.vmap(self.objective_fn, in_axes=(None, 0))(
            current_score, next_states
        )
        return next_states, new_scores

    def _step_fn(
        self,
        carry: Tuple[
            Float[Array, "beam_width latent_dim"],
            Int[Array, "beam_width sequence_len"],
            Float[Array, "beam_width"],
        ],
        step_idx: Int[Array, ""],
        actions_to_try: Int[Array, "num_actions"],
    ) -> Tuple[
        Tuple[
            Float[Array, "beam_width latent_dim"],
            Int[Array, "beam_width sequence_len"],
            Float[Array, "beam_width"],
        ],
        None,
    ]:
        # The carry maintains the necessary state and the per step
        # output is not needed. None returned in that slot to prevent
        # memory alloc for per step scores.
        beam_states, beam_actions, beam_scores = carry
        expand_wrapper = Partial(self._expand_beam, actions_to_try=actions_to_try)
        next_states, new_scores = jax.vmap(expand_wrapper)(beam_states, beam_scores)

        flat_scores = new_scores.flatten()
        topk_scores, topk_indices = jax.lax.top_k(flat_scores, self.beam_width)

        beam_indices = topk_indices // NUM_ACTIONS
        action_indices = topk_indices % NUM_ACTIONS

        new_beam_states = next_states[beam_indices, action_indices]
        new_beam_actions = beam_actions[beam_indices]
        new_beam_actions = new_beam_actions.at[:, step_idx].set(action_indices)

        return (new_beam_states, new_beam_actions, topk_scores), None

    def __call__(
        self,
        current_latent_state: Float[Array, "latent_dim"],
        **kwargs: Any,
    ) -> Int[Array, "sequence_len"]:
        init_states = jnp.repeat(current_latent_state[None, :], self.beam_width, axis=0)
        init_actions = jnp.zeros((self.beam_width, self.sequence_len), dtype=jnp.int32)
        init_scores = jnp.full((self.beam_width,), -jnp.inf)
        init_scores = init_scores.at[0].set(0.0)

        actions_to_try = jnp.arange(NUM_ACTIONS)
        scan_step = Partial(self._step_fn, actions_to_try=actions_to_try)

        (final_states, final_actions, final_scores), _ = jax.lax.scan(
            scan_step,
            (init_states, init_actions, init_scores),
            jnp.arange(self.sequence_len),
        )

        best_beam_idx = jnp.argmax(final_scores)
        return final_actions[best_beam_idx]
