from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int, PRNGKeyArray

from core.actions import unit_transition_cost_matrix
from core.config import NUM_ACTIONS, PlannerConfig
from core.interfaces import Predictor


def _compute_sequence_penalties(
    action_seqs: Int[Array, "... sequence_len"],
    smoothness_weight: float,
    prev_action: Optional[Int[Array, ""]] = None,
) -> Float[Array, "..."]:
    if smoothness_weight <= 0.0:
        return jnp.zeros(action_seqs.shape[:-1], dtype=jnp.float32)

    cost_matrix = smoothness_weight * unit_transition_cost_matrix()
    transitions = cost_matrix[action_seqs[..., :-1], action_seqs[..., 1:]]
    penalties = jnp.sum(transitions, axis=-1)

    if prev_action is not None:
        first_actions = action_seqs[..., 0]
        penalties = penalties + cost_matrix[prev_action, first_actions]

    return penalties


class RandomShootingPlanner(eqx.Module):
    """
    Randomly samples N action sequences, unrolls them through the predictor, and
    returns the sequence with the highest objective score.
    """

    predictor: Predictor
    objective_fn: Callable[[Float[Array, "latent_dim"]], Float[Array, ""]]
    sequence_len: int
    num_samples: int
    smoothness_weight: float = 0.0

    def _step_fn(
        self,
        latent: Float[Array, "latent_dim"],
        action: Int[Array, ""],
    ) -> Tuple[Float[Array, "latent_dim"], None]:
        next_state = self.predictor(latent, action)
        return next_state, None

    def _rollout_fn(
        self,
        actions: Int[Array, "sequence_len"],
        current_latent_state: Float[Array, "latent_dim"],
    ) -> Float[Array, ""]:
        scan_step = Partial(self._step_fn)
        final_latent, _ = jax.lax.scan(scan_step, current_latent_state, actions)
        return self.objective_fn(final_latent)

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

        prev_action: Optional[Int[Array, ""]] = kwargs.get("prev_action")

        action_seqs = jax.random.randint(
            key, (self.num_samples, self.sequence_len), minval=0, maxval=NUM_ACTIONS
        )
        rollout_wrapper = Partial(
            self._rollout_fn, current_latent_state=current_latent_state
        )
        scores = jax.vmap(rollout_wrapper)(action_seqs)
        if self.smoothness_weight > 0.0:
            penalties = _compute_sequence_penalties(
                action_seqs, self.smoothness_weight, prev_action
            )
            scores = scores - penalties
        best_idx = jnp.argmax(scores)
        return action_seqs[best_idx]


class CEMPlanner(eqx.Module):
    """
    Iteratively samples action sequences from a parameterized distribution,
    evaluates them, and updates the distribution towards high-reward trajectories.
    """

    predictor: Predictor
    objective_fn: Callable[[Float[Array, "latent_dim"]], Float[Array, ""]]
    sequence_len: int
    num_iters: int
    num_samples: int
    num_elites: int
    alpha: float
    smoothness_weight: float = 0.0

    def _step_fn(
        self,
        latent: Float[Array, "latent_dim"],
        action: Int[Array, ""],
    ) -> Tuple[Float[Array, "latent_dim"], None]:
        next_state = self.predictor(latent, action)
        return next_state, None

    def _rollout_fn(
        self,
        actions: Int[Array, "sequence_len"],
        current_latent_state: Float[Array, "latent_dim"],
    ) -> Float[Array, ""]:
        scan_step = Partial(self._step_fn)
        final_latent, _ = jax.lax.scan(scan_step, current_latent_state, actions)
        return self.objective_fn(final_latent)

    def _cem_iter_fn(
        self,
        logits: Float[Array, "sequence_len NUM_ACTIONS"],
        iter_key: PRNGKeyArray,
        *,
        current_latent_state: Float[Array, "latent_dim"],
        prev_action: Optional[Int[Array, ""]] = None,
    ) -> Tuple[Float[Array, "sequence_len NUM_ACTIONS"], Int[Array, "sequence_len"]]:
        action_seqs = jax.random.categorical(
            iter_key, logits, shape=(self.num_samples, self.sequence_len)
        )
        rollout_wrapper = Partial(
            self._rollout_fn, current_latent_state=current_latent_state
        )
        scores = jax.vmap(rollout_wrapper)(action_seqs)
        if self.smoothness_weight > 0.0:
            penalties = _compute_sequence_penalties(
                action_seqs, self.smoothness_weight, prev_action
            )
            scores = scores - penalties

        _, topk_indices = jax.lax.top_k(scores, self.num_elites)
        elites = action_seqs[topk_indices]

        elite_one_hot = jax.nn.one_hot(elites, NUM_ACTIONS)
        elite_probs = jnp.mean(elite_one_hot, axis=0)

        current_probs = jax.nn.softmax(logits, axis=-1)
        updated_probs = (1 - self.alpha) * current_probs + self.alpha * elite_probs

        updated_logits = jnp.log(updated_probs + 1e-6)
        return updated_logits, elites[0]

    def __call__(
        self,
        current_latent_state: Float[Array, "latent_dim"],
        **kwargs: Any,
    ) -> Int[Array, "sequence_len"]:
        key: Optional[PRNGKeyArray] = kwargs.get("key")
        if key is None:
            raise ValueError("CEMPlanner requires a PRNGKey passed as 'key' in kwargs.")

        prev_action: Optional[Int[Array, ""]] = kwargs.get("prev_action")

        init_logits = jnp.zeros((self.sequence_len, NUM_ACTIONS))

        cem_iter_wrapper = Partial(
            self._cem_iter_fn,
            current_latent_state=current_latent_state,
            prev_action=prev_action,
        )

        keys = jax.random.split(key, self.num_iters)
        _, best_seqs = jax.lax.scan(cem_iter_wrapper, init_logits, keys)

        return best_seqs[-1]


class BeamSearchPlanner(eqx.Module):
    """
    Maintains a set of the top-K highest scoring nodes of the search tree.
    Only expands them in the next iteration.
    """

    predictor: Predictor
    objective_fn: Callable[[Float[Array, "latent_dim"]], Float[Array, ""]]
    sequence_len: int
    beam_width: int
    smoothness_weight: float = 0.0

    def _expand_beam(
        self,
        state: Float[Array, "latent_dim"],
        actions_to_try: Int[Array, "num_actions"],
    ) -> Tuple[Float[Array, "num_actions latent_dim"], Float[Array, "num_actions"]]:
        next_states = jax.vmap(self.predictor, in_axes=(None, 0))(state, actions_to_try)
        new_scores = jax.vmap(self.objective_fn)(next_states)
        return next_states, new_scores

    def _step_fn(
        self,
        carry: Tuple[
            Float[Array, "beam_width latent_dim"],
            Int[Array, "beam_width sequence_len"],
        ],
        step_idx: Int[Array, ""],
        actions_to_try: Int[Array, "num_actions"],
        prev_action: Optional[Int[Array, ""]] = None,
    ) -> Tuple[
        Tuple[
            Float[Array, "beam_width latent_dim"],
            Int[Array, "beam_width sequence_len"],
        ],
        None,
    ]:
        """Performs a single step of beam expansion and pruning across candidates.

        Arguments:
          carry: Tuple of current beam latent states and accumulated action sequences
          step_idx: Current planning horizon step index
          actions_to_try: Array of candidate discrete actions to evaluate
          prev_action: Optional previous action for smoothness penalty calculation

        Returns:
          Updated (beam_states, beam_actions) carry tuple and empty scan output
        """
        beam_states, beam_actions = carry
        expand_wrapper = Partial(self._expand_beam, actions_to_try=actions_to_try)
        next_states, new_scores = jax.vmap(expand_wrapper)(beam_states)

        if self.smoothness_weight > 0.0:
            cost_matrix = self.smoothness_weight * unit_transition_cost_matrix()
            prev_actions = jnp.where(
                step_idx > 0,
                beam_actions[:, step_idx - 1],
                prev_action if prev_action is not None else 0,
            )
            should_penalize = (step_idx > 0) | (prev_action is not None)
            step_penalties = jnp.where(
                should_penalize,
                cost_matrix[prev_actions[:, None], actions_to_try[None, :]],
                0.0,
            )
            new_scores = new_scores - step_penalties

        # At step 0, all beam states are identical (root state).
        # Mask out beams 1..beam_width-1 so we only expand beam 0
        mask = jnp.where(step_idx == 0, jnp.arange(self.beam_width) > 0, False)
        new_scores = jnp.where(mask[:, None], -jnp.inf, new_scores)

        flat_scores = new_scores.flatten()
        _, topk_indices = jax.lax.top_k(flat_scores, self.beam_width)

        beam_indices = topk_indices // NUM_ACTIONS
        action_indices = topk_indices % NUM_ACTIONS

        new_beam_states = next_states[beam_indices, action_indices]
        new_beam_actions = beam_actions[beam_indices]
        new_beam_actions = new_beam_actions.at[:, step_idx].set(action_indices)

        return (new_beam_states, new_beam_actions), None

    def __call__(
        self,
        current_latent_state: Float[Array, "latent_dim"],
        **kwargs: Any,
    ) -> Int[Array, "sequence_len"]:
        init_states = jnp.repeat(current_latent_state[None, :], self.beam_width, axis=0)
        init_actions = jnp.zeros((self.beam_width, self.sequence_len), dtype=jnp.int32)

        prev_action: Optional[Int[Array, ""]] = kwargs.get("prev_action")

        actions_to_try = jnp.arange(NUM_ACTIONS)
        scan_step = Partial(
            self._step_fn, actions_to_try=actions_to_try, prev_action=prev_action
        )

        (final_states, final_actions), _ = jax.lax.scan(
            scan_step,
            (init_states, init_actions),
            jnp.arange(self.sequence_len),
        )

        final_scores = jax.vmap(self.objective_fn)(final_states)
        if self.smoothness_weight > 0.0:
            penalties = _compute_sequence_penalties(
                final_actions, self.smoothness_weight, prev_action
            )
            final_scores = final_scores - penalties

        best_idx = jnp.argmax(final_scores)
        return final_actions[best_idx]


def create_planner(
    cfg: PlannerConfig,
    predictor: Predictor,
    objective_fn: Callable[[Float[Array, "latent_dim"]], Float[Array, ""]],
) -> Any:
    """Instantiates a trajectory planner from a PlannerConfig."""
    if cfg.type in ("beam", "beam_search"):
        return BeamSearchPlanner(
            predictor=predictor,
            objective_fn=objective_fn,
            sequence_len=cfg.sequence_len,
            smoothness_weight=cfg.smoothness_weight,
            beam_width=cfg.beam_width,
        )
    elif cfg.type == "cem":
        return CEMPlanner(
            predictor=predictor,
            objective_fn=objective_fn,
            sequence_len=cfg.sequence_len,
            smoothness_weight=cfg.smoothness_weight,
            num_iters=cfg.cem_iters,
            num_samples=cfg.cem_samples,
            num_elites=cfg.cem_elites,
            alpha=cfg.cem_alpha,
        )
    elif cfg.type in ("random", "random_shooting"):
        return RandomShootingPlanner(
            predictor=predictor,
            objective_fn=objective_fn,
            sequence_len=cfg.sequence_len,
            smoothness_weight=cfg.smoothness_weight,
            num_samples=cfg.rs_samples,
        )
    else:
        raise ValueError(
            f"Unsupported planner '{cfg.type}'. Choose from 'beam', 'cem', or 'random'."
        )
