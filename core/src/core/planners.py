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


class CEMPlanner(eqx.Module):
    """
    Iteratively samples action sequences from a parameterized distribution,
    evaluates them, and updates the distribution towards high-reward trajectories.
    """

    predictor: Predictor
    objective_fn: Callable[[Float[Array, "latent_dim"]], Float[Array, ""]]
    sequence_len: int = eqx.field(static=True)
    num_iters: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    num_elites: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    policy_head: Optional[
        Callable[[Float[Array, "latent_dim"]], Float[Array, "NUM_ACTIONS"]]
    ] = None
    smoothness_weight: float = eqx.field(default=0.0, static=True)

    def __init__(
        self,
        predictor: Predictor,
        objective_fn: Callable[[Float[Array, "latent_dim"]], Float[Array, ""]],
        sequence_len: int,
        num_iters: int,
        num_samples: int,
        num_elites: int,
        alpha: float,
        policy_head: Optional[
            Callable[[Float[Array, "latent_dim"]], Float[Array, "NUM_ACTIONS"]]
        ] = None,
        smoothness_weight: float = 0.0,
    ) -> None:
        if sequence_len <= 0:
            raise ValueError("CEM sequence_len must be positive.")
        if num_iters <= 0:
            raise ValueError("CEM num_iters must be positive.")
        if num_samples <= 0:
            raise ValueError("CEM num_samples must be positive.")
        if not 0 < num_elites <= num_samples:
            raise ValueError("CEM num_elites must be in [1, num_samples].")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("CEM alpha must be in [0, 1].")
        if smoothness_weight < 0.0:
            raise ValueError("CEM smoothness_weight must be non-negative.")

        self.predictor = predictor
        self.objective_fn = objective_fn
        self.sequence_len = sequence_len
        self.num_iters = num_iters
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.alpha = alpha
        self.policy_head = policy_head
        self.smoothness_weight = smoothness_weight

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
        if self.policy_head is not None:
            prior_probs = self.policy_head(current_latent_state)
            # Epsilon-mixing guarantees probs > 0, so log is perfectly safe
            init_logits = init_logits.at[0].set(jnp.log(prior_probs))

        cem_iter_wrapper = Partial(
            self._cem_iter_fn,
            current_latent_state=current_latent_state,
            prev_action=prev_action,
        )

        keys = jax.random.split(key, self.num_iters)
        _, best_seqs = jax.lax.scan(cem_iter_wrapper, init_logits, keys)

        return best_seqs[-1]


def create_planner(
    cfg: PlannerConfig,
    predictor: Predictor,
    objective_fn: Callable[[Float[Array, "latent_dim"]], Float[Array, ""]],
    policy_head: Optional[
        Callable[[Float[Array, "latent_dim"]], Float[Array, "NUM_ACTIONS"]]
    ] = None,
) -> Any:
    """Instantiates the CEM trajectory planner from a PlannerConfig."""
    return CEMPlanner(
        predictor=predictor,
        objective_fn=objective_fn,
        policy_head=policy_head,
        sequence_len=cfg.sequence_len,
        smoothness_weight=cfg.smoothness_weight,
        num_iters=cfg.cem_iters,
        num_samples=cfg.cem_samples,
        num_elites=cfg.cem_elites,
        alpha=cfg.cem_alpha,
    )
