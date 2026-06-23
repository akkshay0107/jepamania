from typing import Any

import equinox as eqx
from jaxtyping import Array, Float, Int


class RandomShootingPlanner(eqx.Module):
    """
    Randomly samples N action sequences, unrolls them through the predictor, and
    returns the sequence with the lowest distance to the goal.
    """

    def __call__(
        self,
        current_latent_state: Float[Array, "latent_dim"],
        goal_latent_state: Float[Array, "latent_dim"],
        **kwargs: Any,
    ) -> Int[Array, "sequence_len"]:
        raise NotImplementedError("RandomShootingPlanner is not yet implemented.")


class CEMPlanner(eqx.Module):
    """
    Iteratively samples action sequences from a parameterized distribution,
    evaluates them, and updates the distribution towards high-reward trajectories.
    """

    def __call__(
        self,
        current_latent_state: Float[Array, "latent_dim"],
        goal_latent_state: Float[Array, "latent_dim"],
        **kwargs: Any,
    ) -> Int[Array, "sequence_len"]:
        raise NotImplementedError("CEMPlanner is not yet implemented.")


class BeamSearchPlanner(eqx.Module):
    """
    Maintains a set of the top-K highest scoring nodes of the search tree.
    Only expands them in the next iteration.
    """

    def __call__(
        self,
        current_latent_state: Float[Array, "latent_dim"],
        goal_latent_state: Float[Array, "latent_dim"],
        **kwargs: Any,
    ) -> Int[Array, "sequence_len"]:
        raise NotImplementedError("BeamSearchPlanner is not yet implemented.")
