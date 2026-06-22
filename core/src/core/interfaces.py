from typing import Any, Dict, Protocol, runtime_checkable

from jaxtyping import Array, Float, Int


@runtime_checkable
class Encoder(Protocol):
    """
    Protocol for an Encoder.
    Any class with this __call__ signature automatically satisfies this contract.
    """

    def __call__(
        self, observations: Dict[str, Float[Array, "..."]]
    ) -> Float[Array, "latent_dim"]: ...


@runtime_checkable
class Predictor(Protocol):
    """
    Protocol for a latent forward dynamics Predictor.
    """

    def __call__(
        self, latent_state: Float[Array, "latent_dim"], action: Int[Array, ""]
    ) -> Float[Array, "latent_dim"]: ...


@runtime_checkable
class Planner(Protocol):
    """
    Protocol for an action Planner.
    Returns a sequence of actions. For single-step plans, return a sequence of length 1.
    """

    def __call__(
        self,
        current_latent_state: Float[Array, "latent_dim"],
        goal_latent_state: Float[Array, "latent_dim"],
        **kwargs: Any,
    ) -> Int[Array, "sequence_len"]: ...
