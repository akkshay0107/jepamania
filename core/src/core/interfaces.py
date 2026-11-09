from typing import Any, Mapping, Protocol, runtime_checkable

from jaxtyping import Array, Float


@runtime_checkable
class Encoder(Protocol):
    """
    Protocol for an Encoder.
    Any class with this __call__ signature automatically satisfies this contract.
    """

    def __call__(
        self, observations: Mapping[str, Any]
    ) -> Float[Array, "latent_dim"]: ...


@runtime_checkable
class Predictor(Protocol):
    """
    Protocol for a latent forward dynamics Predictor.
    """

    def __call__(
        self, latent_state: Float[Array, "latent_dim"], action: Float[Array, "3"]
    ) -> Float[Array, "latent_dim"]: ...
