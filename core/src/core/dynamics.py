import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from core.config import (
    ACTION_HISTORY_FEATURES,
    NUM_ACTIONS,
    PredictorConfig,
)


class MLPPredictor(eqx.Module):
    """Predict learned dynamics and advance the known control history exactly."""

    latent_dim: int = eqx.field(static=True)
    predictor_mlp: eqx.nn.MLP

    def __init__(self, cfg: PredictorConfig, key: PRNGKeyArray):
        self.latent_dim = cfg.latent_dim
        mlp = eqx.nn.MLP(
            in_size=self.state_dim + 3,
            out_size=cfg.latent_dim,
            width_size=cfg.hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=key,
        )
        output_weights = mlp.layers[-1].weight
        # Start close to persistence so errors do not compound at initialization.
        self.predictor_mlp = eqx.tree_at(
            lambda model: (model.layers[-1].weight, model.layers[-1].bias),
            mlp,
            (
                0.02 * output_weights,
                jnp.zeros(cfg.latent_dim, dtype=output_weights.dtype),
            ),
        )

    def __check_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("Learned latent width must be positive.")

    @property
    def state_dim(self) -> int:
        return self.latent_dim + ACTION_HISTORY_FEATURES

    def __call__(
        self, latent_state: Float[Array, "state_dim"], action: Float[Array, "3"]
    ) -> Float[Array, "state_dim"]:
        if latent_state.shape != (self.state_dim,):
            raise ValueError("Model state must include the learned latent and history.")
        x = jnp.concatenate([latent_state, action])
        next_latent = latent_state[: self.latent_dim] + self.predictor_mlp(x)
        next_history = jnp.concatenate([latent_state[-3:], action])
        return jnp.concatenate([next_latent, next_history])


class RewardQHead(eqx.Module):
    """Predict interval rewards and returns for the same discrete actions."""

    latent_dim: int = eqx.field(static=True)
    hidden: tuple[eqx.nn.Linear, eqx.nn.Linear]
    reward: eqx.nn.Linear
    q: eqx.nn.Linear

    def __init__(self, latent_dim: int, key: PRNGKeyArray) -> None:
        self.latent_dim = latent_dim
        first, second, reward, value = jax.random.split(key, 4)
        width = 256
        self.hidden = (
            eqx.nn.Linear(latent_dim + ACTION_HISTORY_FEATURES, width, key=first),
            eqx.nn.Linear(width, width, key=second),
        )

        def zero_projection(output_key):
            projection = eqx.nn.Linear(width, NUM_ACTIONS, key=output_key)
            return eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                projection,
                (
                    jnp.zeros_like(projection.weight),
                    jnp.zeros(NUM_ACTIONS, dtype=projection.weight.dtype),
                ),
            )

        self.reward = zero_projection(reward)
        self.q = zero_projection(value)

    def __check_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("Learned latent width must be positive.")

    def __call__(
        self, state: Float[Array, "state_dim"]
    ) -> tuple[Float[Array, "actions"], Float[Array, "actions"]]:
        features = state
        for layer in self.hidden:
            features = jax.nn.silu(layer(features))
        return self.reward(features), self.q(features)

    def for_search(
        self, state: Float[Array, "state_dim"]
    ) -> tuple[Float[Array, "actions"], Float[Array, "actions"]]:
        reward, q = self(state)
        absorbing = is_absorbing(state, self.latent_dim)
        return jnp.where(absorbing, 0.0, reward), jnp.where(absorbing, 0.0, q)


def is_absorbing(state: Float[Array, "state_dim"], latent_dim: int) -> Bool[Array, ""]:
    return jnp.all(state[:latent_dim] == 0)


def search_transition(
    predictor: MLPPredictor, state: Float[Array, "state_dim"], action: Float[Array, "3"]
) -> Float[Array, "state_dim"]:
    next_state = predictor(state, action)
    absorbing = is_absorbing(state, predictor.latent_dim) | is_absorbing(
        next_state, predictor.latent_dim
    )
    return jnp.where(absorbing, jnp.zeros_like(next_state), next_state)
