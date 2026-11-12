import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from core.actions import NUM_ACTIONS
from core.config import ACTION_HISTORY_FEATURES, PredictorConfig


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


class RewardValueHead(eqx.Module):
    """Predict per-action rewards and termination, and a scalar state value."""

    latent_dim: int = eqx.field(static=True)
    hidden: tuple[eqx.nn.Linear, eqx.nn.Linear]
    reward: eqx.nn.Linear
    termination: eqx.nn.Linear
    value_projection: eqx.nn.Linear

    def __init__(self, latent_dim: int, key: PRNGKeyArray) -> None:
        self.latent_dim = latent_dim
        first, second, reward, termination, value = jax.random.split(key, 5)
        width = 256
        self.hidden = (
            eqx.nn.Linear(latent_dim + ACTION_HISTORY_FEATURES, width, key=first),
            eqx.nn.Linear(width, width, key=second),
        )

        def projection(output_key: PRNGKeyArray, outputs: int, bias: float = 0.0):
            layer = eqx.nn.Linear(width, outputs, key=output_key)
            return eqx.tree_at(
                lambda linear: (linear.weight, linear.bias),
                layer,
                (
                    jnp.zeros_like(layer.weight),
                    jnp.full(outputs, bias, dtype=layer.weight.dtype),
                ),
            )

        self.reward = projection(reward, NUM_ACTIONS)
        self.termination = projection(termination, NUM_ACTIONS, math.log(0.01 / 0.99))
        self.value_projection = projection(value, 1)

    def __check_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("Learned latent width must be positive.")

    def features(self, state: Float[Array, "state_dim"]) -> Float[Array, "hidden"]:
        features = state
        for layer in self.hidden:
            features = jax.nn.silu(layer(features))
        return features

    def state_value(self, state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        return self.value_projection(self.features(state))[0]

    def __call__(
        self, state: Float[Array, "state_dim"]
    ) -> tuple[Float[Array, "actions"], Float[Array, "actions"], Float[Array, ""]]:
        features = self.features(state)
        return (
            self.reward(features),
            self.termination(features),
            self.value_projection(features)[0],
        )
