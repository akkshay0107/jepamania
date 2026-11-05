import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from core.config import (
    ACTION_HISTORY_FEATURES,
    NUM_ACTIONS,
    PolicyHeadConfig,
    PredictorConfig,
    ValueHeadConfig,
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


class MLPValueHead(eqx.Module):
    value_mlp: eqx.nn.MLP

    def __init__(self, cfg: ValueHeadConfig, key: PRNGKeyArray):
        self.value_mlp = eqx.nn.MLP(
            in_size=cfg.latent_dim + ACTION_HISTORY_FEATURES,
            out_size=1,
            width_size=cfg.hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=key,
        )

    def __call__(self, latent_state: Float[Array, "state_dim"]) -> Float[Array, ""]:
        return self.value_mlp(latent_state).squeeze(axis=-1)


class MLPPolicyHead(eqx.Module):
    policy_mlp: eqx.nn.MLP
    temperature: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(self, cfg: PolicyHeadConfig, key: PRNGKeyArray):
        if cfg.temperature <= 0.0:
            raise ValueError("Policy temperature must be positive.")
        if not 0.0 <= cfg.epsilon <= 1.0:
            raise ValueError("Policy epsilon must be in [0, 1].")

        self.policy_mlp = eqx.nn.MLP(
            in_size=cfg.latent_dim + ACTION_HISTORY_FEATURES,
            out_size=NUM_ACTIONS,
            width_size=cfg.hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=key,
        )
        self.temperature = cfg.temperature
        self.epsilon = cfg.epsilon

    def raw_logits(
        self, latent_state: Float[Array, "state_dim"]
    ) -> Float[Array, "NUM_ACTIONS"]:
        """Return unsmoothed categorical logits for training."""
        return self.policy_mlp(latent_state)

    def __call__(
        self, latent_state: Float[Array, "state_dim"]
    ) -> Float[Array, "NUM_ACTIONS"]:
        """
        Returns smoothed probabilities using temperature scaling and epsilon mixing.
        """
        logits = self.raw_logits(latent_state)
        scaled_logits = logits / self.temperature
        probs = jax.nn.softmax(scaled_logits)
        return self.epsilon / NUM_ACTIONS + (1.0 - self.epsilon) * probs
