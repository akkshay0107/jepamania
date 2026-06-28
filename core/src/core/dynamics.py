import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from core.config import NUM_ACTIONS, PredictorConfig, ValueHeadConfig


class MLPPredictor(eqx.Module):
    action_embedding: eqx.nn.Embedding
    predictor_mlp: eqx.nn.MLP

    def __init__(self, cfg: PredictorConfig, key: PRNGKeyArray):
        key_emb, key_mlp = jax.random.split(key, 2)
        self.action_embedding = eqx.nn.Embedding(
            num_embeddings=NUM_ACTIONS,
            embedding_size=cfg.action_embed_dim,
            key=key_emb,
        )
        self.predictor_mlp = eqx.nn.MLP(
            in_size=cfg.latent_dim + cfg.action_embed_dim,
            out_size=cfg.latent_dim,
            width_size=cfg.hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=key_mlp,
        )

    def __call__(
        self, latent_state: Float[Array, "latent_dim"], action: Int[Array, ""]
    ) -> Float[Array, "latent_dim"]:
        a_emb = self.action_embedding(action)
        x = jnp.concatenate([latent_state, a_emb], axis=0)
        return self.predictor_mlp(x)


class MLPValueHead(eqx.Module):
    value_mlp: eqx.nn.MLP

    def __init__(self, cfg: ValueHeadConfig, key: PRNGKeyArray):
        self.value_mlp = eqx.nn.MLP(
            in_size=cfg.latent_dim,
            out_size=1,
            width_size=cfg.hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=key,
        )

    def __call__(self, latent_state: Float[Array, "latent_dim"]) -> Float[Array, ""]:
        return self.value_mlp(latent_state).squeeze(axis=-1)
