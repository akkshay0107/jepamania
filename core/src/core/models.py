import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from core.config import (
    EncoderConfig,
    PredictorConfig,
    IMG_HIST_LEN,
    LIDAR_FEATURES,
    TELEMETRY_FEATURES,
    NUM_ACTIONS,
)


class SubJepaEncoder(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    lidar_mlp: eqx.nn.MLP
    telemetry_mlp: eqx.nn.MLP
    fusion_mlp: eqx.nn.MLP

    def __init__(self, cfg: EncoderConfig, key: PRNGKeyArray):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        self.conv1 = eqx.nn.Conv2d(IMG_HIST_LEN, 32, kernel_size=8, stride=4, key=k1)
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=4, stride=2, key=k2)
        self.conv3 = eqx.nn.Conv2d(64, 64, kernel_size=3, stride=1, key=k3)

        flattened_img_size = 64 * 4 * 4
        flattened_lidar_size = IMG_HIST_LEN * LIDAR_FEATURES

        self.lidar_mlp = eqx.nn.MLP(
            in_size=flattened_lidar_size,
            out_size=64,
            width_size=128,
            depth=2,
            activation=jax.nn.silu,
            key=k4,
        )
        self.telemetry_mlp = eqx.nn.MLP(
            in_size=TELEMETRY_FEATURES,
            out_size=64,
            width_size=128,
            depth=2,
            activation=jax.nn.silu,
            key=k5,
        )

        fusion_in = flattened_img_size + 64 + 64
        self.fusion_mlp = eqx.nn.MLP(
            in_size=fusion_in,
            out_size=cfg.latent_dim,
            width_size=512,
            depth=2,
            activation=jax.nn.silu,
            key=k6,
        )

    def __call__(
        self,
        screen: Float[Array, "hist h w"],
        lidar: Float[Array, "hist rays"],
        telemetry: Float[Array, "features"],
    ) -> Float[Array, "latent_dim"]:
        x_screen = jax.nn.relu(self.conv1(screen))
        x_screen = jax.nn.relu(self.conv2(x_screen))
        x_screen = jax.nn.relu(self.conv3(x_screen))
        x_screen = x_screen.reshape(-1)

        x_lidar = self.lidar_mlp(lidar.reshape(-1))
        x_telemetry = self.telemetry_mlp(telemetry)

        x_fused = jnp.concatenate([x_screen, x_lidar, x_telemetry], axis=0)
        return self.fusion_mlp(x_fused)


class SubJepaPredictor(eqx.Module):
    action_embedding: eqx.nn.Embedding
    predictor_mlp: eqx.nn.MLP

    def __init__(self, cfg: PredictorConfig, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key, 2)
        self.action_embedding = eqx.nn.Embedding(
            num_embeddings=NUM_ACTIONS, embedding_size=cfg.action_embed_dim, key=k1
        )
        self.predictor_mlp = eqx.nn.MLP(
            in_size=cfg.latent_dim + cfg.action_embed_dim,
            out_size=cfg.latent_dim,
            width_size=cfg.hidden_dim,
            depth=3,
            activation=jax.nn.silu,
            key=k2,
        )

    def __call__(
        self, latent_state: Float[Array, "latent_dim"], action: Int[Array, ""]
    ) -> Float[Array, "latent_dim"]:
        a_emb = self.action_embedding(action)
        x = jnp.concatenate([latent_state, a_emb], axis=0)
        return self.predictor_mlp(x)
