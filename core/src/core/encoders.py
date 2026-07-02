import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from core.config import IMG_HIST_LEN, TELEMETRY_FEATURES, EncoderConfig


def get_2d_sincos_pos_embed(
    latent_dim: int, grid_size: int = 4
) -> Float[Array, "grid_size*grid_size latent_dim"]:
    grid_y, grid_x = jnp.meshgrid(
        jnp.arange(grid_size), jnp.arange(grid_size), indexing="ij"
    )
    grid_y = grid_y.flatten()
    grid_x = grid_x.flatten()

    assert latent_dim % 2 == 0, f"latent_dim must be even, got {latent_dim}"
    d = latent_dim // 2

    # Standard absolute positional embedding
    omega = 1.0 / (10000 ** (jnp.arange(d // 2) * 2 / d))
    out_y = grid_y[:, None] * omega[None, :]
    emb_y = jnp.concatenate([jnp.sin(out_y), jnp.cos(out_y)], axis=-1)

    out_x = grid_x[:, None] * omega[None, :]
    emb_x = jnp.concatenate([jnp.sin(out_x), jnp.cos(out_x)], axis=-1)

    emb = jnp.concatenate([emb_y, emb_x], axis=-1)
    return emb


class ConvStem(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    pool: eqx.nn.AvgPool2d

    def __init__(self, in_channels: int, out_channels: int, key: PRNGKeyArray):
        key1, key2, key3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, key=key1
        )
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, key=key2)
        self.conv3 = eqx.nn.Conv2d(
            64, out_channels, kernel_size=3, stride=2, padding=1, key=key3
        )
        self.pool = eqx.nn.AvgPool2d(kernel_size=2, stride=2)

    def __call__(
        self, x: Float[Array, "in_channels H W"]
    ) -> Float[Array, "out_channels 4 4"]:
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = self.pool(x)
        return x


class TransformerBlock(eqx.Module):
    ln1: eqx.nn.LayerNorm
    mha: eqx.nn.MultiheadAttention
    ln2: eqx.nn.LayerNorm
    mlp_linear1: eqx.nn.Linear
    mlp_linear2: eqx.nn.Linear

    def __init__(
        self, latent_dim: int, num_heads: int, mlp_ratio: float, key: PRNGKeyArray
    ):
        key_mha, key_l1, key_l2 = jax.random.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(shape=(latent_dim,))
        self.mha = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=latent_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=key_mha,
        )
        self.ln2 = eqx.nn.LayerNorm(shape=(latent_dim,))
        hidden_dim = int(latent_dim * mlp_ratio)
        self.mlp_linear1 = eqx.nn.Linear(latent_dim, hidden_dim, key=key_l1)
        self.mlp_linear2 = eqx.nn.Linear(hidden_dim, latent_dim, key=key_l2)

    def __call__(
        self, x: Float[Array, "seq_len latent_dim"]
    ) -> Float[Array, "seq_len latent_dim"]:
        # Pre-LN attention
        x_ln1 = jax.vmap(self.ln1)(x)
        attn_out = self.mha(x_ln1, x_ln1, x_ln1)
        x = x + attn_out

        # MLP
        x_ln2 = jax.vmap(self.ln2)(x)
        mlp_out = jax.vmap(self.mlp_linear1)(x_ln2)
        mlp_out = jax.nn.gelu(mlp_out)
        mlp_out = jax.vmap(self.mlp_linear2)(mlp_out)
        x = x + mlp_out
        return x


class TransformerStack(eqx.Module):
    layers: list[TransformerBlock]

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, num_layers)
        self.layers = [
            TransformerBlock(latent_dim, num_heads, mlp_ratio, keys[i])
            for i in range(num_layers)
        ]

    def __call__(
        self, x: Float[Array, "seq_len latent_dim"]
    ) -> Float[Array, "seq_len latent_dim"]:
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionPool(eqx.Module):
    query: Float[Array, "1 latent_dim"]
    mha: eqx.nn.MultiheadAttention
    ln: eqx.nn.LayerNorm

    def __init__(self, latent_dim: int, num_heads: int, key: PRNGKeyArray):
        key_q, key_mha = jax.random.split(key)
        self.query = jax.random.normal(key_q, (1, latent_dim)) * 0.02
        self.mha = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=latent_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=key_mha,
        )
        self.ln = eqx.nn.LayerNorm(shape=(latent_dim,))

    def __call__(
        self, x: Float[Array, "seq_len latent_dim"]
    ) -> Float[Array, "latent_dim"]:
        x_ln = jax.vmap(self.ln)(x)
        pooled = self.mha(self.query, x_ln, x_ln)
        return jnp.squeeze(pooled, axis=0)


class ViTEncoder(eqx.Module):
    latent_dim: int
    conv_stem: ConvStem

    tel_progress_proj: eqx.nn.Linear
    tel_kinematics_proj: eqx.nn.Linear
    tel_mechanics_proj: eqx.nn.Linear
    tel_control_proj: eqx.nn.Linear
    telemetry_type_embed: eqx.nn.Embedding

    transformer: TransformerStack
    pool: AttentionPool

    def __init__(self, cfg: EncoderConfig, key: PRNGKeyArray):
        self.latent_dim = cfg.latent_dim

        key_stem, key_telemetry, key_tf, key_pool = jax.random.split(key, 4)

        self.conv_stem = ConvStem(
            in_channels=IMG_HIST_LEN, out_channels=cfg.latent_dim, key=key_stem
        )

        key_t_prog, key_t_kin, key_t_mech, key_t_ctrl, key_t_emb = jax.random.split(
            key_telemetry, 5
        )

        self.tel_progress_proj = eqx.nn.Linear(4, cfg.latent_dim, key=key_t_prog)
        self.tel_kinematics_proj = eqx.nn.Linear(12, cfg.latent_dim, key=key_t_kin)
        self.tel_mechanics_proj = eqx.nn.Linear(11, cfg.latent_dim, key=key_t_mech)
        self.tel_control_proj = eqx.nn.Linear(6, cfg.latent_dim, key=key_t_ctrl)
        self.telemetry_type_embed = eqx.nn.Embedding(4, cfg.latent_dim, key=key_t_emb)

        self.transformer = TransformerStack(
            latent_dim=cfg.latent_dim,
            num_layers=cfg.transformer.num_layers,
            num_heads=cfg.transformer.num_heads,
            mlp_ratio=cfg.transformer.mlp_ratio,
            key=key_tf,
        )
        self.pool = AttentionPool(
            latent_dim=cfg.latent_dim,
            num_heads=cfg.transformer.num_heads,
            key=key_pool,
        )

    def __call__(self, observations: dict[str, Array]) -> Float[Array, "latent_dim"]:
        screen = observations["screen"]  # (4, 64, 64)
        telemetry = observations["telemetry"]  # (33,)

        x_visual = self.conv_stem(screen)  # (latent_dim, 4, 4)
        x_visual = x_visual.reshape(self.latent_dim, 16)
        x_visual = x_visual.T  # (16, latent_dim)
        x_visual = x_visual + get_2d_sincos_pos_embed(self.latent_dim, grid_size=4)

        tel_progress = telemetry[0:4]
        tel_kinematics = telemetry[4:16]
        tel_mechanics = telemetry[16:27]
        tel_control = telemetry[27:33]

        token_tel_progress = self.tel_progress_proj(
            tel_progress
        ) + self.telemetry_type_embed(jnp.array(0))
        token_tel_kinematics = self.tel_kinematics_proj(
            tel_kinematics
        ) + self.telemetry_type_embed(jnp.array(1))
        token_tel_mechanics = self.tel_mechanics_proj(
            tel_mechanics
        ) + self.telemetry_type_embed(jnp.array(2))
        token_tel_control = self.tel_control_proj(
            tel_control
        ) + self.telemetry_type_embed(jnp.array(3))

        tel_tokens = jnp.stack(
            [
                token_tel_progress,
                token_tel_kinematics,
                token_tel_mechanics,
                token_tel_control,
            ],
            axis=0,
        )
        tokens = jnp.concatenate([x_visual, tel_tokens], axis=0)

        tokens = self.transformer(tokens)
        z_t = self.pool(tokens)
        return z_t


# simpler model with conv backbone and late fusion
class ConvEncoder(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    telemetry_mlp: eqx.nn.MLP
    fusion_mlp: eqx.nn.MLP

    def __init__(self, cfg: EncoderConfig, key: PRNGKeyArray):
        key_conv1, key_conv2, key_conv3, key_telemetry, key_fusion = jax.random.split(
            key, 5
        )

        self.conv1 = eqx.nn.Conv2d(
            IMG_HIST_LEN, 32, kernel_size=8, stride=4, key=key_conv1
        )
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=4, stride=2, key=key_conv2)
        self.conv3 = eqx.nn.Conv2d(64, 64, kernel_size=3, stride=1, key=key_conv3)

        flattened_img_size = 64 * 4 * 4

        self.telemetry_mlp = eqx.nn.MLP(
            in_size=TELEMETRY_FEATURES,
            out_size=64,
            width_size=128,
            depth=2,
            activation=jax.nn.silu,
            key=key_telemetry,
        )

        fusion_in = flattened_img_size + 64
        self.fusion_mlp = eqx.nn.MLP(
            in_size=fusion_in,
            out_size=cfg.latent_dim,
            width_size=512,
            depth=2,
            activation=jax.nn.silu,
            key=key_fusion,
        )

    def __call__(
        self,
        observations: dict[str, Array],
    ) -> Float[Array, "latent_dim"]:
        screen = observations["screen"]
        telemetry = observations["telemetry"]

        x_screen = jax.nn.relu(self.conv1(screen))
        x_screen = jax.nn.relu(self.conv2(x_screen))
        x_screen = jax.nn.relu(self.conv3(x_screen))
        x_screen = x_screen.reshape(-1)

        x_telemetry = self.telemetry_mlp(telemetry)

        x_fused = jnp.concatenate([x_screen, x_telemetry], axis=0)
        return self.fusion_mlp(x_fused)
