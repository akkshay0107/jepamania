import functools

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray


@functools.partial(jax.vmap, in_axes=(0, None, None))
def _make_projector(
    key: PRNGKeyArray, latent_dim: int, subspace_dim: int
) -> Float[Array, "latent_dim subspace_dim"]:
    matrix = jax.random.normal(key, (latent_dim, subspace_dim))
    q, _ = jnp.linalg.qr(matrix)
    return q


@functools.partial(jax.vmap, in_axes=(0, None, None))
def _make_slice_projector(
    key: PRNGKeyArray, subspace_dim: int, num_slices: int
) -> Float[Array, "subspace_dim num_slices"]:
    matrix = jax.random.normal(key, (subspace_dim, num_slices))
    norms = jnp.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / norms


def generate_projectors(
    key: PRNGKeyArray,
    latent_dim: int,
    num_subspaces: int,
    subspace_dim: int,
    num_slices: int,
):
    key_subspace_split, key_slice_split = jax.random.split(key)
    keys_subspace = jax.random.split(key_subspace_split, num_subspaces)
    keys_slices = jax.random.split(key_slice_split, num_subspaces)

    subspace_projectors = _make_projector(keys_subspace, latent_dim, subspace_dim)
    slice_projectors = _make_slice_projector(keys_slices, subspace_dim, num_slices)

    return subspace_projectors, slice_projectors


@functools.partial(jax.vmap, in_axes=(None, 0))
def _project_into_subspace(
    target_latents: Float[Array, "batch latent_dim"],
    p: Float[Array, "latent_dim subspace_dim"],
) -> Float[Array, "batch subspace_dim"]:
    return target_latents @ p


@functools.partial(jax.vmap, in_axes=(0, 0))
def _project_into_slices(
    subspace_latents: Float[Array, "batch subspace_dim"],
    slice_p: Float[Array, "subspace_dim num_slices"],
) -> Float[Array, "batch num_slices"]:
    return subspace_latents @ slice_p


@functools.lru_cache(maxsize=4)
def _get_epps_pulley_constants_np(
    knots: int = 17, t_max: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, t_max, knots, dtype=np.float32)
    dt = t_max / (knots - 1)
    weights = np.full((knots,), 2 * dt, dtype=np.float32)
    weights[0] = dt
    weights[-1] = dt
    phi = np.exp(-(t**2) / 2.0).astype(np.float32)
    weights = weights * phi
    return t, phi, weights


def get_epps_pulley_constants(
    knots: int = 17, t_max: float = 3.0
) -> tuple[Float[Array, "knots"], Float[Array, "knots"], Float[Array, "knots"]]:
    """Computes and caches constant grid, target characteristic function,
    and quadrature weights for Epps-Pulley test."""
    t_np, phi_np, weights_np = _get_epps_pulley_constants_np(knots, t_max)
    return jnp.asarray(t_np), jnp.asarray(phi_np), jnp.asarray(weights_np)


def epps_pulley_1d(h: Float[Array, "batch"]) -> Float[Array, ""]:
    """Epps-Pulley normality test statistic along a 1D projection slice"""
    t, phi, weights = get_epps_pulley_constants()
    x_t = h[:, None] * t[None, :]
    cos_mean = jnp.mean(jnp.cos(x_t), axis=0)
    sin_mean = jnp.mean(jnp.sin(x_t), axis=0)

    err = (cos_mean - phi) ** 2 + sin_mean**2
    return jnp.dot(err, weights) * h.shape[0]


_vmap_slices = jax.vmap(epps_pulley_1d, in_axes=1)
_vmap_subspaces = jax.vmap(_vmap_slices, in_axes=0)


def sub_jepa_regularization(
    target_latents: Float[Array, "batch latent_dim"],
    subspace_projectors: Float[Array, "num_subspaces latent_dim subspace_dim"],
    slice_projectors: Float[Array, "num_subspaces subspace_dim num_slices"],
) -> Float[Array, ""]:
    subspace_latents = _project_into_subspace(target_latents, subspace_projectors)
    sliced_latents = _project_into_slices(subspace_latents, slice_projectors)
    ep_stats = _vmap_subspaces(sliced_latents)
    return jnp.mean(ep_stats)


def sub_jepa_loss(
    predicted_latents: Float[Array, "batch latent_dim"],
    target_latents: Float[Array, "batch latent_dim"],
    subspace_projectors: Float[Array, "num_subspaces latent_dim subspace_dim"],
    slice_projectors: Float[Array, "num_subspaces subspace_dim num_slices"],
    reg_weight: float = 1.0,
) -> Float[Array, ""]:
    pred_loss = jnp.mean(
        (predicted_latents - jax.lax.stop_gradient(target_latents)) ** 2
    )
    reg_loss = sub_jepa_regularization(
        target_latents, subspace_projectors, slice_projectors
    )
    return pred_loss + reg_weight * reg_loss
