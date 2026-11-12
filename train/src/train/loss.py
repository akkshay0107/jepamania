import functools

import jax
import jax.numpy as jnp
import numpy as np
from core.config import ACTION_HISTORY_FEATURES
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


def generate_subspace_projectors(
    key: PRNGKeyArray,
    latent_dim: int,
    num_subspaces: int,
    subspace_dim: int,
) -> Float[Array, "num_subspaces latent_dim subspace_dim"]:
    """Generate the frozen orthogonal subspace bases used by Sub-JEPA."""
    keys_subspace = jax.random.split(key, num_subspaces)
    return _make_projector(keys_subspace, latent_dim, subspace_dim)


def sample_slice_projectors(
    key: PRNGKeyArray,
    subspace_dim: int,
    num_subspaces: int,
    num_slices: int,
) -> Float[Array, "num_subspaces subspace_dim num_slices"]:
    """Sample fresh unit slice directions for one regularizer evaluation."""
    keys_slices = jax.random.split(key, num_subspaces)
    return _make_slice_projector(keys_slices, subspace_dim, num_slices)


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


def _epps_pulley_1d(h: Float[Array, "batch"]) -> Float[Array, ""]:
    t, phi, weights = get_epps_pulley_constants()
    # At zero frequency the characteristic-function error is identically zero.
    x_t = h[:, None] * t[None, 1:]
    cos_error = jnp.mean(jnp.cos(x_t), axis=0) - phi[1:]
    sin_mean = jnp.mean(jnp.sin(x_t), axis=0)
    return jnp.dot(cos_error**2 + sin_mean**2, weights[1:]) * h.size


_vmap_slices = jax.vmap(_epps_pulley_1d, in_axes=1)
_vmap_subspaces = jax.vmap(_vmap_slices)


def sub_jepa_regularization(
    target_latents: Float[Array, "batch latent_dim"],
    subspace_projectors: Float[Array, "num_subspaces latent_dim subspace_dim"],
    num_slices: int,
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    """Match available learned targets to Gaussian projections."""
    subspace_latents = _project_into_subspace(target_latents, subspace_projectors)
    slice_projectors = sample_slice_projectors(
        key,
        subspace_dim=subspace_latents.shape[-1],
        num_subspaces=subspace_latents.shape[0],
        num_slices=num_slices,
    )
    sliced_latents = _project_into_slices(subspace_latents, slice_projectors)
    ep_stats = _vmap_subspaces(sliced_latents)
    return jnp.mean(ep_stats)


def sub_jepa_components(
    predicted_latents: Float[Array, "batch state_dim"],
    target_latents: Float[Array, "batch state_dim"],
    subspace_projectors: Float[Array, "num_subspaces latent_dim subspace_dim"],
    num_slices: int,
    key: PRNGKeyArray,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Return prediction and Gaussian losses on learned coordinates."""
    latent_dim = subspace_projectors.shape[1]
    if predicted_latents.shape != target_latents.shape:
        raise ValueError("Predicted and target model states must have matching shapes.")
    if target_latents.shape[-1] not in (
        latent_dim,
        latent_dim + ACTION_HISTORY_FEATURES,
    ):
        raise ValueError("State width must match the learned latent plus history.")
    target_latents = target_latents[:, :latent_dim]
    prediction_error = predicted_latents[:, :latent_dim] - jax.lax.stop_gradient(
        target_latents
    )
    return jnp.mean(prediction_error**2), sub_jepa_regularization(
        target_latents, subspace_projectors, num_slices, key
    )
