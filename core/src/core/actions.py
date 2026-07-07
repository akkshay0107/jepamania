import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

# Explicit steering grid values (7 targets)
STEER_VALUES_NP = np.array([-1.0, -0.5, -0.15, 0.0, 0.15, 0.5, 1.0], dtype=np.float32)
STEER_VALUES_JAX = jnp.array(
    [-1.0, -0.5, -0.15, 0.0, 0.15, 0.5, 1.0], dtype=jnp.float32
)

# Explicit independent gas/brake target combinations (5 targets)
GAS_BRAKE_VALUES_NP = np.array(
    [
        [0.0, 0.0],  # 0: Coast / Clean 0
        [1.0, 0.0],  # 1: Full acceleration
        [0.0, 1.0],  # 2: Full braking
        [0.5, 0.0],  # 3: Half acceleration (traction control)
        [1.0, 1.0],  # 4: Drift (full gas + full brake simultaneously)
    ],
    dtype=np.float32,
)
GAS_BRAKE_VALUES_JAX = jnp.array(
    [
        [0.0, 0.0],  # 0: Coast / Clean 0
        [1.0, 0.0],  # 1: Full acceleration
        [0.0, 1.0],  # 2: Full braking
        [0.5, 0.0],  # 3: Half acceleration (traction control)
        [1.0, 1.0],  # 4: Drift (full gas + full brake simultaneously)
    ],
    dtype=jnp.float32,
)


def discretize_action(continuous_action: Float[Array, "... 3"]) -> Int[Array, "..."]:
    """Discretizes a continuous action [steer, gas, brake] into a flat integer index.

    Steering is snapped to the closest of 7 custom values:
        [-1.0, -0.5, -0.15, 0.0, 0.15, 0.5, 1.0]

    Gas/Brake are snapped to the closest of 5 independent coordinate combinations:
        [0.0, 0.0] (Coast)
        [1.0, 0.0] (Full Gas)
        [0.0, 1.0] (Full Brake)
        [0.5, 0.0] (Half Gas / Traction control)
        [1.0, 1.0] (Drift / Left-foot braking)

    Closeness is determined by minimum absolute difference for steering,
    and minimum Euclidean distance (L2 norm) for gas/brake.

    Args:
        continuous_action: JAX Array of shape (..., 3) containing [steer, gas, brake].

    Returns:
        JAX Array of shape (...) containing flat integer action indices in [0, 34].
    """
    steer = continuous_action[..., 0]
    gas_brake = continuous_action[..., 1:3]

    steer_diffs = jnp.abs(steer[..., None] - STEER_VALUES_JAX)
    steer_bin = jnp.argmin(steer_diffs, axis=-1)

    gb_diffs = gas_brake[..., None, :] - GAS_BRAKE_VALUES_JAX
    gb_dist = jnp.sum(jnp.square(gb_diffs), axis=-1)
    gb_bin = jnp.argmin(gb_dist, axis=-1)

    return steer_bin * 5 + gb_bin


def to_continuous_action(discrete_action: Int[Array, "..."]) -> Float[Array, "... 3"]:
    """Reconstructs a continuous action [steer, gas, brake] from a flat integer index.

    Inverse mapping of discretize_action. Maps indices back to the exact
    optimal coordinate targets.

    Args:
        discrete_action: JAX Array of shape (...) containing flat action indices
            in [0, 34].

    Returns:
        JAX Array of shape (..., 3) containing continuous actions [steer, gas, brake].
    """
    # Clamp to valid range [0, 34] to prevent JAX out-of-bounds indexing errors
    discrete_action = jnp.clip(discrete_action, 0, 34)
    steer_bin = discrete_action // 5
    gb_bin = discrete_action % 5

    steer = STEER_VALUES_JAX[steer_bin]
    gb_pair = GAS_BRAKE_VALUES_JAX[gb_bin]
    gas = gb_pair[..., 0]
    brake = gb_pair[..., 1]

    return jnp.stack([steer, gas, brake], axis=-1)


def discretize_action_np(continuous_action: np.ndarray) -> np.ndarray:
    """Discretizes a continuous action [steer, gas, brake] into a flat integer index.

    NumPy implementation of discretize_action for client-side or CPU-only
    compatibility.

    Args:
        continuous_action: NumPy array of shape (..., 3) containing [steer, gas, brake].

    Returns:
        NumPy array of shape (...) containing flat integer action indices in [0, 34].
    """
    steer = continuous_action[..., 0]
    gas_brake = continuous_action[..., 1:3]

    steer_diffs = np.abs(steer[..., None] - STEER_VALUES_NP)
    steer_bin = np.argmin(steer_diffs, axis=-1)

    gb_diffs = gas_brake[..., None, :] - GAS_BRAKE_VALUES_NP
    gb_dist = np.sum(np.square(gb_diffs), axis=-1)
    gb_bin = np.argmin(gb_dist, axis=-1)

    return steer_bin * 5 + gb_bin


def to_continuous_action_np(discrete_action: np.ndarray) -> np.ndarray:
    """Reconstructs a continuous action [steer, gas, brake] from a flat integer index.

    NumPy implementation of to_continuous_action for client-side or CPU-only
    compatibility.

    Args:
        discrete_action: NumPy array of shape (...) containing flat action indices
            in [0, 34].

    Returns:
        NumPy array of shape (..., 3) containing continuous actions [steer, gas, brake].
    """
    if not np.all((discrete_action >= 0) & (discrete_action < 35)):
        raise ValueError("Discrete action index out of bounds [0, 34]")
    steer_bin = discrete_action // 5
    gb_bin = discrete_action % 5

    steer = STEER_VALUES_NP[steer_bin]
    gb_pair = GAS_BRAKE_VALUES_NP[gb_bin]
    gas = gb_pair[..., 0]
    brake = gb_pair[..., 1]

    return np.stack([steer, gas, brake], axis=-1)
