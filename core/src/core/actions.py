import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

# steering grid
STEER_VALUES_NP = np.array(
    [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32
)
STEER_VALUES_JAX = jnp.asarray(STEER_VALUES_NP, dtype=jnp.float32)
# combined (gas, brake) grid
GAS_BRAKE_VALUES_NP = np.array(
    [
        [0.0, 0.0],  # 0: Coast / Clean 0
        [1.0, 0.0],  # 1: Full acceleration
        [0.0, 1.0],  # 2: Full braking
        [0.5, 0.0],  # 3: Half acceleration (traction control)
        [1.0, 1.0],  # 4: Drift (full gas + full brake simultaneously)
        [0.5, 1.0],  # 5: Half acceleration + Full braking
    ],
    dtype=np.float32,
)
GAS_BRAKE_VALUES_JAX = jnp.asarray(GAS_BRAKE_VALUES_NP, dtype=jnp.float32)

NUM_STEER = len(STEER_VALUES_NP)
NUM_GAS_BRAKE = len(GAS_BRAKE_VALUES_NP)
NUM_ACTIONS = NUM_STEER * NUM_GAS_BRAKE
ACTION_GRID_NP = np.column_stack(
    (
        np.tile(GAS_BRAKE_VALUES_NP, (NUM_STEER, 1)),
        np.repeat(STEER_VALUES_NP, NUM_GAS_BRAKE),
    )
)
ACTION_GRID_JAX = jnp.asarray(ACTION_GRID_NP, dtype=jnp.float32)
NEUTRAL_ACTION = int(np.flatnonzero(np.all(ACTION_GRID_NP == 0.0, axis=1))[0])


def discretize_action(continuous_action: Float[Array, "... 3"]) -> Int[Array, "..."]:
    """Discretizes a continuous action [gas, brake, steer] into a flat integer index.
    Closeness is determined by minimum absolute difference for steering,
    and minimum Euclidean distance (L2 norm) for gas/brake.

    Args:
        continuous_action: JAX Array of shape (..., 3) containing [gas, brake, steer].

    Returns:
        JAX Array of shape (...) containing flat integer action indices
        in [0, NUM_ACTIONS - 1].
    """
    gas_brake = continuous_action[..., 0:2]
    steer = continuous_action[..., 2]

    steer_diffs = jnp.abs(steer[..., None] - STEER_VALUES_JAX)
    steer_bin = jnp.argmin(steer_diffs, axis=-1)

    gb_diffs = gas_brake[..., None, :] - GAS_BRAKE_VALUES_JAX
    gb_dist = jnp.sum(jnp.square(gb_diffs), axis=-1)
    gb_bin = jnp.argmin(gb_dist, axis=-1)

    return steer_bin * NUM_GAS_BRAKE + gb_bin


def to_continuous_action(discrete_action: Int[Array, "..."]) -> Float[Array, "... 3"]:
    """Reconstructs a continuous action [gas, brake, steer] from a flat integer index.

    Inverse mapping of discretize_action. Maps indices back to the exact
    optimal coordinate targets.

    Args:
        discrete_action: JAX Array of shape (...) containing flat action indices
            in [0, NUM_ACTIONS - 1].

    Returns:
        JAX Array of shape (..., 3) containing continuous actions [gas, brake, steer].
    """
    discrete_action = eqx.error_if(
        discrete_action,
        (discrete_action < 0) | (discrete_action >= NUM_ACTIONS),
        f"Discrete action index must be in [0, {NUM_ACTIONS - 1}]",
    )
    return ACTION_GRID_JAX[discrete_action]


def discretize_action_np(continuous_action: np.ndarray) -> np.ndarray:
    """Discretizes a continuous action [gas, brake, steer] into a flat integer index.

    NumPy implementation of discretize_action for client-side or CPU-only
    compatibility.

    Args:
        continuous_action: NumPy array of shape (..., 3) containing [gas, brake, steer].

    Returns:
        NumPy array of shape (...) containing flat integer action indices
        in [0, NUM_ACTIONS - 1].
    """
    gas_brake = continuous_action[..., 0:2]
    steer = continuous_action[..., 2]

    steer_diffs = np.abs(steer[..., None] - STEER_VALUES_NP)
    steer_bin = np.argmin(steer_diffs, axis=-1)

    gb_diffs = gas_brake[..., None, :] - GAS_BRAKE_VALUES_NP
    gb_dist = np.sum(np.square(gb_diffs), axis=-1)
    gb_bin = np.argmin(gb_dist, axis=-1)

    return steer_bin * NUM_GAS_BRAKE + gb_bin


def to_continuous_action_np(
    discrete_action: np.ndarray | int | np.integer,
) -> np.ndarray:
    """Reconstructs a continuous action [gas, brake, steer] from a flat integer index.

    NumPy implementation of to_continuous_action for client-side or CPU-only
    compatibility.

    Args:
        discrete_action: NumPy array or integer scalar containing flat action index
            in [0, NUM_ACTIONS - 1].

    Returns:
        NumPy array of shape (..., 3) containing continuous actions [gas, brake, steer].
    """
    discrete_arr = np.asarray(discrete_action)
    if not np.all((discrete_arr >= 0) & (discrete_arr < NUM_ACTIONS)):
        raise ValueError(f"Discrete action index out of bounds [0, {NUM_ACTIONS - 1}]")
    return ACTION_GRID_NP[discrete_arr]


def rescale_gas_np(actions: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    """Rescale signed gas to [0, 1], detecting it along the specified gas axis.

    Human Openplanet sessions record raw analog triggers in [-1.0, 1.0] where -1.0
    is unpressed and 1.0 is fully depressed. Agent actions are already in [0.0, 1.0].
    """
    out = actions.copy()
    gas = out[..., 0]
    signed = np.any(gas < -0.01, axis=axis, keepdims=True)
    out[..., 0] = np.where(signed, (gas + 1.0) / 2.0, gas)
    return out
