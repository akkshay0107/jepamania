"""Plan once per fresh observation on a background worker."""

import logging
import threading
from collections import deque

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray, Shaped

from core.actions import ACTION_GRID_JAX, NEUTRAL_ACTION, NUM_ACTIONS
from core.config import IMG_HIST_LEN, IMG_SIZE, TELEMETRY_FEATURES
from core.interfaces import Encoder, Planner, Predictor


@eqx.filter_jit
def _plan_step(
    encoder: Encoder,
    predictor: Predictor,
    planner: Planner,
    screen: Shaped[np.ndarray | Array, "4 64 64"],
    telemetry: Float[np.ndarray | Array, "9"],
    previous_action: Int[np.ndarray | Array, ""],
    key: PRNGKeyArray,
) -> tuple[Int[Array, "sequence_len"], PRNGKeyArray]:
    frames = jnp.asarray(screen, dtype=jnp.float32)
    if jnp.issubdtype(screen.dtype, jnp.integer):
        frames = frames / 255.0
    latent = encoder(
        {"screen": frames, "telemetry": jnp.asarray(telemetry, dtype=jnp.float32)}
    )
    # The control chosen for this tick will be applied while the worker plans.
    next_latent = predictor(latent, ACTION_GRID_JAX[previous_action])
    next_key, plan_key = jax.random.split(key)
    return planner(next_latent, key=plan_key, prev_action=previous_action), next_key


class AsyncPlannerWrapper:
    """Buffer actions from an asynchronous, one-tick-ahead latent plan."""

    def __init__(
        self,
        encoder: Encoder,
        predictor: Predictor,
        planner: Planner,
        default_action: int = NEUTRAL_ACTION,
        seed: int = 0,
    ) -> None:
        if not 0 <= default_action < NUM_ACTIONS:
            raise ValueError("Fallback action must be a valid action ID.")
        self.encoder = encoder
        self.predictor = predictor
        self.planner = planner
        self.default_action = default_action

        self._condition = threading.Condition()
        self._running = False
        self._warmed_up = False
        self._thread: threading.Thread | None = None
        self._rng_key = jax.random.key(seed)
        self._thread_error: Exception | None = None
        self._reset_generation = 0

        self._latest_obs: dict[str, np.ndarray] | None = None
        self._current_action = default_action
        self._ticks_passed = 0
        self._action_buffer = deque([default_action] * planner.sequence_len)

    def start(self) -> None:
        """Warm up the live input signature and start the background worker."""
        if self._thread is not None:
            return
        if not self._warmed_up:
            self._rng_key, warmup_key = jax.random.split(self._rng_key)
            result = _plan_step(
                self.encoder,
                self.predictor,
                self.planner,
                np.zeros((IMG_HIST_LEN, IMG_SIZE, IMG_SIZE), dtype=np.uint8),
                np.zeros(TELEMETRY_FEATURES, dtype=np.float32),
                np.asarray(self.default_action, dtype=np.int32),
                warmup_key,
            )
            jax.block_until_ready(result)
            self._warmed_up = True

        self._thread_error = None
        self._running = True
        self._thread = threading.Thread(target=self._planning_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Wake the worker and wait for any running plan to finish."""
        with self._condition:
            self._running = False
            self._latest_obs = None
            self._condition.notify()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def reset(self) -> None:
        """Discard observations and plans from the previous episode."""
        with self._condition:
            self._reset_generation += 1
            self._action_buffer.clear()
            self._action_buffer.extend(
                [self.default_action] * self.planner.sequence_len
            )
            self._latest_obs = None
            self._ticks_passed = 0
            self._current_action = self.default_action

    def step(self, obs_dict: dict[str, np.ndarray]) -> int:
        """Publish the latest observation and return the next buffered action."""
        with self._condition:
            if self._thread_error is not None:
                raise RuntimeError("Background planner failed") from self._thread_error
            self._latest_obs = obs_dict
            self._ticks_passed += 1
            self._current_action = (
                self._action_buffer.popleft()
                if self._action_buffer
                else self.default_action
            )
            self._condition.notify()
            return self._current_action

    def _planning_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: not self._running or self._latest_obs is not None
                    )
                    if not self._running or self._latest_obs is None:
                        return
                    observation = self._latest_obs
                    self._latest_obs = None
                    action = self._current_action
                    generation = self._reset_generation
                    self._ticks_passed = 0

                action_seq, self._rng_key = _plan_step(
                    self.encoder,
                    self.predictor,
                    self.planner,
                    observation["screen"],
                    observation["telemetry"],
                    np.asarray(action, dtype=np.int32),
                    self._rng_key,
                )
                actions = np.asarray(action_seq).tolist()

                with self._condition:
                    if self._running and generation == self._reset_generation:
                        self._action_buffer.clear()
                        self._action_buffer.extend(actions[self._ticks_passed :])
        except Exception as error:
            # Surface worker failures to the driver on its next tick.
            with self._condition:
                self._thread_error = error
                self._running = False
            logging.exception("Background planner loop failed")
