import threading
import time
from collections import deque
from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from core.interfaces import Encoder, Planner, Predictor


class AsyncPlannerWrapper:
    """
    A Delay-Compensated Asynchronous Action Buffer.
    Runs the heavy JAX Encoder and Planner in a background daemon thread,
    while exposing an O(1) step() function for the real-time control loop.
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: Predictor,
        planner: Planner,
        default_action: int = 25,
        seed: int = 0,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.planner = planner
        self.default_action = default_action

        def _normalize_and_encode(enc: Encoder, obs_dict: Dict[str, jax.Array]):
            norm_obs = {}
            for k, v in obs_dict.items():
                if k == "screen" and jnp.issubdtype(v.dtype, jnp.integer):
                    norm_obs[k] = v.astype(jnp.float32) / 255.0
                elif jnp.issubdtype(v.dtype, jnp.floating) or jnp.issubdtype(
                    v.dtype, jnp.integer
                ):
                    norm_obs[k] = v.astype(jnp.float32)
                else:
                    norm_obs[k] = v
            return enc(norm_obs)

        self._encode_jit = eqx.filter_jit(
            lambda obs_dict: _normalize_and_encode(self.encoder, obs_dict)
        )
        self._predict_jit = eqx.filter_jit(self.predictor)
        self._plan_jit = eqx.filter_jit(self.planner)

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._rng_key = jax.random.PRNGKey(seed)

        self._latest_obs: Optional[Dict[str, np.ndarray]] = None
        self._current_action: int = self.default_action
        self._ticks_passed: int = 0

        self._action_buffer = deque(
            [self.default_action] * self.planner.sequence_len,
            maxlen=self.planner.sequence_len,
        )

    def start(self):
        """Starts the background planning thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._planning_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the background planning thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join()

    def reset(self):
        """Resets planner buffers and internal state between episodes."""
        with self._lock:
            self._action_buffer.clear()
            self._latest_obs = None
            self._ticks_passed = 0
            self._current_action = self.default_action

    def step(self, obs_dict: Dict[str, np.ndarray]) -> int:
        """
        O(1) real-time response.
        Updates the shared observation and pops the next action from the buffer.
        """
        with self._lock:
            self._latest_obs = obs_dict
            self._ticks_passed += 1
            if len(self._action_buffer) > 0:
                next_action = self._action_buffer.popleft()
            else:
                next_action = self.default_action

            self._current_action = next_action

        return next_action

    def _planning_loop(self):
        """
        Background daemon loop.
        Continuously reads the latest observation, fast-forwards 1 tick
        (delay compensation),
        and plans the optimal trajectory from the future state.
        """
        while self._running:
            with self._lock:
                obs = self._latest_obs
                curr_act = self._current_action
                self._ticks_passed = 0

            if obs is None:
                # The environment hasn't provided an observation yet.
                # Avoid spinning the CPU at 100% while waiting for the client connection
                time.sleep(0.001)
                continue

            jax_obs = {}
            for k, v in obs.items():
                arr = jnp.asarray(v)
                if k == "screen" and jnp.issubdtype(arr.dtype, jnp.integer):
                    arr = arr.astype(jnp.float32) / 255.0
                elif jnp.issubdtype(arr.dtype, jnp.floating) or jnp.issubdtype(
                    arr.dtype, jnp.integer
                ):
                    arr = arr.astype(jnp.float32)
                jax_obs[k] = arr
            latent_t = self._encode_jit(jax_obs)

            # We fast-forward the state by 1 tick using the currently executing action.
            # This compensates for the fact that the car physically moves while
            # JAX computes the plan
            action_jax = jnp.asarray(curr_act, dtype=jnp.int32)
            latent_t_plus_1 = self._predict_jit(latent_t, action_jax)

            self._rng_key, plan_key = jax.random.split(self._rng_key)
            action_seq = self._plan_jit(
                latent_t_plus_1, key=plan_key, prev_action=action_jax
            )

            action_seq_list = np.asarray(action_seq).tolist()

            with self._lock:
                ticks = self._ticks_passed
                if ticks < len(action_seq_list):
                    valid_actions = action_seq_list[ticks:]
                else:
                    valid_actions = []

                self._action_buffer.clear()
                self._action_buffer.extend(valid_actions)
