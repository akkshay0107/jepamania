"""Synchronous discrete actions and explicit Trackmania episode boundaries."""

from collections import deque
from typing import Any, NamedTuple

import numpy as np
from core.actions import ACTION_GRID_NP
from core.config import IMG_HIST_LEN, IMG_SIZE

from win_client.utils import obs_to_dict

REWARD_DEFINITION = "step:-0.001,finish:+1,genuine_failure:-1"
IDLE_ACTION = np.zeros(3, dtype=np.float32)
GAME_HZ = 20
MAX_EPISODE_STEPS = GAME_HZ * 120
COLLECTION_CUTS = frozenset({"stuck", "frame_budget", "respawn", "timeout"})
type Observation = dict[str, np.ndarray]


class Step(NamedTuple):
    observation: Observation | None
    action_id: int | None
    reward: float
    done: bool
    reason: str | None
    action_error: float | None


class DrivingEnvironment:
    """Build histories from the observations and controls of one episode."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.frames: deque[np.ndarray] = deque(maxlen=IMG_HIST_LEN)
        self.controls: deque[np.ndarray] = deque(maxlen=2)
        self.steps = 0
        self.pending_action: int | None = None
        interface = getattr(env, "unwrapped", env)
        self.native_rtgym = type(interface).__module__.startswith("rtgym.")

    def _observation(self, raw: Any, *, reset: bool = False) -> Observation:
        obs = obs_to_dict(raw)
        if obs["screen"].shape != (1, IMG_SIZE, IMG_SIZE):
            raise ValueError("The game must return a grayscale 64×64 observation.")
        frame = obs["screen"][0].copy()
        if reset:
            self.frames.clear()
            self.frames.extend([frame] * IMG_HIST_LEN)
        else:
            self.frames.append(frame)
        return {
            "screen": np.stack(self.frames),
            "telemetry": np.concatenate((obs["telemetry"][:3], *self.controls)),
        }

    def reset(self) -> Observation:
        self.steps = 0
        self.pending_action = None
        self.controls.clear()
        self.controls.extend([IDLE_ACTION] * 2)
        raw, _ = self.env.reset()
        if raw is None:
            raise RuntimeError("The game reset did not return an observation.")
        return self._observation(raw, reset=True)

    def step(self, action_id: int) -> Step:
        if not 0 <= action_id < len(ACTION_GRID_NP):
            raise ValueError(f"Action ID outside the discrete table: {action_id}")
        action = ACTION_GRID_NP[action_id]
        try:
            raw, _, terminated, truncated, info = self.env.step(action.copy())
        except (TimeoutError, ConnectionError):
            return Step(None, None, 0.0, False, "missing_observation", None)
        self.steps += 1
        if raw is None:
            return Step(None, None, 0.0, False, "missing_observation", None)

        applied_id: int | None = action_id
        if self.native_rtgym:
            # Native step returns the completed interval while sending the next action.
            applied_id, self.pending_action = self.pending_action, action_id

        reason = info.get("termination_reason")
        finished = bool(info.get("end_of_track") or reason == "done")
        failure = bool(
            info.get("genuine_failure")
            or (terminated and reason not in COLLECTION_CUTS and not finished)
        )
        done = finished or failure
        if finished:
            reason = "done"
        elif failure:
            reason = "failure"
        elif reason not in COLLECTION_CUTS:
            reason = "timeout" if truncated or self.steps >= MAX_EPISODE_STEPS else None
        reward = -0.001 + float(finished) - float(failure and not finished)
        if applied_id is None:
            # Reset primes a neutral interval before any selected action has run.
            return Step(
                self._observation(raw, reset=True), None, reward, done, reason, None
            )
        applied_action = ACTION_GRID_NP[applied_id]
        self.controls.append(applied_action)
        measured = info.get("action")
        action_error = (
            None
            if measured is None
            else float(np.max(np.abs(measured - applied_action)))
        )
        return Step(
            self._observation(raw), applied_id, reward, done, reason, action_error
        )

    def close(self) -> None:
        self.env.close()
