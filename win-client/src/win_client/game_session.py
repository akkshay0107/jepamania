"""Long-lived TMRL session that remains responsive between MPC collections."""

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from win_client.utils import get_tmrl_env

BRAKE_ACTION = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


@dataclass
class _CollectionRequest:
    driver: Any
    rollout_file: Path
    num_episodes: int
    iteration: int
    finished: threading.Event = field(default_factory=threading.Event)
    result: Optional[tuple[int, ...]] = None
    error: Optional[BaseException] = None


class GameSessionWorker:
    """Own a TMRL environment and keep its Openplanet stream active."""

    def __init__(self, env_factory: Callable[[], Any] = get_tmrl_env) -> None:
        self._env_factory = env_factory
        self._requests: queue.Queue[_CollectionRequest] = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._startup_error: Optional[BaseException] = None
        self._fatal_error: Optional[BaseException] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="tmrl-game-session", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                "Unable to start the TMRL game session"
            ) from self._startup_error

    def collect(
        self, driver: Any, rollout_file: Path, num_episodes: int, iteration: int
    ) -> list[int]:
        if self._thread is None:
            raise RuntimeError("Game session has not been started")
        if self._fatal_error is not None:
            raise RuntimeError("TMRL heartbeat has failed") from self._fatal_error
        request = _CollectionRequest(
            driver=driver,
            rollout_file=Path(rollout_file),
            num_episodes=int(num_episodes),
            iteration=int(iteration),
        )
        self._requests.put(request)
        request.finished.wait()
        if request.error is not None:
            raise RuntimeError("MPC rollout collection failed") from request.error
        return list(request.result or ())

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        env = None
        try:
            env = self._env_factory()
            env.reset()
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            if env is not None:
                env.close()
            return

        self._ready.set()
        try:
            while not self._stop.is_set():
                try:
                    request = self._requests.get_nowait()
                except queue.Empty:
                    # Idle control deliberately has no writer or rollout metadata.
                    _, _, terminated, truncated, _ = env.step(BRAKE_ACTION)
                    if terminated or truncated:
                        env.reset()
                    continue

                try:
                    request.result = tuple(
                        request.driver.collect(
                            env,
                            request.rollout_file,
                            request.num_episodes,
                            request.iteration,
                        )
                    )
                    env.reset()
                except BaseException as error:
                    request.error = error
                finally:
                    request.finished.set()
        except BaseException as error:
            self._fatal_error = error
            while True:
                try:
                    pending = self._requests.get_nowait()
                except queue.Empty:
                    break
                pending.error = error
                pending.finished.set()
        finally:
            if env is not None:
                env.close()
