import logging
import queue
import threading
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
from core.config import TELEMETRY_FEATURES
from src.settings import cfg

IMG_SIZE: int = 64

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class HDF5Writer:
    """Asynchronous HDF5 writer with per-episode metadata support."""

    def __init__(
        self,
        filepath: Path,
        chunk_size: int = cfg.hdf5_chunk_size,
        obs_type: str = "screen",
    ) -> None:
        if obs_type not in ("screen", "lidar"):
            raise ValueError("obs_type must be either 'screen' or 'lidar'.")
        self.obs_type = obs_type
        self.filepath = filepath
        self.chunk_size = chunk_size

        # Queue shared between main thread (producer) and writer thread (consumer).
        self._queue: queue.Queue = queue.Queue()

        # Per-type frame buffers (populated by the writer thread only).
        self._obs_buf: list[np.ndarray] = []
        self._telemetry_buf: list[np.ndarray] = []
        self._action_buf: list[np.ndarray] = []
        self._episode_id_buf: list[int] = []

        self.current_size: int = 0
        self._current_episode_id: int = -1

        self._running: bool = True
        self._closed: bool = False

        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(filepath, "w")

        obs_grp = self._file.create_group("observations")
        if self.obs_type == "lidar":
            from core.config import LIDAR_BEAMS

            obs_grp.create_dataset(
                "lidar",
                shape=(0, LIDAR_BEAMS),
                maxshape=(None, LIDAR_BEAMS),
                chunks=(chunk_size, LIDAR_BEAMS),
                dtype=np.float32,
            )
        else:
            obs_grp.create_dataset(
                "screen",
                shape=(0, 1, IMG_SIZE, IMG_SIZE),
                maxshape=(None, 1, IMG_SIZE, IMG_SIZE),
                chunks=(chunk_size, 1, IMG_SIZE, IMG_SIZE),
                dtype=np.uint8,
                compression="gzip",
            )
        obs_grp.create_dataset(
            "telemetry",
            shape=(0, TELEMETRY_FEATURES),
            maxshape=(None, TELEMETRY_FEATURES),
            chunks=(chunk_size, TELEMETRY_FEATURES),
            dtype=np.float32,
        )
        self._file.create_dataset(
            "actions",
            shape=(0, cfg.action_dim),
            maxshape=(None, cfg.action_dim),
            chunks=(chunk_size, cfg.action_dim),
            dtype=np.float32,
            compression="gzip",
        )
        self._file.create_dataset(
            "episode_id",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_size,),
            dtype=np.int32,
        )
        self._file.create_group("metadata")

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        logging.info(f"HDF5Writer started → {filepath}")

    def new_episode(self, metadata: dict[str, Any]) -> None:
        """Signal the start of a new episode."""
        self._current_episode_id += 1
        self._queue.put(
            {
                "_type": "episode_start",
                "episode_id": self._current_episode_id,
                "metadata": dict(metadata),
            }
        )

    def end_episode(self, termination: str = "done") -> None:
        """
        Signal the end of the current episode.

        The writer thread will flush all pending frame data for this episode
        before recording frame_end in the metadata group.
        """
        if self._current_episode_id < 0:
            return
        self._queue.put(
            {
                "_type": "episode_end",
                "episode_id": self._current_episode_id,
                "termination": termination,
            }
        )

    def append(self, obs_dict: dict[str, np.ndarray], action: np.ndarray) -> None:
        """Queue a single frame for writing."""
        qsize = self._queue.qsize()
        if qsize > 500:
            logging.warning(
                f"I/O Warning: Queue size is {qsize} frames! "
                "Writing to disk is falling behind the real-time game loop."
            )

        frame_data = {
            "_type": "frame",
            "telemetry": np.copy(obs_dict["telemetry"]),
            "action": np.copy(action).astype(np.float32),
            "episode_id": self._current_episode_id,
        }
        if self.obs_type == "lidar":
            frame_data["lidar"] = np.copy(obs_dict["lidar"])
        else:
            frame_data["screen"] = np.copy(obs_dict["screen"])
        self._queue.put(frame_data)

    def close(self) -> None:
        """Drain the queue, flush remaining data, and close the file."""
        if self._closed:
            return
        self._closed = True
        self._running = False
        if self._thread.is_alive():
            self._thread.join()
        self._flush()
        self._file.close()
        logging.info(f"HDF5Writer closed. Total frames written: {self.current_size}")

    def _writer_loop(self) -> None:
        """Background thread: processes tokens and frame data from the queue."""
        # Maps episode_id → pending metadata dict (frame_start unknown until flush).
        pending: dict[int, dict] = {}

        while self._running or not self._queue.empty():
            try:
                token = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            ttype = token["_type"]

            if ttype == "episode_start":
                # Flush any residual frames from the previous episode so that
                # current_size reflects the true boundary before recording frame_start.
                self._flush()
                pending[token["episode_id"]] = {
                    **token["metadata"],
                    "frame_start": self.current_size,
                }

            elif ttype == "episode_end":
                # Flush this episode's buffered frames before recording frame_end.
                self._flush()
                ep_id = token["episode_id"]
                if ep_id in pending:
                    meta = pending.pop(ep_id)
                    meta["frame_end"] = self.current_size
                    meta["termination"] = token["termination"]

                    # h5py groups and attributes typing is loose in pyright
                    grp = cast(h5py.Group, self._file["metadata"]).create_group(
                        f"episode_{ep_id}"
                    )
                    for k, v in meta.items():
                        # HDF5 attributes must be scalar or string.
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            grp.attrs[k] = v
                        else:
                            grp.attrs[k] = str(v)
                    self._file.flush()

            else:
                if self.obs_type == "lidar":
                    self._obs_buf.append(token["lidar"])
                else:
                    self._obs_buf.append(token["screen"])
                self._telemetry_buf.append(token["telemetry"])
                self._action_buf.append(token["action"])
                self._episode_id_buf.append(token["episode_id"])

                if len(self._obs_buf) >= self.chunk_size:
                    self._flush()

    def _flush(self) -> None:
        """Write all buffered frames to HDF5 and clear the buffers."""
        n = len(self._obs_buf)
        if n == 0:
            return

        new_size = self.current_size + n

        # casting to avoid pyright errors
        obs = cast(h5py.Group, self._file["observations"])

        if self.obs_type == "lidar":
            obs_ds = cast(h5py.Dataset, obs["lidar"])
        else:
            obs_ds = cast(h5py.Dataset, obs["screen"])
        telem_ds = cast(h5py.Dataset, obs["telemetry"])
        actions_ds = cast(h5py.Dataset, self._file["actions"])
        ep_id_ds = cast(h5py.Dataset, self._file["episode_id"])

        obs_ds.resize(new_size, axis=0)
        telem_ds.resize(new_size, axis=0)
        actions_ds.resize(new_size, axis=0)
        ep_id_ds.resize(new_size, axis=0)

        obs_ds[self.current_size : new_size] = np.stack(self._obs_buf)
        telem_ds[self.current_size : new_size] = np.stack(self._telemetry_buf)
        actions_ds[self.current_size : new_size] = np.stack(self._action_buf)
        ep_id_ds[self.current_size : new_size] = np.array(
            self._episode_id_buf, dtype=np.int32
        )

        self.current_size = new_size

        self._obs_buf.clear()
        self._telemetry_buf.clear()
        self._action_buf.clear()
        self._episode_id_buf.clear()
