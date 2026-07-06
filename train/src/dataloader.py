import queue
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

import h5py
import numpy as np
from core.actions import discretize_action_np


class SlidingWindowDataset:
    """Indexes and retrieves transitions across multiple HDF5 shards."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        history_len: int = 4,
        rollout_len: int = 5,
        discretize_actions: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.H = int(history_len)
        self.K = int(rollout_len)
        self.discretize_actions = bool(discretize_actions)

        self.shards: List[Dict[str, Any]] = []
        self.shard_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.local_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.episode_indices_map: np.ndarray = np.empty(0, dtype=np.int32)

        self._build_index()

    def _build_index(self) -> None:
        if not self.data_dir.exists():
            return

        h5_files = sorted(list(self.data_dir.glob("*.h5")))
        shard_indices = []
        local_indices = []
        episode_indices = []

        for shard_idx, file_path in enumerate(h5_files):
            try:
                with h5py.File(file_path, "r") as f:
                    if "episode_id" not in f or "observations/screen" not in f:
                        continue

                    ep_id_ds = cast(h5py.Dataset, f["episode_id"])
                    episode_ids = np.asarray(ep_id_ds, dtype=np.int32)
                    total_frames = len(episode_ids)
                    if total_frames <= self.K:
                        continue

                    indices = np.arange(total_frames - self.K)
                    valid_mask = episode_ids[indices] == episode_ids[indices + self.K]
                    valid_local_ts = indices[valid_mask]

                    if len(valid_local_ts) == 0:
                        continue

                    episode_boundaries = {}
                    if "metadata" in f:
                        metadata_grp = cast(h5py.Group, f["metadata"])
                        for ep_name in metadata_grp.keys():
                            ep_id = int(ep_name.split("_")[1])
                            ep_grp = cast(h5py.Group, metadata_grp[ep_name])
                            frame_start = int(cast(Any, ep_grp.attrs["frame_start"]))
                            frame_end = int(cast(Any, ep_grp.attrs["frame_end"]))
                            episode_boundaries[ep_id] = (frame_start, frame_end)

                    for ep_id in np.unique(episode_ids):
                        if ep_id not in episode_boundaries:
                            first_idx = int(
                                np.searchsorted(episode_ids, ep_id, side="left")
                            )
                            last_idx = int(
                                np.searchsorted(episode_ids, ep_id, side="right")
                            )
                            episode_boundaries[ep_id] = (first_idx, last_idx)

                    shard_data = {
                        "file_path": file_path,
                        "episode_boundaries": episode_boundaries,
                        "episode_ids": episode_ids,
                    }

                    self.shards.append(shard_data)

                    shard_indices.append(
                        np.full_like(
                            valid_local_ts, len(self.shards) - 1, dtype=np.int32
                        )
                    )
                    local_indices.append(valid_local_ts)
                    episode_indices.append(episode_ids[valid_local_ts])

            except Exception as e:
                # Avoid blocking initialization if a single shard is corrupted
                print(f"Warning: Failed to index HDF5 shard {file_path}: {e}")

        if shard_indices:
            self.shard_indices_map = np.concatenate(shard_indices, axis=0)
            self.local_indices_map = np.concatenate(local_indices, axis=0)
            self.episode_indices_map = np.concatenate(episode_indices, axis=0)

    def split(
        self, val_ratio: float = 0.1, seed: int = 42
    ) -> Tuple["SlidingWindowDataset", "SlidingWindowDataset"]:
        """Splits into train and val sets by episode."""
        if val_ratio <= 0.0 or val_ratio >= 1.0:
            raise ValueError("val_ratio must be between 0.0 and 1.0 exclusive.")

        if len(self) == 0:
            return self, self

        pairs = np.stack([self.shard_indices_map, self.episode_indices_map], axis=1)
        unique_pairs, inverse_indices = np.unique(pairs, axis=0, return_inverse=True)
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(len(unique_pairs))

        num_val = int(len(unique_pairs) * val_ratio)
        if num_val == 0 and len(unique_pairs) > 1:
            num_val = 1

        val_pair_indices = shuffled_indices[:num_val]
        val_mask = np.isin(inverse_indices, val_pair_indices)
        train_mask = ~val_mask

        train_ds = SlidingWindowDataset.__new__(SlidingWindowDataset)
        val_ds = SlidingWindowDataset.__new__(SlidingWindowDataset)

        for ds, mask in ((train_ds, train_mask), (val_ds, val_mask)):
            ds.data_dir = self.data_dir
            ds.H = self.H
            ds.K = self.K
            ds.discretize_actions = self.discretize_actions
            ds.shards = self.shards
            ds.shard_indices_map = self.shard_indices_map[mask]
            ds.local_indices_map = self.local_indices_map[mask]
            ds.episode_indices_map = self.episode_indices_map[mask]

        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self.shard_indices_map)

    def _get_history_stack(
        self, f: h5py.File, idx: int, frame_start: int
    ) -> np.ndarray:
        slice_start = max(idx - self.H + 1, frame_start)
        slice_end = idx
        slice_len = slice_end - slice_start + 1

        screen_ds = cast(h5py.Dataset, f["observations/screen"])
        raw_slice = screen_ds[slice_start : slice_end + 1]

        # Squeezing avoids extra axis overhead while preserving features
        raw_slice_sq = raw_slice[:, 0]

        if slice_len < self.H:
            pad_len = self.H - slice_len
            first_frame = raw_slice_sq[0]
            # Leverage numpy broadcasting for memory-efficient padding repetition
            padding = np.repeat(first_frame[np.newaxis, ...], pad_len, axis=0)
            obs_stack = np.concatenate([padding, raw_slice_sq], axis=0)
        else:
            obs_stack = raw_slice_sq

        return obs_stack

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shard_idx = self.shard_indices_map[idx]
        local_t = self.local_indices_map[idx]
        shard = self.shards[shard_idx]

        ep_id = shard["episode_ids"][local_t]
        frame_start, _ = shard["episode_boundaries"][ep_id]

        with h5py.File(shard["file_path"], "r") as f:
            obs_stack_t = self._get_history_stack(f, local_t, frame_start)

            telem_ds = cast(h5py.Dataset, f["observations/telemetry"])
            telemetry_t = np.asarray(telem_ds[local_t], dtype=np.float32)
            telemetry_target = np.asarray(telem_ds[local_t + self.K], dtype=np.float32)
            actions_ds = cast(h5py.Dataset, f["actions"])
            actions_seq = np.asarray(
                actions_ds[local_t : local_t + self.K], dtype=np.float32
            )

            obs_stack_target = self._get_history_stack(f, local_t + self.K, frame_start)

        if self.discretize_actions:
            actions_seq = discretize_action_np(actions_seq)

        return obs_stack_t, telemetry_t, actions_seq, obs_stack_target, telemetry_target


class DataLoader:
    """Prefetches batches of transitions in a background thread."""

    def __init__(
        self,
        dataset: SlidingWindowDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.num_workers = int(num_workers)
        # Owned generator (not global np.random) so shuffle order is
        # reproducible given a seed; state advances across epochs.
        self.rng = np.random.default_rng(seed)

        self.indices = np.arange(len(self.dataset))
        self.queue: queue.Queue = queue.Queue(maxsize=8)
        self.thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()

    def __iter__(self) -> Generator[Dict[str, np.ndarray], None, None]:
        if len(self.dataset) == 0:
            return

        indices = np.copy(self.indices)
        if self.shuffle:
            self.rng.shuffle(indices)

        num_batches = len(indices) // self.batch_size
        if not self.drop_last and len(indices) % self.batch_size != 0:
            num_batches += 1

        if num_batches == 0:
            return

        if self.num_workers > 0:
            self.shutdown_event.clear()
            # Background thread hides latency of CPU-based batch assembly
            self.thread = threading.Thread(
                target=self._prefetch_loop, args=(indices, num_batches), daemon=True
            )
            self.thread.start()

            try:
                for _ in range(num_batches):
                    batch = self.queue.get()
                    if batch is None:
                        break
                    yield batch
            finally:
                # Runs even if the consumer abandons the generator mid-epoch;
                # join before draining so the worker cannot enqueue a stale
                # item (or sentinel) after the queue has been emptied.
                self.shutdown_event.set()
                self.thread.join()
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break
        else:
            for i in range(num_batches):
                batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
                yield self._collate(batch_indices)

    def _put_blocking(self, item: Optional[Dict[str, np.ndarray]]) -> bool:
        """Puts an item, waiting for queue space unless shutdown is requested."""
        while not self.shutdown_event.is_set():
            try:
                self.queue.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_loop(self, indices: np.ndarray, num_batches: int) -> None:
        for i in range(num_batches):
            if self.shutdown_event.is_set():
                break
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
            try:
                batch = self._collate(batch_indices)
            except Exception as e:
                # Sentinel below signals the error so the generator can exit
                print(f"Warning: DataLoader worker failed to collate batch: {e}")
                break
            # A slow consumer (e.g. JIT compilation) must not terminate the
            # epoch early; wait for space instead of treating Full as an error.
            if not self._put_blocking(batch):
                break
        self._put_blocking(None)

    def _collate(self, batch_indices: np.ndarray) -> Dict[str, np.ndarray]:
        obs_stack_t_list = []
        telemetry_t_list = []
        actions_seq_list = []
        obs_stack_target_list = []
        telemetry_target_list = []

        for idx in batch_indices:
            obs_t, telem_t, act_seq, obs_target, telem_target = self.dataset[idx]
            obs_stack_t_list.append(obs_t)
            telemetry_t_list.append(telem_t)
            actions_seq_list.append(act_seq)
            obs_stack_target_list.append(obs_target)
            telemetry_target_list.append(telem_target)

        return {
            "obs_stack_t": np.stack(obs_stack_t_list, axis=0),
            "telemetry_t": np.stack(telemetry_t_list, axis=0),
            "actions_seq": np.stack(actions_seq_list, axis=0),
            "obs_stack_target": np.stack(obs_stack_target_list, axis=0),
            "telemetry_target": np.stack(telemetry_target_list, axis=0),
        }
