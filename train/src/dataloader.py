"""
HDF5-backed sliding-window dataset and prefetching dataloader.

Reads observation/telemetry/action transitions from sharded HDF5 files
without preloading into RAM.  The dataset builds a flat index at init time
and serves individual samples via lazy file reads.
"""

import queue
import threading
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

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
        obs_type: str = "screen",
    ) -> None:
        if obs_type not in ("screen", "lidar"):
            raise ValueError("obs_type must be either 'screen' or 'lidar'.")
        self.obs_type = obs_type
        self.data_dir = Path(data_dir)
        self.H = int(history_len)
        self.K = int(rollout_len)
        self.discretize_actions = bool(discretize_actions)

        self.shards: list[Dict[str, Any]] = []
        self.shard_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.local_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.episode_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.frame_starts_map: np.ndarray = np.empty(0, dtype=np.int32)

        self._build_index()

    def _build_index(self) -> None:
        if not self.data_dir.exists():
            return

        h5_files = sorted(self.data_dir.glob("*.h5"))
        shard_indices: list[np.ndarray] = []
        local_indices: list[np.ndarray] = []
        episode_indices: list[np.ndarray] = []

        for shard_idx, file_path in enumerate(h5_files):
            try:
                with h5py.File(file_path, "r") as f:
                    obs_key = f"observations/{self.obs_type}"
                    if "episode_id" not in f or obs_key not in f:
                        continue

                    episode_ids: np.ndarray = np.asarray(
                        f["episode_id"], dtype=np.int32
                    )
                    total_frames = len(episode_ids)
                    if total_frames <= self.K:
                        continue

                    indices = np.arange(total_frames - self.K)
                    valid_mask = episode_ids[indices] == episode_ids[indices + self.K]
                    valid_local_ts = indices[valid_mask]

                    if len(valid_local_ts) == 0:
                        continue

                    episode_boundaries = self._read_episode_boundaries(f, episode_ids)

                    shard_data = {
                        "file_path": file_path,
                        "episode_boundaries": episode_boundaries,
                    }
                    self.shards.append(shard_data)

                    shard_indices.append(
                        np.full(len(valid_local_ts), shard_idx, dtype=np.int32)
                    )
                    local_indices.append(valid_local_ts.astype(np.int32))
                    episode_indices.append(episode_ids[valid_local_ts])
            except (OSError, KeyError) as e:
                print(f"Warning: Failed to read {file_path}, skipping shard: {e}")

        if shard_indices:
            self.shard_indices_map = np.concatenate(shard_indices)
            self.local_indices_map = np.concatenate(local_indices)
            self.episode_indices_map = np.concatenate(episode_indices)
            self._build_frame_starts_map()

    @staticmethod
    def _read_episode_boundaries(
        f: h5py.File, episode_ids: np.ndarray
    ) -> Dict[int, Tuple[int, int]]:
        episode_boundaries: Dict[int, Tuple[int, int]] = {}

        if "metadata" in f:
            metadata_grp = f["metadata"]
            for ep_name in metadata_grp.keys():  # type: ignore[union-attr]
                ep_id = int(ep_name.split("_")[1])
                ep_grp = metadata_grp[ep_name]  # type: ignore[union-attr]
                frame_start = int(ep_grp.attrs["frame_start"])  # type: ignore[union-attr]
                frame_end = int(ep_grp.attrs["frame_end"])  # type: ignore[union-attr]
                episode_boundaries[ep_id] = (frame_start, frame_end)

        # fallback for shards missing metadata or with incomplete
        # episode coverage — derive boundaries from consecutive-id changes
        present_ids = set(episode_ids)
        if not present_ids.issubset(episode_boundaries):
            change_mask = np.empty(len(episode_ids), dtype=bool)
            change_mask[0] = True
            change_mask[1:] = episode_ids[1:] != episode_ids[:-1]
            starts = np.flatnonzero(change_mask)
            ends = np.append(starts[1:], len(episode_ids))

            for ep_id, start, end in zip(episode_ids[starts], starts, ends):
                ep_id_int = int(ep_id)
                if ep_id_int not in episode_boundaries:
                    episode_boundaries[ep_id_int] = (int(start), int(end))

        return episode_boundaries

    def _build_frame_starts_map(self) -> None:
        """
        Pre-caches the episode frame_start for every sample.
        """
        shards = self.shards
        shard_map = self.shard_indices_map
        ep_map = self.episode_indices_map

        frame_starts = np.empty(len(shard_map), dtype=np.int32)
        for i in range(len(shard_map)):
            boundaries = shards[shard_map[i]]["episode_boundaries"]
            frame_starts[i] = boundaries[ep_map[i]][0]
        self.frame_starts_map = frame_starts

    def split(
        self, val_ratio: float = 0.1, seed: int = 42
    ) -> Tuple["SlidingWindowDataset", "SlidingWindowDataset"]:
        if len(self.shard_indices_map) == 0:
            raise ValueError("Cannot split empty dataset")

        unique_pairs = np.unique(
            np.stack([self.shard_indices_map, self.episode_indices_map], axis=1),
            axis=0,
        )
        num_episodes = len(unique_pairs)
        num_val_episodes = max(1, int(num_episodes * val_ratio))

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(num_episodes)
        val_pair_indices = unique_pairs[shuffled_indices[:num_val_episodes]]

        # guaranteed unique keys - might overflow if the episodes are no
        # longer tagged incrementally from 0.
        # TODO: replace with a hash if episode id convention changes
        offset = int(self.episode_indices_map.max()) + 1
        val_keys = (
            val_pair_indices[:, 0].astype(np.int64) * offset + val_pair_indices[:, 1]
        )
        sample_keys = (
            self.shard_indices_map.astype(np.int64) * offset + self.episode_indices_map
        )
        val_mask = np.isin(sample_keys, val_keys)
        train_mask = ~val_mask

        train_ds = SlidingWindowDataset.__new__(SlidingWindowDataset)
        val_ds = SlidingWindowDataset.__new__(SlidingWindowDataset)

        for ds, mask in ((train_ds, train_mask), (val_ds, val_mask)):
            ds.data_dir = self.data_dir
            ds.H = self.H
            ds.K = self.K
            ds.discretize_actions = self.discretize_actions
            ds.obs_type = self.obs_type
            ds.shards = self.shards
            ds.shard_indices_map = self.shard_indices_map[mask]
            ds.local_indices_map = self.local_indices_map[mask]
            ds.episode_indices_map = self.episode_indices_map[mask]
            ds.frame_starts_map = self.frame_starts_map[mask]

        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self.shard_indices_map)

    def _get_history_stack(
        self, obs_ds: h5py.Dataset, idx: int, frame_start: int
    ) -> np.ndarray:
        slice_start = max(idx - self.H + 1, frame_start)
        raw_slice: np.ndarray = obs_ds[slice_start : idx + 1]

        is_screen = self.obs_type == "screen"
        if is_screen and raw_slice.ndim >= 3 and raw_slice.shape[1] == 1:
            raw_slice = raw_slice[:, 0]

        pad_len = self.H - len(raw_slice)
        if pad_len > 0:
            pad_widths = [(pad_len, 0)] + [(0, 0)] * (raw_slice.ndim - 1)
            return np.pad(raw_slice, pad_widths, mode="edge")
        return raw_slice

    def _read_sample(
        self,
        obs_ds: h5py.Dataset,
        telem_ds: h5py.Dataset,
        actions_ds: h5py.Dataset,
        local_t: int,
        frame_start: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_stack_t = self._get_history_stack(obs_ds, local_t, frame_start)
        obs_stack_target = self._get_history_stack(
            obs_ds, local_t + self.K, frame_start
        )

        telem_slice = np.asarray(
            telem_ds[local_t : local_t + self.K + 1],
            dtype=np.float32,
        )
        telemetry_t = telem_slice[0]
        telemetry_target = telem_slice[-1]

        actions_seq = np.asarray(
            actions_ds[local_t : local_t + self.K], dtype=np.float32
        )
        if self.discretize_actions:
            actions_seq = discretize_action_np(actions_seq)

        return obs_stack_t, telemetry_t, actions_seq, obs_stack_target, telemetry_target

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shard_idx = self.shard_indices_map[idx]
        local_t = self.local_indices_map[idx]
        frame_start = self.frame_starts_map[idx]
        shard = self.shards[shard_idx]

        with h5py.File(shard["file_path"], "r") as f:
            obs_ds: h5py.Dataset = f[f"observations/{self.obs_type}"]  # type: ignore[assignment]
            telem_ds: h5py.Dataset = f["observations/telemetry"]  # type: ignore[assignment]
            actions_ds: h5py.Dataset = f["actions"]  # type: ignore[assignment]
            return self._read_sample(obs_ds, telem_ds, actions_ds, local_t, frame_start)


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
        # Owned RNG so shuffle order is reproducible given a seed; state
        # advances across epochs.
        self.rng = np.random.default_rng(seed)

        self.indices = np.arange(len(self.dataset))
        self.queue: queue.Queue[Optional[Dict[str, np.ndarray]]] = queue.Queue(
            maxsize=8
        )
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
            self.thread = threading.Thread(
                target=self._prefetch_loop,
                args=(indices, num_batches),
                daemon=True,
            )
            self.thread.start()

            try:
                for _ in range(num_batches):
                    batch = self.queue.get()
                    if batch is None:
                        break
                    yield batch
            finally:
                # Runs even if the consumer abandons the generator mid-epoch
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
                print(f"Warning: DataLoader worker failed to collate batch: {e}")
                break
            if not self._put_blocking(batch):
                break
        self._put_blocking(None)

    def _collate(self, batch_indices: np.ndarray) -> Dict[str, np.ndarray]:
        ds = self.dataset
        batch_size = len(batch_indices)

        shard_ids = ds.shard_indices_map[batch_indices]
        sort_order = np.argsort(shard_ids, kind="mergesort")
        sorted_indices = batch_indices[sort_order]
        sorted_shard_ids = shard_ids[sort_order]

        change_mask = np.empty(batch_size, dtype=bool)
        change_mask[0] = True
        change_mask[1:] = sorted_shard_ids[1:] != sorted_shard_ids[:-1]
        group_starts = np.flatnonzero(change_mask)
        group_ends = np.append(group_starts[1:], batch_size)

        samples: list[Tuple[np.ndarray, ...]] = [None] * batch_size  # type: ignore[list-item]

        for g_start, g_end in zip(group_starts, group_ends):
            shard_idx = int(sorted_shard_ids[g_start])
            shard = ds.shards[shard_idx]

            with h5py.File(shard["file_path"], "r") as f:
                obs_ds: h5py.Dataset = f[f"observations/{ds.obs_type}"]  # type: ignore[assignment]
                telem_ds: h5py.Dataset = f["observations/telemetry"]  # type: ignore[assignment]
                actions_ds: h5py.Dataset = f["actions"]  # type: ignore[assignment]

                for pos in range(g_start, g_end):
                    ds_idx = sorted_indices[pos]
                    local_t = int(ds.local_indices_map[ds_idx])
                    frame_start = int(ds.frame_starts_map[ds_idx])
                    samples[pos] = ds._read_sample(
                        obs_ds, telem_ds, actions_ds, local_t, frame_start
                    )

        # Unsort into original batch order before stacking to avoid a
        # stack-then-index double allocation.
        unsort_order = np.argsort(sort_order)
        ordered = [samples[i] for i in unsort_order]

        return {
            "obs_stack_t": np.stack([s[0] for s in ordered]),
            "telemetry_t": np.stack([s[1] for s in ordered]),
            "actions_seq": np.stack([s[2] for s in ordered]),
            "obs_stack_target": np.stack([s[3] for s in ordered]),
            "telemetry_target": np.stack([s[4] for s in ordered]),
        }
