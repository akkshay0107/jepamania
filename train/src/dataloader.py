"""HDF5-backed sliding-window dataset and prefetching dataloader.

Indexes observation/telemetry/action transitions from sharded HDF5 files and
serves samples using a thread-safe bounded LRU RAM shard cache with fallback
to lazy contiguous file reads.
"""

import collections
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

import h5py
import numpy as np
from core.actions import discretize_action_np, rescale_gas_np


class SlidingWindowDataset:
    """Indexes and retrieves transitions across multiple HDF5 shards."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        history_len: int = 4,
        rollout_len: int = 5,
        discretize_actions: bool = True,
        obs_type: str = "screen",
        load_rewards: bool = False,
        max_cache_bytes: int = 4 * 1024**3,  # 4 GB LRU shard pool
    ) -> None:
        if obs_type not in ("screen", "lidar"):
            raise ValueError("obs_type must be either 'screen' or 'lidar'.")
        self.obs_type = obs_type
        self.data_dir = Path(data_dir)
        self.H = int(history_len)
        self.K = int(rollout_len)
        self.discretize_actions = bool(discretize_actions)
        self.load_rewards = bool(load_rewards)
        self.max_cache_bytes = int(max_cache_bytes)

        self._shard_cache: collections.OrderedDict[int, Dict[str, Any]] = (
            collections.OrderedDict()
        )
        self._current_cache_bytes: int = 0
        self._cache_lock = threading.Lock()

        self.shards: list[Dict[str, Any]] = []
        self.shard_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.local_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.episode_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.frame_starts_map: np.ndarray = np.empty(0, dtype=np.int32)

        self._build_index()

    def _build_index(self) -> None:
        if not self.data_dir.exists():
            return

        h5_files = sorted(self.data_dir.rglob("*.h5"))
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
            ds.load_rewards = self.load_rewards
            ds.max_cache_bytes = self.max_cache_bytes
            ds._shard_cache = collections.OrderedDict()
            ds._current_cache_bytes = 0
            ds._cache_lock = threading.Lock()
            ds.shards = self.shards
            ds.shard_indices_map = self.shard_indices_map[mask]
            ds.local_indices_map = self.local_indices_map[mask]
            ds.episode_indices_map = self.episode_indices_map[mask]
            ds.frame_starts_map = self.frame_starts_map[mask]

        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self.shard_indices_map)

    def _read_sample(
        self,
        f: h5py.File,
        local_t: int,
        frame_start: int,
    ) -> Dict[str, np.ndarray]:
        obs_ds = f[f"observations/{self.obs_type}"]
        telem_ds = f["observations/telemetry"]
        actions_ds = f["actions"]

        slice_start = max(local_t - self.H + 1, frame_start)
        raw_chunk: np.ndarray = obs_ds[slice_start : local_t + self.K + 1]

        is_screen = self.obs_type == "screen"
        if is_screen and raw_chunk.ndim >= 3 and raw_chunk.shape[1] == 1:
            raw_chunk = raw_chunk[:, 0]

        pad_len = self.H - ((local_t + 1) - slice_start)
        if pad_len > 0:
            pad_widths = [(pad_len, 0)] + [(0, 0)] * (raw_chunk.ndim - 1)
            padded_chunk = np.pad(raw_chunk, pad_widths, mode="edge")
        else:
            padded_chunk = raw_chunk

        obs_stack_t = padded_chunk[0 : self.H]
        obs_stack_targets = np.stack(
            [padded_chunk[k : k + self.H] for k in range(1, self.K + 1)]
        )

        telem_slice = np.asarray(
            telem_ds[local_t : local_t + self.K + 1],  # pyright: ignore[reportIndexIssue]
            dtype=np.float32,
        )
        telemetry_t = telem_slice[0]

        actions_seq = np.asarray(
            actions_ds[local_t : local_t + self.K],  # pyright: ignore[reportIndexIssue]
            dtype=np.float32,
        )
        actions_seq = rescale_gas_np(actions_seq)
        if self.discretize_actions:
            actions_seq = discretize_action_np(actions_seq)

        telemetry_targets = telem_slice[1:]

        sample = {
            "obs_stack_t": obs_stack_t,
            "telemetry_t": telemetry_t,
            "actions_seq": actions_seq,
            "obs_stack_targets": obs_stack_targets,
            "telemetry_targets": telemetry_targets,
        }

        if self.load_rewards:
            if "rewards" in f:
                rewards_ds = f["rewards"]
                sample["rewards_seq"] = np.asarray(
                    rewards_ds[local_t : local_t + self.K],  # pyright: ignore[reportIndexIssue]
                    dtype=np.float32,
                )
            else:
                sample["rewards_seq"] = np.zeros(self.K, dtype=np.float32)

        return sample

    def _get_shard_data(self, shard_idx: int) -> Optional[Dict[str, Any]]:
        """Retrieve shard arrays from LRU cache, evicting oldest shards when full."""
        if self.max_cache_bytes <= 0:
            return None
        with self._cache_lock:
            if shard_idx in self._shard_cache:
                self._shard_cache.move_to_end(shard_idx)
                return self._shard_cache[shard_idx]

        shard = self.shards[shard_idx]
        with h5py.File(shard["file_path"], "r") as f:
            obs_ds = f[f"observations/{self.obs_type}"]
            raw_obs = obs_ds[:]  # pyright: ignore[reportIndexIssue]
            if (
                self.obs_type == "screen"
                and raw_obs.ndim >= 3  # pyright: ignore[reportAttributeAccessIssue]
                and raw_obs.shape[1] == 1  # pyright: ignore[reportAttributeAccessIssue]
            ):
                raw_obs = raw_obs[:, 0]  # pyright: ignore[reportIndexIssue]
            telem = np.asarray(
                f["observations/telemetry"][:],  # pyright: ignore[reportIndexIssue]
                dtype=np.float32,
            )
            actions = np.asarray(
                f["actions"][:],  # pyright: ignore[reportIndexIssue]
                dtype=np.float32,
            )
            rewards = (
                np.asarray(
                    f["rewards"][:],  # pyright: ignore[reportIndexIssue]
                    dtype=np.float32,
                )
                if self.load_rewards and "rewards" in f
                else None
            )

        new_bytes = int(
            raw_obs.nbytes  # pyright: ignore[reportAttributeAccessIssue]
            + telem.nbytes
            + actions.nbytes
        )
        if rewards is not None:
            new_bytes += int(rewards.nbytes)

        with self._cache_lock:
            if shard_idx in self._shard_cache:
                self._shard_cache.move_to_end(shard_idx)
                return self._shard_cache[shard_idx]

            while (
                self._current_cache_bytes + new_bytes > self.max_cache_bytes
                and len(self._shard_cache) > 0
            ):
                _, old_data = self._shard_cache.popitem(last=False)
                old_bytes = int(
                    old_data["obs"].nbytes
                    + old_data["telem"].nbytes
                    + old_data["actions"].nbytes
                )
                if old_data["rewards"] is not None:
                    old_bytes += int(old_data["rewards"].nbytes)
                self._current_cache_bytes = max(
                    0, self._current_cache_bytes - old_bytes
                )

            self._shard_cache[shard_idx] = {
                "obs": raw_obs,
                "telem": telem,
                "actions": actions,
                "rewards": rewards,
            }
            self._current_cache_bytes += new_bytes
            return self._shard_cache[shard_idx]

    def _read_sample_ram(
        self, cached: Dict[str, Any], local_t: int, frame_start: int
    ) -> Dict[str, np.ndarray]:
        """Slice contiguous transition stack directly from cached RAM arrays."""
        obs_arr = cached["obs"]
        telem_arr = cached["telem"]
        actions_arr = cached["actions"]

        slice_start = max(local_t - self.H + 1, frame_start)
        raw_chunk: np.ndarray = obs_arr[slice_start : local_t + self.K + 1]

        pad_len = self.H - ((local_t + 1) - slice_start)
        if pad_len > 0:
            pad_widths = [(pad_len, 0)] + [(0, 0)] * (raw_chunk.ndim - 1)
            padded_chunk = np.pad(raw_chunk, pad_widths, mode="edge")
        else:
            padded_chunk = raw_chunk

        obs_stack_t = padded_chunk[0 : self.H]
        obs_stack_targets = np.stack(
            [padded_chunk[k : k + self.H] for k in range(1, self.K + 1)]
        )

        telem_slice = telem_arr[local_t : local_t + self.K + 1]
        telemetry_t = telem_slice[0]
        telemetry_targets = telem_slice[1:]

        actions_seq = actions_arr[local_t : local_t + self.K]
        actions_seq = rescale_gas_np(actions_seq)
        if self.discretize_actions:
            actions_seq = discretize_action_np(actions_seq)

        sample = {
            "obs_stack_t": obs_stack_t,
            "telemetry_t": telemetry_t,
            "actions_seq": actions_seq,
            "obs_stack_targets": obs_stack_targets,
            "telemetry_targets": telemetry_targets,
        }

        if self.load_rewards:
            if cached["rewards"] is not None:
                sample["rewards_seq"] = cached["rewards"][local_t : local_t + self.K]
            else:
                sample["rewards_seq"] = np.zeros(self.K, dtype=np.float32)

        return sample

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        shard_idx = int(self.shard_indices_map[idx])
        local_t = int(self.local_indices_map[idx])
        frame_start = int(self.frame_starts_map[idx])

        cached = self._get_shard_data(shard_idx)
        if cached is not None:
            return self._read_sample_ram(cached, local_t, frame_start)

        shard = self.shards[shard_idx]
        with h5py.File(shard["file_path"], "r") as f:
            return self._read_sample(f, local_t, frame_start)


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
        from concurrent.futures import ThreadPoolExecutor

        workers = max(1, self.num_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prefetch_count = max(4, workers * 2)
            futures = []
            next_submit_idx = 0

            while next_submit_idx < min(num_batches, prefetch_count):
                batch_indices = indices[
                    next_submit_idx * self.batch_size : (next_submit_idx + 1)
                    * self.batch_size
                ]
                futures.append(executor.submit(self._collate, batch_indices))
                next_submit_idx += 1

            for i in range(num_batches):
                if self.shutdown_event.is_set():
                    break
                try:
                    batch = futures.pop(0).result()
                except Exception as e:
                    print(f"Warning: DataLoader worker failed to collate batch: {e}")
                    break

                if not self._put_blocking(batch):
                    break

                if next_submit_idx < num_batches:
                    batch_indices = indices[
                        next_submit_idx * self.batch_size : (next_submit_idx + 1)
                        * self.batch_size
                    ]
                    futures.append(executor.submit(self._collate, batch_indices))
                    next_submit_idx += 1

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

        samples: list[Dict[str, np.ndarray]] = [None] * batch_size  # type: ignore[list-item]

        for g_start, g_end in zip(group_starts, group_ends):
            shard_idx = int(sorted_shard_ids[g_start])
            shard = ds.shards[shard_idx]

            with h5py.File(shard["file_path"], "r") as f:
                for pos in range(g_start, g_end):
                    ds_idx = sorted_indices[pos]
                    local_t = int(ds.local_indices_map[ds_idx])
                    frame_start = int(ds.frame_starts_map[ds_idx])
                    samples[pos] = ds._read_sample(f, local_t, frame_start)

        unsort_order = np.argsort(sort_order)
        ordered = [samples[i] for i in unsort_order]

        keys = ordered[0].keys()
        return {k: np.stack([s[k] for s in ordered]) for k in keys}
