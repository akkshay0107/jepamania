"""HDF5-backed sliding-window dataset and prefetching dataloader."""

import queue
import threading
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from core.actions import discretize_action_np, rescale_gas_np
from core.config import IMAGINED_ROLLOUT_LEN, OBSERVED_ROLLOUT_LEN


class _ShardPool:
    """Thread-safe byte-budgeted store of fully materialized shard arrays.

    Shards are pinned by refcount while a loader window consumes them.
    Entries whose refcount drops to zero stay soft-resident and are evicted
    (oldest first) only when a later acquisition needs the budget, so a
    dataset that fits in RAM stays warm across epochs. Shared by reference
    across dataset splits so a shard containing both train and validation
    episodes is only materialized once.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max_bytes)
        self.cache: Dict[int, Dict[str, Any]] = {}
        self.refcounts: Dict[int, int] = {}
        self.bytes_used: int = 0
        self.lock = threading.Lock()

    def peek(self, shard_idx: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.cache.get(shard_idx)


class SlidingWindowDataset:
    """Indexed sequences of consecutive transitions from HDF5 rollouts.

    Constructs (obs_stack_t, actions_seq, obs_stack_targets, rewards_seq) tuples
    where obs stacks have length H and rollout targets/actions have length K.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        history_len: int = 4,
        rollout_len: int = 5,
        discretize_actions: bool = True,
        obs_type: str = "screen",
        load_rewards: bool = False,
        max_cache_bytes: int = 4 * 1024**3,  # 4 GB RAM shard pool
        episode_ids: Optional[Sequence[int]] = None,
    ) -> None:
        if obs_type not in ("screen", "lidar"):
            raise ValueError("obs_type must be either 'screen' or 'lidar'.")
        self.obs_type = obs_type
        self.data_dir = Path(data_dir)
        self.H = int(history_len)
        self.K_obs = int(rollout_len) if rollout_len != 5 else OBSERVED_ROLLOUT_LEN
        self.K_imag = max(self.K_obs, IMAGINED_ROLLOUT_LEN)
        self.K = self.K_obs
        self.discretize_actions = bool(discretize_actions)
        self.load_rewards = bool(load_rewards)
        self.max_cache_bytes = int(max_cache_bytes)
        self.episode_ids = (
            frozenset(int(episode_id) for episode_id in episode_ids)
            if episode_ids is not None
            else None
        )

        self._pool = _ShardPool(self.max_cache_bytes)

        self.shards: list[Dict[str, Any]] = []
        self.shard_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.local_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.episode_indices_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.frame_starts_map: np.ndarray = np.empty(0, dtype=np.int32)
        self.frame_ends_map: np.ndarray = np.empty(0, dtype=np.int32)

        self._build_index()

    def _build_index(self) -> None:
        if not self.data_dir.exists():
            return

        h5_files = (
            [self.data_dir]
            if self.data_dir.is_file()
            else sorted(self.data_dir.rglob("*.h5"))
        )
        shard_indices: list[np.ndarray] = []
        local_indices: list[np.ndarray] = []
        episode_indices: list[np.ndarray] = []

        for file_path in h5_files:
            try:
                with h5py.File(file_path, "r") as f:
                    obs_key = f"observations/{self.obs_type}"
                    if "episode_id" not in f or obs_key not in f:
                        continue

                    episode_ids: np.ndarray = np.asarray(
                        f["episode_id"], dtype=np.int32
                    )
                    total_frames = len(episode_ids)
                    if total_frames <= self.K_obs:
                        continue

                    indices = np.arange(total_frames - self.K_obs)
                    valid_mask = (
                        episode_ids[indices] == episode_ids[indices + self.K_obs]
                    )
                    if self.episode_ids is not None:
                        valid_mask &= np.isin(
                            episode_ids[indices], tuple(self.episode_ids)
                        )
                    valid_local_ts = indices[valid_mask]

                    if len(valid_local_ts) == 0:
                        continue

                    episode_boundaries = self._read_episode_boundaries(f, episode_ids)

                    shard_idx = len(self.shards)
                    shard_data = {
                        "file_path": file_path,
                        "episode_boundaries": episode_boundaries,
                        "ram_bytes": self._estimate_shard_bytes(f),
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
        frame_ends = np.empty(len(shard_map), dtype=np.int32)
        for i in range(len(shard_map)):
            boundaries = shards[shard_map[i]]["episode_boundaries"]
            frame_starts[i] = boundaries[ep_map[i]][0]
            frame_ends[i] = boundaries[ep_map[i]][1]
        self.frame_starts_map = frame_starts
        self.frame_ends_map = frame_ends

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
            ds.K_obs = self.K_obs
            ds.K_imag = self.K_imag
            ds.K = self.K
            ds.discretize_actions = self.discretize_actions
            ds.obs_type = self.obs_type
            ds.load_rewards = self.load_rewards
            ds.max_cache_bytes = self.max_cache_bytes
            ds.episode_ids = self.episode_ids
            ds._pool = self._pool
            ds.shards = self.shards
            ds.shard_indices_map = self.shard_indices_map[mask]
            ds.local_indices_map = self.local_indices_map[mask]
            ds.episode_indices_map = self.episode_indices_map[mask]
            ds.frame_starts_map = self.frame_starts_map[mask]
            ds.frame_ends_map = self.frame_ends_map[mask]

        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self.shard_indices_map)

    def _read_sample(
        self,
        f: h5py.File,
        local_t: int,
        frame_start: int,
        frame_end: int,
    ) -> Dict[str, np.ndarray]:
        obs_ds = f[f"observations/{self.obs_type}"]
        telem_ds = f["observations/telemetry"]
        actions_ds = f["actions"]

        slice_start = max(local_t - self.H + 1, frame_start)
        raw_chunk: np.ndarray = obs_ds[  # pyright: ignore[reportIndexIssue, reportAssignmentType]
            slice_start : local_t + self.K_obs + 1
        ]

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
            [padded_chunk[k : k + self.H] for k in range(1, self.K_obs + 1)]
        )

        telem_slice = np.asarray(
            telem_ds[local_t : local_t + self.K_obs + 1],  # pyright: ignore[reportIndexIssue]
            dtype=np.float32,
        )
        telemetry_t = telem_slice[0]
        telemetry_targets = telem_slice[1:]

        # Slice actions and rewards up to K_imag without crossing episode boundaries
        available_steps = min(self.K_imag, max(0, frame_end - local_t))
        if "actions" in f:
            available_steps = min(
                available_steps,
                max(0, len(actions_ds) - local_t),  # pyright: ignore[reportArgumentType]
            )

        raw_actions = np.asarray(
            actions_ds[local_t : local_t + available_steps],  # pyright: ignore[reportIndexIssue]
            dtype=np.float32,
        )
        raw_actions = rescale_gas_np(raw_actions)
        if self.discretize_actions:
            raw_actions = discretize_action_np(raw_actions)

        if available_steps < self.K_imag:
            actions_seq = np.zeros(
                (self.K_imag, *raw_actions.shape[1:]), dtype=raw_actions.dtype
            )
            if available_steps > 0:
                actions_seq[:available_steps] = raw_actions
        else:
            actions_seq = raw_actions

        mask_seq = np.zeros(self.K_imag, dtype=np.float32)
        if available_steps > 0:
            mask_seq[:available_steps] = 1.0

        sample = {
            "obs_stack_t": obs_stack_t,
            "telemetry_t": telemetry_t,
            "actions_seq": actions_seq,
            "obs_stack_targets": obs_stack_targets,
            "telemetry_targets": telemetry_targets,
            "mask_seq": mask_seq,
        }

        if self.load_rewards:
            if "rewards" in f:
                rewards_ds = f["rewards"]
                avail_rew = min(
                    available_steps,
                    max(0, len(rewards_ds) - local_t),  # pyright: ignore[reportArgumentType]
                )
                raw_rewards = np.asarray(
                    rewards_ds[local_t : local_t + avail_rew],  # pyright: ignore[reportIndexIssue]
                    dtype=np.float32,
                )
                rewards_seq = np.zeros(self.K_imag, dtype=np.float32)
                if avail_rew > 0:
                    rewards_seq[:avail_rew] = raw_rewards
                sample["rewards_seq"] = rewards_seq
            else:
                sample["rewards_seq"] = np.zeros(self.K_imag, dtype=np.float32)

        return sample

    def _estimate_shard_bytes(self, f: h5py.File) -> int:
        """Predicted in-RAM footprint of a shard from HDF5 metadata alone.

        Telemetry/actions/rewards are stored as float32 in the cache, so their
        footprint is element count times four bytes regardless of on-disk dtype.
        """
        obs_ds = f[f"observations/{self.obs_type}"]
        total = int(np.prod(obs_ds.shape)) * obs_ds.dtype.itemsize  # pyright: ignore[reportAttributeAccessIssue]
        for name in ("observations/telemetry", "actions"):
            total += int(np.prod(f[name].shape)) * 4  # pyright: ignore[reportAttributeAccessIssue]
        if self.load_rewards and "rewards" in f:
            total += int(np.prod(f["rewards"].shape)) * 4  # pyright: ignore[reportAttributeAccessIssue]
        return total

    def _materialize_shard(self, shard_idx: int) -> Dict[str, Any]:
        """Read a whole shard from disk into the array layout the pool serves."""
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
        return {
            "obs": raw_obs,
            "telem": telem,
            "actions": actions,
            "rewards": rewards,
            "nbytes": int(shard["ram_bytes"]),
        }

    def load_shards(self, shard_ids: Sequence[int]) -> list[int]:
        """Pin shards into the shared RAM pool, evicting idle entries if needed.

        Budget for each shard is reserved under the lock before the slow disk
        read so concurrent loaders cannot jointly overshoot it. Shards that
        still don't fit are skipped and stream from file instead.

        Returns the shard ids actually pinned; the caller must hand exactly
        that list back to `release_shards` when done with the window.
        """
        pool = self._pool
        pinned: list[int] = []
        to_load: list[int] = []
        for shard_idx in shard_ids:
            nbytes = int(self.shards[shard_idx]["ram_bytes"])
            with pool.lock:
                if shard_idx in pool.refcounts:
                    pool.refcounts[shard_idx] += 1
                    pinned.append(shard_idx)
                    continue
                if shard_idx in pool.cache:
                    pool.refcounts[shard_idx] = 1
                    pinned.append(shard_idx)
                    continue
                if pool.max_bytes <= 0:
                    continue
                idle = [i for i in pool.cache if i not in pool.refcounts]
                for victim in idle:
                    if pool.bytes_used + nbytes <= pool.max_bytes:
                        break
                    evicted = pool.cache.pop(victim)
                    pool.bytes_used -= int(evicted["nbytes"])
                if pool.bytes_used + nbytes > pool.max_bytes:
                    continue
                pool.bytes_used += nbytes
                pool.refcounts[shard_idx] = 1
            pinned.append(shard_idx)
            to_load.append(shard_idx)

        for shard_idx in to_load:
            try:
                entry = self._materialize_shard(shard_idx)
            except (OSError, KeyError) as e:
                print(f"Warning: failed to load shard {shard_idx} into RAM: {e}")
                with pool.lock:
                    del pool.refcounts[shard_idx]
                    pool.bytes_used -= int(self.shards[shard_idx]["ram_bytes"])
                pinned.remove(shard_idx)
                continue
            with pool.lock:
                pool.cache[shard_idx] = entry
        return pinned

    def release_shards(self, shard_ids: Sequence[int]) -> None:
        """Unpin shards previously returned by `load_shards`.

        Entries stay soft-resident after their last unpin; they are only
        dropped when a later `load_shards` call needs the budget.
        """
        pool = self._pool
        with pool.lock:
            for shard_idx in shard_ids:
                rc = pool.refcounts.get(shard_idx)
                if rc is None:
                    continue
                if rc <= 1:
                    del pool.refcounts[shard_idx]
                    if shard_idx not in pool.cache:
                        pool.bytes_used -= int(self.shards[shard_idx]["ram_bytes"])
                else:
                    pool.refcounts[shard_idx] = rc - 1

    def _read_sample_ram(
        self, cached: Dict[str, Any], local_t: int, frame_start: int, frame_end: int
    ) -> Dict[str, np.ndarray]:
        """Slice contiguous transition stack directly from cached RAM arrays."""
        obs_arr = cached["obs"]
        telem_arr = cached["telem"]
        actions_arr = cached["actions"]

        slice_start = max(local_t - self.H + 1, frame_start)
        raw_chunk: np.ndarray = obs_arr[slice_start : local_t + self.K_obs + 1]

        pad_len = self.H - ((local_t + 1) - slice_start)
        if pad_len > 0:
            pad_widths = [(pad_len, 0)] + [(0, 0)] * (raw_chunk.ndim - 1)
            padded_chunk = np.pad(raw_chunk, pad_widths, mode="edge")
        else:
            padded_chunk = raw_chunk

        obs_stack_t = padded_chunk[0 : self.H]
        obs_stack_targets = np.stack(
            [padded_chunk[k : k + self.H] for k in range(1, self.K_obs + 1)]
        )

        telem_slice = telem_arr[local_t : local_t + self.K_obs + 1]
        telemetry_t = telem_slice[0]
        telemetry_targets = telem_slice[1:]

        available_steps = min(self.K_imag, max(0, frame_end - local_t))
        available_steps = min(available_steps, max(0, len(actions_arr) - local_t))

        raw_actions = actions_arr[local_t : local_t + available_steps]
        raw_actions = rescale_gas_np(raw_actions)
        if self.discretize_actions:
            raw_actions = discretize_action_np(raw_actions)

        if available_steps < self.K_imag:
            actions_seq = np.zeros(
                (self.K_imag, *raw_actions.shape[1:]), dtype=raw_actions.dtype
            )
            if available_steps > 0:
                actions_seq[:available_steps] = raw_actions
        else:
            actions_seq = raw_actions

        mask_seq = np.zeros(self.K_imag, dtype=np.float32)
        if available_steps > 0:
            mask_seq[:available_steps] = 1.0

        sample = {
            "obs_stack_t": obs_stack_t,
            "telemetry_t": telemetry_t,
            "actions_seq": actions_seq,
            "obs_stack_targets": obs_stack_targets,
            "telemetry_targets": telemetry_targets,
            "mask_seq": mask_seq,
        }

        if self.load_rewards:
            if cached["rewards"] is not None:
                avail_rew = min(
                    available_steps, max(0, len(cached["rewards"]) - local_t)
                )
                rewards_seq = np.zeros(self.K_imag, dtype=np.float32)
                if avail_rew > 0:
                    rewards_seq[:avail_rew] = cached["rewards"][
                        local_t : local_t + avail_rew
                    ]
                sample["rewards_seq"] = rewards_seq
            else:
                sample["rewards_seq"] = np.zeros(self.K_imag, dtype=np.float32)

        return sample

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        shard_idx = int(self.shard_indices_map[idx])
        local_t = int(self.local_indices_map[idx])
        frame_start = int(self.frame_starts_map[idx])
        frame_end = int(self.frame_ends_map[idx])

        cached = self._pool.peek(shard_idx)
        if cached is not None:
            return self._read_sample_ram(cached, local_t, frame_start, frame_end)

        shard = self.shards[shard_idx]
        with h5py.File(shard["file_path"], "r") as f:
            return self._read_sample(f, local_t, frame_start, frame_end)


# One window's worth of an epoch: sample indices to serve, in order, and the
# shard ids that must be RAM-resident while serving them.
Window = Tuple[np.ndarray, list[int]]


class DataLoader:
    """Prefetches batches of transitions in a background thread.

    Each epoch is split into shard windows sized to half the dataset's RAM
    budget; a window is served entirely from RAM while the next one loads in
    the background, then released. With `shuffle`, shard order is permuted
    across windows and sample order within each window (not globally), and
    `drop_last` drops the remainder of each window.
    """

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

        self.queue: queue.Queue[Optional[Dict[str, np.ndarray]]] = queue.Queue(
            maxsize=8
        )
        self.thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()

    def _num_batches(self, num_samples: int) -> int:
        num_batches = num_samples // self.batch_size
        if not self.drop_last and num_samples % self.batch_size != 0:
            num_batches += 1
        return num_batches

    def _plan_windows(self) -> list[Window]:
        """Partition the epoch into shard windows fitting half the RAM budget.

        Half so the next window can prefetch while the current one is being
        consumed; the two resident windows together respect the full budget.
        A zero/negative budget yields a single window that streams from file.
        """
        ds = self.dataset
        shard_of = ds.shard_indices_map
        order = np.argsort(shard_of, kind="mergesort")
        unique_shards, starts = np.unique(shard_of[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        groups = {int(s): order[a:b] for s, a, b in zip(unique_shards, starts, ends)}

        shard_order = [int(s) for s in unique_shards]
        if self.shuffle:
            perm = self.rng.permutation(len(shard_order))
            shard_order = [shard_order[i] for i in perm]

        def finish_window(shard_ids: list[int]) -> Window:
            indices = np.concatenate([groups[sid] for sid in shard_ids])
            if self.shuffle:
                self.rng.shuffle(indices)
            else:
                indices = np.sort(indices)
            return indices, shard_ids

        window_budget = ds._pool.max_bytes // 2
        windows: list[Window] = []
        current_ids: list[int] = []
        current_bytes = 0
        for sid in shard_order:
            nbytes = int(ds.shards[sid]["ram_bytes"])
            if current_ids and 0 < window_budget < current_bytes + nbytes:
                windows.append(finish_window(current_ids))
                current_ids, current_bytes = [], 0
            current_ids.append(sid)
            current_bytes += nbytes
        if current_ids:
            windows.append(finish_window(current_ids))
        return windows

    def __iter__(self) -> Generator[Dict[str, np.ndarray], None, None]:
        if len(self.dataset) == 0:
            return

        windows = self._plan_windows()
        total_batches = sum(self._num_batches(len(idx)) for idx, _ in windows)
        if total_batches == 0:
            return

        if self.num_workers > 0:
            self.shutdown_event.clear()
            self.thread = threading.Thread(
                target=self._prefetch_loop,
                args=(windows,),
                daemon=True,
            )
            self.thread.start()

            try:
                for _ in range(total_batches):
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
            for win_indices, shard_ids in windows:
                pinned = self.dataset.load_shards(shard_ids)
                try:
                    for i in range(self._num_batches(len(win_indices))):
                        yield self._collate(
                            win_indices[i * self.batch_size : (i + 1) * self.batch_size]
                        )
                finally:
                    self.dataset.release_shards(pinned)

    def _put_blocking(self, item: Optional[Dict[str, np.ndarray]]) -> bool:
        while not self.shutdown_event.is_set():
            try:
                self.queue.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def _prefetch_loop(self, windows: list[Window]) -> None:
        from concurrent.futures import ThreadPoolExecutor

        ds = self.dataset
        workers = max(1, self.num_workers)
        prefetch_count = max(4, workers * 2)

        pinned: Dict[int, list[int]] = {}

        def load_window(w: int) -> None:
            pinned[w] = ds.load_shards(windows[w][1])

        loader: Optional[threading.Thread] = None
        load_window(0)
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for w, (win_indices, _) in enumerate(windows):
                    if self.shutdown_event.is_set():
                        break

                    loader = None
                    if w + 1 < len(windows):
                        loader = threading.Thread(
                            target=load_window, args=(w + 1,), daemon=True
                        )
                        loader.start()

                    window_ok = self._emit_window(executor, win_indices, prefetch_count)

                    if loader is not None:
                        loader.join()
                    ds.release_shards(pinned.pop(w, []))
                    if not window_ok:
                        break
        finally:
            if loader is not None and loader.is_alive():
                loader.join()
            for held in pinned.values():
                ds.release_shards(held)
            self._put_blocking(None)

    def _emit_window(self, executor, indices: np.ndarray, prefetch_count: int) -> bool:
        """Collates and enqueues one window's batches; False aborts the epoch."""
        num_batches = self._num_batches(len(indices))
        futures = []
        next_submit_idx = 0

        while next_submit_idx < min(num_batches, prefetch_count):
            batch_indices = indices[
                next_submit_idx * self.batch_size : (next_submit_idx + 1)
                * self.batch_size
            ]
            futures.append(executor.submit(self._collate, batch_indices))
            next_submit_idx += 1

        for _ in range(num_batches):
            if self.shutdown_event.is_set():
                return False
            try:
                batch = futures.pop(0).result()
            except Exception as e:
                print(f"Warning: DataLoader worker failed to collate batch: {e}")
                return False

            if not self._put_blocking(batch):
                return False

            if next_submit_idx < num_batches:
                batch_indices = indices[
                    next_submit_idx * self.batch_size : (next_submit_idx + 1)
                    * self.batch_size
                ]
                futures.append(executor.submit(self._collate, batch_indices))
                next_submit_idx += 1
        return True

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

            cached = ds._pool.peek(shard_idx)
            if cached is not None:
                for pos in range(g_start, g_end):
                    ds_idx = sorted_indices[pos]
                    local_t = int(ds.local_indices_map[ds_idx])
                    frame_start = int(ds.frame_starts_map[ds_idx])
                    frame_end = int(ds.frame_ends_map[ds_idx])
                    samples[pos] = ds._read_sample_ram(
                        cached, local_t, frame_start, frame_end
                    )
                continue

            shard = ds.shards[shard_idx]
            with h5py.File(shard["file_path"], "r") as f:
                for pos in range(g_start, g_end):
                    ds_idx = sorted_indices[pos]
                    local_t = int(ds.local_indices_map[ds_idx])
                    frame_start = int(ds.frame_starts_map[ds_idx])
                    frame_end = int(ds.frame_ends_map[ds_idx])
                    samples[pos] = ds._read_sample(f, local_t, frame_start, frame_end)

        unsort_order = np.argsort(sort_order)
        ordered = [samples[i] for i in unsort_order]

        keys = ordered[0].keys()
        return {k: np.stack([s[k] for s in ordered]) for k in keys}
