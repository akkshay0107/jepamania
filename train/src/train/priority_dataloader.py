"""Priority-weighted sliding-window dataset and dataloader for online MBRL."""

import logging
from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy as np

from train.dataloader import DataLoader, SlidingWindowDataset


class PrioritySlidingWindowDataset(SlidingWindowDataset):
    """Indexes transition windows across HDF5 shards, selecting a priority subset
    of episodes according to Option 2 (Linear Normalized Boosting + Recency).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        history_len: int = 4,
        rollout_len: int = 5,
        discretize_actions: bool = True,
        obs_type: str = "screen",
        load_rewards: bool = False,
        max_cache_bytes: int = 4 * 1024**3,
        max_sampled_episodes: int = 32,
        beta: float = 2.0,
        alpha: float = 1.0,
        recency_decay: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.max_sampled_episodes = int(max_sampled_episodes)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.recency_decay = float(recency_decay)
        self.seed = int(seed)

        # Call super().__init__, which sets up attributes and invokes
        # self._build_index()
        super().__init__(
            data_dir=data_dir,
            history_len=history_len,
            rollout_len=rollout_len,
            discretize_actions=discretize_actions,
            obs_type=obs_type,
            load_rewards=load_rewards,
            max_cache_bytes=max_cache_bytes,
        )

    def _build_index(self) -> None:
        if not self.data_dir.exists():
            return

        h5_files = sorted(self.data_dir.rglob("*.h5"), key=lambda p: p.stat().st_mtime)
        total_shards = len(h5_files)
        if total_shards == 0:
            return

        raw_shard_indices: list[np.ndarray] = []
        raw_local_indices: list[np.ndarray] = []
        raw_episode_indices: list[np.ndarray] = []

        # Catalog: list of (shard_idx, ep_id, ep_return, shard_age)
        catalog: list[Tuple[int, int, float, int]] = []
        seen_episodes: set[Tuple[int, int]] = set()

        for shard_idx, file_path in enumerate(h5_files):
            shard_age = (total_shards - 1) - shard_idx
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
                        "ram_bytes": self._estimate_shard_bytes(f),
                    }
                    self.shards.append(shard_data)

                    raw_shard_indices.append(
                        np.full(len(valid_local_ts), shard_idx, dtype=np.int32)
                    )
                    raw_local_indices.append(valid_local_ts.astype(np.int32))
                    raw_episode_indices.append(episode_ids[valid_local_ts])

                    rewards_arr = (
                        np.asarray(f["rewards"], dtype=np.float32)
                        if "rewards" in f
                        else np.zeros(total_frames, dtype=np.float32)
                    )
                    unique_eps = np.unique(episode_ids[valid_local_ts])
                    for ep_id in unique_eps:
                        ep_id_int = int(ep_id)
                        pair = (shard_idx, ep_id_int)
                        if pair not in seen_episodes:
                            seen_episodes.add(pair)
                            ep_mask = episode_ids == ep_id_int
                            ep_return = float(np.sum(rewards_arr[ep_mask]))
                            catalog.append((shard_idx, ep_id_int, ep_return, shard_age))
            except (OSError, KeyError) as e:
                logging.warning(f"Failed to read {file_path}, skipping shard: {e}")

        if not catalog:
            return

        # Compute Option 2 Importance Weights across cataloged episodes
        returns = np.array([item[2] for item in catalog], dtype=np.float32)
        ages = np.array([item[3] for item in catalog], dtype=np.float32)

        r_min = float(np.min(returns))
        r_max = float(np.max(returns))
        if r_max > r_min:
            norm_returns = (returns - r_min) / (r_max - r_min + 1e-6)
        else:
            norm_returns = np.zeros_like(returns)

        weights = (1.0 + self.beta * (norm_returns**self.alpha)) * (
            self.recency_decay**ages
        )
        probs = weights / np.sum(weights)

        # Subset episodes if available catalog exceeds max_sampled_episodes
        num_to_sample = min(len(catalog), self.max_sampled_episodes)
        rng = np.random.default_rng(self.seed)
        if len(catalog) > self.max_sampled_episodes:
            sampled_indices = rng.choice(
                len(catalog), size=num_to_sample, replace=False, p=probs
            )
            selected_pairs = {(catalog[i][0], catalog[i][1]) for i in sampled_indices}
            logging.info(
                f"PriorityDataset: Selected {num_to_sample}/{len(catalog)} "
                f"episodes across {total_shards} shards using Option 2 weights "
                f"(beta={self.beta}, recency_decay={self.recency_decay})."
            )
        else:
            selected_pairs = {(item[0], item[1]) for item in catalog}
            logging.info(
                f"PriorityDataset: Retaining all {len(catalog)} cataloged "
                f"episodes across {total_shards} shards."
            )

        # Filter the transition arrays to only include the selected priority episodes
        filtered_shard_indices: list[np.ndarray] = []
        filtered_local_indices: list[np.ndarray] = []
        filtered_episode_indices: list[np.ndarray] = []

        for s_arr, l_arr, e_arr in zip(
            raw_shard_indices, raw_local_indices, raw_episode_indices
        ):
            if len(s_arr) == 0:
                continue
            shard_id = int(s_arr[0])
            valid_shard_eps = np.array(
                [ep for (s, ep) in selected_pairs if s == shard_id], dtype=np.int32
            )
            if len(valid_shard_eps) == 0:
                continue
            mask = np.isin(e_arr, valid_shard_eps)
            if np.any(mask):
                filtered_shard_indices.append(s_arr[mask])
                filtered_local_indices.append(l_arr[mask])
                filtered_episode_indices.append(e_arr[mask])

        if filtered_shard_indices:
            self.shard_indices_map = np.concatenate(filtered_shard_indices)
            self.local_indices_map = np.concatenate(filtered_local_indices)
            self.episode_indices_map = np.concatenate(filtered_episode_indices)
            self._build_frame_starts_map()


# Direct alias to DataLoader as PrioritySlidingWindowDataset is fully compatible with it
PriorityDataLoader = DataLoader
