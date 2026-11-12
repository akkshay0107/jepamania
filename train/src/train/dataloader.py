"""Read batches of observation and action windows from screen rollout files."""

from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Literal, NamedTuple, Self, cast

import h5py
import numpy as np
from core.actions import ACTION_GRID_NP
from jaxtyping import Int

Batch = dict[str, np.ndarray]


class _Episode(NamedTuple):
    shard: int
    episode_id: int
    start: int
    end: int
    action_end: int
    terminated: bool
    num_windows: int


class SlidingWindowDataset:
    """Index each episode once and assemble windows from contiguous HDF5 reads."""

    def __init__(
        self,
        data_dir: str | Path,
        history_len: int = 4,
        rollout_len: int = 5,
        data_files: Sequence[str | Path] | None = None,
        view: Literal["ssl", "online_ssl", "rl"] = "ssl",
    ) -> None:
        """Index complete windows from a rollout file, directory, or explicit files."""
        self.H = int(history_len)
        self.K = 1 if view == "rl" else int(rollout_len)
        if self.H <= 0 or self.K <= 0:
            raise ValueError("history_len and rollout_len must be positive.")
        self.view = view

        root = Path(data_dir)
        if data_files is not None:
            files = tuple(map(Path, data_files))
        elif root.is_file():
            files = (root,)
        else:
            files = tuple(sorted(root.rglob("*.h5")))
        shards: list[Path] = []
        episodes: list[_Episode] = []
        for path in files:
            indexed = self._index_shard(path, len(shards))
            if indexed:
                shards.append(path)
                episodes.extend(indexed)

        self.shards = tuple(shards)
        self._set_episodes(tuple(episodes))

    def _index_shard(
        self,
        path: Path,
        shard: int,
    ) -> list[_Episode]:
        with h5py.File(path, "r") as file:
            if "episode_id" not in file or "observations/screen" not in file:
                return []
            ids = np.asarray(file["episode_id"], dtype=np.int64)
            if not len(ids):
                return []
            screen = cast(h5py.Dataset, file["observations/screen"])
            telemetry = cast(h5py.Dataset, file["observations/telemetry"])
            if ids.ndim != 1 or len(screen) != len(ids) or len(telemetry) != len(ids):
                raise ValueError(
                    f"{path}: screen, telemetry, and episode IDs must align."
                )
            if "actions" not in file:
                raise ValueError(f"{path}: missing actions.")

            final_observation_only = bool(
                file.attrs.get("final_observation_only", False)
            )
            online = self.view in ("online_ssl", "rl")
            if online and (not final_observation_only or "action_ids" not in file):
                raise ValueError(
                    f"{path}: online replay requires IDs and final observations."
                )
            actions = cast(h5py.Dataset, file["actions"])
            if online and any(
                len(cast(h5py.Dataset, file[name])) != len(ids)
                for name in ("action_ids", "rewards", "actions")
            ):
                raise ValueError(f"{path}: online replay rows do not align.")
            metadata = cast(h5py.Group | None, file.get("metadata"))

            # Row IDs identify the runs, even in older files without metadata.
            boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
            if len(np.unique(ids[boundaries[:-1]])) != len(boundaries) - 1:
                raise ValueError(f"{path}: an episode ID occurs in multiple runs.")
            episodes = []
            for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
                episode_id = int(ids[start])
                action_end = min(end - int(final_observation_only), len(actions))
                num_windows = min(action_end, end - 1) - start - self.K + 1
                if num_windows <= 0:
                    continue

                episode = cast(
                    h5py.Group | None,
                    None if metadata is None else metadata.get(f"episode_{episode_id}"),
                )
                reason = None
                if episode is not None:
                    if (
                        int(episode.attrs.get("frame_start", start)) != start
                        or int(episode.attrs.get("frame_end", end)) != end
                    ):
                        raise ValueError(
                            f"{path}: metadata bounds for episode {episode_id} "
                            "disagree with its row IDs."
                        )
                    reason = episode.attrs.get("termination")
                    if isinstance(reason, bytes):
                        reason = reason.decode("utf-8")
                if online and (episode is None or reason is None):
                    continue
                terminated = (
                    online and episode is not None and bool(episode.attrs["done"])
                )

                episodes.append(
                    _Episode(
                        shard,
                        episode_id,
                        start,
                        end,
                        action_end,
                        terminated,
                        num_windows,
                    )
                )
            return episodes

    def _set_episodes(self, episodes: tuple[_Episode, ...]) -> None:
        self._episodes = episodes
        self._offsets = np.cumsum(
            [0, *(ep.num_windows for ep in episodes)], dtype=np.int64
        )

    def split(self, val_ratio: float = 0.1, seed: int = 42) -> tuple[Self, Self]:
        """Split by (file, episode ID), keeping overlapping windows together."""
        keys = sorted({(ep.shard, ep.episode_id) for ep in self._episodes})
        if not keys:
            raise ValueError("Cannot split empty dataset")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(keys))
        val_count = max(1, int(len(keys) * val_ratio))
        val_keys = {keys[i] for i in order[:val_count]}

        train, validation = copy(self), copy(self)
        train._set_episodes(
            tuple(
                ep for ep in self._episodes if (ep.shard, ep.episode_id) not in val_keys
            )
        )
        validation._set_episodes(
            tuple(ep for ep in self._episodes if (ep.shard, ep.episode_id) in val_keys)
        )
        return train, validation

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def episode_keys(self) -> list[tuple[str, int]]:
        return [(str(self.shards[ep.shard]), ep.episode_id) for ep in self._episodes]

    def include_online_shard(self, path: Path) -> None:
        """Index a newly published immutable shard once."""
        if self.view not in ("online_ssl", "rl"):
            raise ValueError("Only online views can receive new shards.")
        episodes = self._index_shard(path, len(self.shards))
        if episodes:
            self.shards += (path,)
            counts = np.cumsum([ep.num_windows for ep in episodes], dtype=np.int64)
            self._episodes += tuple(episodes)
            self._offsets = np.concatenate((self._offsets, self._offsets[-1] + counts))

    def __getitem__(self, idx: int) -> Batch:
        return {
            name: values[0] for name, values in self.read_batch(np.array([idx])).items()
        }

    def read_batch(self, indices: Int[np.ndarray, "batch"]) -> Batch:
        """Read nearby rows together and restore the caller's sample order."""
        indices = np.asarray(indices, dtype=np.intp)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("Expected a nonempty vector of sample indices.")
        indices = np.where(indices < 0, indices + len(self), indices)
        if np.any((indices < 0) | (indices >= len(self))):
            raise IndexError("Sample index out of range.")

        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        episode_indices = np.searchsorted(self._offsets[1:], indices, side="right")
        episodes = [self._episodes[i] for i in episode_indices]
        shards = np.array([ep.shard for ep in episodes])
        starts = np.array([ep.start for ep in episodes])
        ends = np.array([ep.end for ep in episodes])
        times = starts + indices - self._offsets[episode_indices]

        batch: Batch = {}
        file_breaks = np.flatnonzero(shards[1:] != shards[:-1]) + 1
        for group in np.split(np.arange(len(indices)), file_breaks):
            path = self.shards[shards[group[0]]]
            with h5py.File(path, "r") as file:
                screen = cast(h5py.Dataset, file["observations/screen"])
                telemetry = cast(h5py.Dataset, file["observations/telemetry"])
                actions = cast(h5py.Dataset, file["actions"])
                rewards = (
                    cast(h5py.Dataset | None, file.get("rewards"))
                    if self.view != "ssl"
                    else None
                )
                action_ids = (
                    cast(h5py.Dataset, file["action_ids"])
                    if self.view in ("online_ssl", "rl")
                    else None
                )

                # A sparse batch can touch distant parts of a shard. Read only
                # the spans where its windows overlap or touch.
                read_starts = np.maximum(times[group] - self.H + 1, starts[group])
                read_ends = np.minimum(times[group] + self.K + 1, ends[group])
                gaps = np.flatnonzero(read_starts[1:] > read_ends[:-1]) + 1
                for part in np.split(group, gaps):
                    first, last = part[0], part[-1]
                    span = slice(
                        int(max(times[first] - self.H + 1, starts[first])),
                        int(min(times[last] + self.K + 1, ends[last])),
                    )
                    samples = self._read_span(
                        screen,
                        telemetry,
                        actions,
                        rewards,
                        action_ids,
                        span,
                        times[part],
                        starts[part],
                        ends[part],
                        [episodes[i] for i in part],
                    )
                    if not batch:
                        batch = {
                            name: np.empty(
                                (len(indices), *values.shape[1:]), dtype=values.dtype
                            )
                            for name, values in samples.items()
                        }
                    for name, values in samples.items():
                        batch[name][order[part]] = values

        return batch

    def _read_span(
        self,
        screen: h5py.Dataset,
        telemetry: h5py.Dataset,
        actions: h5py.Dataset,
        rewards: h5py.Dataset | None,
        action_ids: h5py.Dataset | None,
        span: slice,
        times: Int[np.ndarray, "batch"],
        starts: Int[np.ndarray, "batch"],
        ends: Int[np.ndarray, "batch"],
        episodes: list[_Episode],
    ) -> Batch:
        """Build history, targets, controls, and masks for one contiguous read."""
        frames = np.asarray(screen[span])
        if frames.ndim == 4 and frames.shape[1] == 1:
            frames = frames[:, 0]

        obs_rows = times[:, None] + np.arange(1 - self.H, self.K + 1)
        obs_rows = np.clip(obs_rows, starts[:, None], ends[:, None] - 1) - span.start
        histories = np.lib.stride_tricks.sliding_window_view(
            frames[obs_rows], self.H, axis=1
        )
        histories = np.moveaxis(histories, -1, 2)
        telemetry_values = np.asarray(telemetry[span], dtype=np.float32)
        telemetry_stack = telemetry_values[obs_rows[:, self.H - 1 :]]

        steps = times[:, None] + np.arange(self.K)
        batch = {
            "obs_stack_t": histories[:, 0],
            "telemetry_t": telemetry_stack[:, 0],
            "obs_stack_targets": histories[:, 1:],
            "telemetry_targets": telemetry_stack[:, 1:],
        }
        if self.view == "ssl":
            batch["actions_seq"] = np.asarray(actions[span], dtype=np.float32)[
                steps - span.start
            ]
            return batch

        if action_ids is None or rewards is None:
            raise ValueError("Online replay is missing action IDs or rewards.")
        ids = np.asarray(action_ids[span], dtype=np.int32)[steps - span.start]
        if np.any((ids < 0) | (ids >= len(ACTION_GRID_NP))):
            raise ValueError("Online replay contains an invalid action ID.")
        reward_values = np.asarray(rewards[span], dtype=np.float32)[steps - span.start]
        action_ends = np.array([ep.action_end for ep in episodes])[:, None]
        terminated = np.array([ep.terminated for ep in episodes])[:, None]
        done = terminated & (steps == action_ends - 1)
        if self.view == "rl":
            batch.update(
                action_ids=ids[:, 0], rewards=reward_values[:, 0], dones=done[:, 0]
            )
        else:
            batch.update(
                actions_seq=ACTION_GRID_NP[ids],
                action_ids=ids,
                rewards=reward_values,
                dones=done,
            )
        return batch
