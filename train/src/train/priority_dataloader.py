"""Small, deterministic episode replay selection for online fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import h5py
import numpy as np


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: int
    episode_return: float
    iteration: int
    transition_count: int


@dataclass(frozen=True)
class HistoricalPriority:
    episode_id: int
    normalized_return: float
    age: int
    priority: float


@dataclass(frozen=True)
class ReplaySelection:
    episode_ids: tuple[int, ...]
    historical: tuple[HistoricalPriority, ...]


def read_completed_episode_summaries(
    rollout_file: str | Path,
) -> list[EpisodeSummary]:
    """Read only episodes whose completion metadata has valid frame bounds."""
    path = Path(rollout_file)
    if not path.exists():
        return []

    summaries: list[EpisodeSummary] = []
    with h5py.File(path, "r") as file:
        if "metadata" not in file or "rewards" not in file:
            return summaries
        frame_count = len(file["rewards"])  # pyright: ignore[reportArgumentType]
        metadata = cast(h5py.Group, file["metadata"])
        for name in metadata.keys():
            if not name.startswith("episode_"):
                continue
            group = cast(h5py.Group, metadata[name])
            attrs = group.attrs
            if not {"frame_start", "frame_end", "termination"}.issubset(attrs):
                continue
            episode_id_text = name.removeprefix("episode_")
            if not episode_id_text.isdigit():
                continue
            frame_start = int(attrs["frame_start"])  # pyright: ignore[reportArgumentType]
            frame_end = int(attrs["frame_end"])  # pyright: ignore[reportArgumentType]
            if frame_start < 0 or frame_end <= frame_start or frame_end > frame_count:
                continue
            rewards = np.asarray(
                file["rewards"][frame_start:frame_end],  # pyright: ignore[reportIndexIssue]
                dtype=np.float32,
            )
            summaries.append(
                EpisodeSummary(
                    episode_id=int(episode_id_text),
                    episode_return=float(np.sum(rewards)),
                    iteration=int(attrs.get("iteration", -1)),  # pyright: ignore[reportArgumentType]
                    transition_count=frame_end - frame_start,
                )
            )
    return sorted(summaries, key=lambda summary: summary.episode_id)


def select_replay_episodes(
    summaries: Sequence[EpisodeSummary],
    new_episode_ids: Sequence[int],
    current_iteration: int,
    historical_limit: int = 32,
    recency_decay: float = 0.95,
    seed: int = 42,
) -> ReplaySelection:
    """Keep new episodes and sample capped history with linear priorities."""
    new_ids = tuple(dict.fromkeys(int(episode_id) for episode_id in new_episode_ids))
    new_id_set = set(new_ids)
    history = [item for item in summaries if item.episode_id not in new_id_set]
    if not history or historical_limit <= 0:
        return ReplaySelection(new_ids, ())

    returns = np.asarray([item.episode_return for item in history], dtype=np.float64)
    return_range = float(np.max(returns) - np.min(returns))
    normalized = (
        (returns - np.min(returns)) / return_range
        if return_range > 0.0
        else np.zeros_like(returns)
    )
    ages = np.asarray(
        [max(0, current_iteration - item.iteration) for item in history],
        dtype=np.int32,
    )
    priorities = (1.0 + 2.0 * normalized) * (recency_decay**ages)
    candidates = tuple(
        HistoricalPriority(
            episode_id=item.episode_id,
            normalized_return=float(norm_return),
            age=int(age),
            priority=float(priority),
        )
        for item, norm_return, age, priority in zip(
            history, normalized, ages, priorities
        )
    )

    sample_count = min(len(candidates), int(historical_limit))
    if sample_count == len(candidates):
        selected = candidates
    else:
        probabilities = priorities / np.sum(priorities)
        indices = np.random.default_rng(seed).choice(
            len(candidates), size=sample_count, replace=False, p=probabilities
        )
        selected = tuple(candidates[int(index)] for index in indices)
    selected_ids = tuple(item.episode_id for item in selected)
    return ReplaySelection(new_ids + selected_ids, selected)
