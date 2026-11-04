"""Read completed episode metadata from rollout files."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py


def _attr_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: int
    iteration: int
    transition_count: int
    termination: str = ""
    source: str = ""


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
        frame_count = len(cast(h5py.Dataset, file["rewards"]))
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
            summaries.append(
                EpisodeSummary(
                    episode_id=int(episode_id_text),
                    iteration=int(attrs.get("iteration", -1)),  # pyright: ignore[reportArgumentType]
                    transition_count=frame_end - frame_start,
                    termination=_attr_text(attrs["termination"]),
                    source=_attr_text(attrs.get("source", "")),
                )
            )
    return sorted(summaries, key=lambda summary: summary.episode_id)
