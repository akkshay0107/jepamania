"""Publish complete game episodes in the existing HDF5 recording layout."""

import os
from pathlib import Path

from core.actions import ACTION_GRID_NP, NEUTRAL_ACTION
from win_client.data_writer import HDF5Writer
from win_client.game_session import REWARD_DEFINITION
from win_client.mpc_driver import Episode
from win_client.settings import cfg


def publish_episode(path: Path, episode: Episode, metadata: dict[str, object]) -> Path:
    """Atomically expose a shard to readers after writing its real final observation."""
    if not episode.actions or len(episode.observations) != len(episode.actions) + 1:
        raise ValueError("An online episode requires at least one complete transition.")
    temp = path.with_suffix(".part")
    if path.exists():
        raise FileExistsError(path)
    writer = HDF5Writer(temp, online_reward=REWARD_DEFINITION)
    try:
        try:
            writer.new_episode(
                {
                    "source": "online",
                    "map_name": cfg.agent.map_name,
                    "map_uid": cfg.agent.map_uid,
                    **metadata,
                }
            )
            for obs, action, reward in zip(
                episode.observations[:-1], episode.actions, episode.rewards, strict=True
            ):
                writer.append(
                    {"screen": obs["screen"][-1:], "telemetry": obs["telemetry"]},
                    ACTION_GRID_NP[action],
                    reward,
                    action_id=action,
                )
            final = episode.observations[-1]
            writer.append(
                {"screen": final["screen"][-1:], "telemetry": final["telemetry"]},
                ACTION_GRID_NP[NEUTRAL_ACTION],
            )
            writer.end_episode(episode.reason, done=episode.done)
        finally:
            writer.close()
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path
