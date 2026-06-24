import logging
import random
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass(slots=True, frozen=True)
class MapConfig:
    uid: str
    name: str
    terrain: str
    policy: str | None


class TerrainScheduler:
    """
    Handles loading custom list of maps and shuffling maps to collect data from
    to have a diverse set of terrains over which the data is collected
    """

    def __init__(self, yaml_path: str | Path) -> None:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Map config not found at {yaml_path}")

        data = OmegaConf.load(yaml_path)
        terrain_defaults = {
            k: v for k, v in getattr(data, "terrain_defaults", {}).items() if v
        }

        self.maps: list[MapConfig] = []
        for entry in data.maps:
            policy = entry.get("policy") or terrain_defaults.get(entry.terrain)
            self.maps.append(
                MapConfig(
                    uid=str(entry.uid),
                    name=str(entry.name),
                    terrain=str(entry.terrain),
                    policy=str(policy) if policy else None,
                )
            )

        if not self.maps:
            raise ValueError("No maps found in terrain_maps.yaml")

        self._cycle: list[MapConfig] = []
        logging.info(f"TerrainScheduler loaded {len(self.maps)} maps.")

    def __iter__(self):
        return self

    def __next__(self) -> MapConfig:
        if not self._cycle:
            self._cycle = list(self.maps)
            random.shuffle(self._cycle)
        return self._cycle.pop()
