"""Drive Trackmania with a frozen Q or MPC model snapshot."""

import argparse
import json
from pathlib import Path

from win_client.env_patches import apply_online_rl_patches
from win_client.game_session import DrivingEnvironment
from win_client.mpc_driver import drive_episode, load_driver
from win_client.utils import get_tmrl_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--controller", choices=("q", "mpc"), default="mpc")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    driver, models = load_driver(args.checkpoint_path, args.seed)
    driver.warmup(models)
    apply_online_rl_patches()
    env = DrivingEnvironment(get_tmrl_env())
    try:
        for _ in range(args.episodes):
            driver.reset()
            episode = drive_episode(
                env,
                lambda obs: driver.select(models, obs, args.controller),
                keep_observations=False,
            )
            print(json.dumps(episode.summary(driver.planner.gamma)), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
