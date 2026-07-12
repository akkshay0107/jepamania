"""
Entry point for Trackmania data collection on Windows.

Supports recording human gameplay via keyboard/gamepad or agent rollouts
via a pretrained SAC policy.
"""

import argparse
import logging
from pathlib import Path

from src.agent_recorder import AgentCollector
from src.env_patches import apply_data_collection_patches, apply_online_rl_patches
from src.human_recorder import HumanRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trackmania Windows Data Collection Engine"
    )
    parser.add_argument(
        "--mode",
        choices=["human", "agent"],
        default="human",
        help=(
            "Recording mode: 'human' for keyboard/gamepad play, "
            "'agent' for trained RL policy rollout."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Record bootstrap episodes into data/rl/bootstrap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    client_root = Path(__file__).resolve().parent
    if args.bootstrap:
        apply_online_rl_patches()
        output_dir = client_root / "data" / "rl" / "bootstrap"
        logging.info(f"Bootstrap recording enabled → writing to {output_dir}")
    else:
        apply_data_collection_patches()
        output_dir = client_root / "data" / "ssl" / args.mode
        logging.info(f"SSL recording enabled → writing to {output_dir}")

    if args.mode == "human":
        logging.info("Initializing Human Play Recorder...")
        recorder = HumanRecorder(output_dir=output_dir)
        recorder.run()
    elif args.mode == "agent":
        logging.info("Initializing Agent Policy Collector...")
        collector = AgentCollector(output_dir=output_dir)
        collector.run()


if __name__ == "__main__":
    main()
