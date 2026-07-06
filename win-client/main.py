"""Entry point for Trackmania data collection on Windows."""

import argparse
import logging

from src.agent_recorder import AgentCollector
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.mode == "human":
        logging.info("Initializing Human Play Recorder...")
        recorder = HumanRecorder()
        recorder.run()
    elif args.mode == "agent":
        logging.info("Initializing Agent Policy Collector...")
        collector = AgentCollector()
        collector.run()


if __name__ == "__main__":
    main()
