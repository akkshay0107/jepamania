"""
Entry point for real-time Sub-JEPA MPC speedrunning on Windows.

Deploys the pretrained JAX/Equinox world model to drive Trackmania
autonomously using asynchronous trajectory optimization.
"""

import argparse
import logging

from src.mpc_driver import MPCDriver
from src.settings import cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trackmania Windows Sub-JEPA Real-Time MPC Driver"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to pretrained Sub-JEPA .eqx checkpoint.",
    )
    parser.add_argument(
        "--value-head-path",
        type=str,
        default=None,
        help="Optional path to pretrained MLPValueHead checkpoint.",
    )
    parser.add_argument(
        "--encoder-type",
        choices=["vit", "conv", "lidar"],
        default=None,
        help="Encoder backbone ('vit', 'conv', or 'lidar').",
    )
    parser.add_argument(
        "--planner-type",
        choices=["cem", "beam", "random"],
        default=None,
        help="Planner algorithm ('cem', 'beam', or 'random').",
    )
    parser.add_argument(
        "--record-rollouts",
        action="store_true",
        help="Record live MPC rollouts to HDF5 shards.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Initializing Sub-JEPA Real-Time MPC Driver...")
    if args.record_rollouts:
        cfg.mpc.record_rollouts = True

    driver = MPCDriver(
        checkpoint_path=args.checkpoint_path,
        value_head_path=args.value_head_path,
        encoder_type=args.encoder_type,
        planner_type=args.planner_type,
    )
    driver.run()


if __name__ == "__main__":
    main()
