#!/usr/bin/env python3
"""
Simple Project-Root Bundle Export & Extract CLI for JEPAmania Inference Assets.

Packages required configs and inference checkpoints into a tar.gz bundle relative
to the project root, or extracts a bundle directly under the project root.
No manifest is created.
"""

from __future__ import annotations

import argparse
import logging
import tarfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIGS = [
    Path("core/config.yaml"),
    Path("win-client/settings.yaml"),
]

DEFAULT_CHECKPOINTS = [
    Path("checkpoints/finetune/ft_encoder_latest.eqx"),
    Path("checkpoints/finetune/ft_predictor_latest.eqx"),
    Path("checkpoints/finetune/ft_value_head_latest.eqx"),
]

FALLBACK_CHECKPOINTS = [
    Path("checkpoints/pretrain/pretrain_encoder_latest.eqx"),
    Path("checkpoints/pretrain/pretrain_predictor_latest.eqx"),
]


def export_bundle(
    bundle_path: Path, extra_checkpoints: list[Path] | None = None
) -> None:
    """Exports required inference configs and weights into a relative .tar.gz bundle."""
    files_to_pack: list[Path] = []

    for cfg in DEFAULT_CONFIGS:
        full_path = ROOT_DIR / cfg
        if not full_path.exists():
            logging.warning(f"Config file not found: {cfg}")
        else:
            files_to_pack.append(full_path)

    if extra_checkpoints:
        ckpts = [ROOT_DIR / c if not c.is_absolute() else c for c in extra_checkpoints]
    else:
        ft_exist = all((ROOT_DIR / c).exists() for c in DEFAULT_CHECKPOINTS)
        if ft_exist:
            ckpts = [ROOT_DIR / c for c in DEFAULT_CHECKPOINTS]
        else:
            logging.info(
                "Fine-tuned checkpoints not found; checking pretrain checkpoints."
            )
            ckpts = [ROOT_DIR / c for c in FALLBACK_CHECKPOINTS]

    for c in ckpts:
        if not c.exists():
            logging.warning(f"Checkpoint not found, skipping: {c}")
        else:
            files_to_pack.append(c)

    if not files_to_pack:
        raise RuntimeError("No files found to bundle!")

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "w:gz") as tar:
        for f in files_to_pack:
            resolved_f = f.resolve()
            try:
                rel_path = resolved_f.relative_to(ROOT_DIR)
            except ValueError:
                raise ValueError(f"Cannot export file outside project root: {f}")
            logging.info(f"Adding {rel_path} to bundle...")
            tar.add(f, arcname=rel_path.as_posix())

    logging.info(f"Successfully created bundle: {bundle_path}")


def extract_bundle(bundle_path: Path) -> None:
    """Extracts files from .tar.gz bundle directly under project root."""
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    with tarfile.open(bundle_path, "r:gz") as tar:
        for member in tar.getmembers():
            dest_path = (ROOT_DIR / member.name).resolve()
            if not dest_path.is_relative_to(ROOT_DIR):
                raise ValueError(f"Unsafe path detected escaping root: {member.name}")
            logging.info(f"Extracting {member.name} -> {dest_path}")
        try:
            tar.extractall(ROOT_DIR, filter="data")
        except TypeError:
            tar.extractall(ROOT_DIR)

    logging.info(f"Successfully extracted bundle {bundle_path} into {ROOT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle or extract JEPAmania inference assets at project root."
    )
    parser.add_argument(
        "--mode",
        choices=["export", "extract"],
        required=True,
        help="'export' to pack bundle, 'extract' to unpack bundle.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT_DIR / "jepamania_inference_bundle.tar.gz",
        help="Path to .tar.gz archive file.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit list of checkpoint files to include during export.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    bundle_path = (
        args.bundle if args.bundle.is_absolute() else (ROOT_DIR / args.bundle).resolve()
    )

    if args.mode == "export":
        export_bundle(bundle_path, args.checkpoints)
    elif args.mode == "extract":
        extract_bundle(bundle_path)


if __name__ == "__main__":
    main()
