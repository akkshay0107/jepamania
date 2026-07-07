"""Entry point for Phase 4 offline Sub-JEPA pretraining."""

import argparse
from pathlib import Path

import jax
from core.config import IMG_HIST_LEN, load_config
from omegaconf import OmegaConf
from src.dataloader import DataLoader, SlidingWindowDataset
from src.train import train

from core import ConvEncoder, LidarEncoder, MLPPredictor, ViTEncoder

# Anchor to the repo layout so the default works from any working directory.
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "core" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Sub-JEPA pretraining")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory of HDF5 shards"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--encoder",
        choices=["vit", "conv", "lidar"],
        default="vit",
        help="Encoder backbone",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-len", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the rolling checkpoints in --checkpoint-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config.exists():
        print(f"Loading config from {args.config}")
    else:
        print(f"Warning: config {args.config} not found; using structured defaults")
    cfg = load_config(str(args.config))

    obs_type = "lidar" if args.encoder == "lidar" else "screen"
    dataset = SlidingWindowDataset(
        data_dir=args.data_dir,
        history_len=IMG_HIST_LEN,
        rollout_len=args.rollout_len,
        discretize_actions=True,
        obs_type=obs_type,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No valid transitions found in {args.data_dir}")
    print(f"Indexed {len(dataset)} transitions across {len(dataset.shards)} shards")

    train_dataset, val_dataset = dataset.split(val_ratio=0.1, seed=args.seed)
    print(
        f"Dataset split by episode: {len(train_dataset)} train transitions, "
        f"{len(val_dataset)} validation transitions"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    key = jax.random.PRNGKey(args.seed)
    key_enc, key_pred, key_train = jax.random.split(key, 3)

    if args.encoder == "vit":
        encoder = ViTEncoder(cfg.encoder, key_enc)
    elif args.encoder == "lidar":
        encoder = LidarEncoder(cfg.encoder, key_enc)
    else:
        encoder = ConvEncoder(cfg.encoder, key_enc)
    predictor = MLPPredictor(cfg.predictor, key_pred)

    config_dict = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }

    train(
        (encoder, predictor),
        train_dataloader,
        latent_dim=cfg.encoder.latent_dim,
        loss_cfg=cfg.loss,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        key=key_train,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        resume=args.resume,
        val_dataloader=val_dataloader,
        config_dict=config_dict,
    )
    print(f"Training complete. Checkpoints written to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
