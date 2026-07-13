# ruff: noqa: E402
"""
Offline Sub-JEPA pretraining loop.

Optimizes an (Encoder, Predictor) pair jointly: the encoder embeds the
observation at time t and at t+K, the predictor rolls the latent forward
through the K recorded action tokens, and the Sub-JEPA loss matches the
rolled-out latent against the target latent while regularizing the target
distribution to prevent collapse.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, TypeVar, Union

TRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from core.config import IMG_HIST_LEN, load_config
from core.dynamics import MLPPredictor
from core.encoders import ConvEncoder, LidarEncoder, ViTEncoder
from core.interfaces import Encoder, Predictor
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from omegaconf import OmegaConf
from src.config import LossConfig, load_train_config
from src.dataloader import DataLoader, SlidingWindowDataset
from src.loss import generate_projectors, sub_jepa_loss

import wandb

DEFAULT_CONFIG = TRAIN_ROOT.parent / "core" / "config.yaml"
DEFAULT_TRAIN_CONFIG = TRAIN_ROOT / "config.yaml"
DEFAULT_DATA_DIR = TRAIN_ROOT.parent / "win-client" / "data"
DEFAULT_CHECKPOINT_DIR = TRAIN_ROOT.parent / "checkpoints" / "pretrain"


# (encoder, predictor) — both are Equinox modules, so the tuple is a PyTree.
Models = Tuple[Encoder, Predictor]
# Batches cross the JIT boundary: host-side NumPy in, traced Arrays inside.
# Mapping (read-only, covariant) lets both views satisfy the same signature.
Batch = Mapping[str, Union[np.ndarray, Array]]

# Checkpoints hold either the (encoder, predictor) models or an optax state.
CheckpointT = TypeVar("CheckpointT")


def _rollout_latent_sequence(
    predictor: Predictor,
    z0: Float[Array, "latent_dim"],
    actions: Int[Array, "K"],
) -> Float[Array, "K latent_dim"]:
    """Rolls a single latent forward through K action tokens, returning all
    intermediate latents."""

    def step(
        z: Float[Array, "latent_dim"], action: Int[Array, ""]
    ) -> Tuple[Float[Array, "latent_dim"], Float[Array, "latent_dim"]]:
        next_z = predictor(z, action)
        return next_z, next_z

    _, z_seq = jax.lax.scan(step, z0, actions)
    return z_seq


def _rollout_latent(
    predictor: Predictor,
    z0: Float[Array, "latent_dim"],
    actions: Int[Array, "K"],
) -> Float[Array, "latent_dim"]:
    """Rolls a single latent forward through K action tokens."""
    z_seq = _rollout_latent_sequence(predictor, z0, actions)
    return z_seq[-1]


def compute_loss(
    models: Models,
    batch: Batch,
    subspace_projectors: Float[Array, "num_subspaces latent_dim subspace_dim"],
    slice_projectors: Float[Array, "num_subspaces subspace_dim num_slices"],
    reg_weight: float,
) -> Float[Array, ""]:
    encoder, predictor = models

    if isinstance(encoder, LidarEncoder):
        obs_t = {
            "lidar": batch["obs_stack_t"].astype(jnp.float32),
            "telemetry": batch["telemetry_t"],
        }
    else:
        # uint8 -> float32 here so the cast runs on-device after the (4x smaller)
        # uint8 transfer, conserving PCIe bandwidth.
        obs_t = {
            "screen": batch["obs_stack_t"].astype(jnp.float32) / 255.0,
            "telemetry": batch["telemetry_t"],
        }
    actions = batch["actions_seq"].astype(jnp.int32)
    z_t = jax.vmap(encoder)(obs_t)

    if isinstance(encoder, LidarEncoder):
        obs_targets = {
            "lidar": batch["obs_stack_targets"].astype(jnp.float32),
            "telemetry": batch["telemetry_targets"],
        }
    else:
        obs_targets = {
            "screen": batch["obs_stack_targets"].astype(jnp.float32) / 255.0,
            "telemetry": batch["telemetry_targets"],
        }
    z_target_seq = jax.vmap(jax.vmap(encoder))(obs_targets)
    z_pred_seq = jax.vmap(
        lambda z0, acts: _rollout_latent_sequence(predictor, z0, acts)
    )(z_t, actions)

    z_pred_flat = z_pred_seq.reshape(-1, z_pred_seq.shape[-1])
    z_target_flat = z_target_seq.reshape(-1, z_target_seq.shape[-1])
    return sub_jepa_loss(
        z_pred_flat,
        z_target_flat,
        subspace_projectors,
        slice_projectors,
        reg_weight,
    )


def make_train_step(
    optimizer: optax.GradientTransformation,
    subspace_projectors: Float[Array, "num_subspaces latent_dim subspace_dim"],
    slice_projectors: Float[Array, "num_subspaces subspace_dim num_slices"],
    reg_weight: float,
):
    """Builds a JIT-compiled, purely functional optimization step."""

    @eqx.filter_jit
    def train_step(
        models: Models,
        opt_state: optax.OptState,
        batch: Batch,
    ) -> Tuple[Models, optax.OptState, Float[Array, ""]]:
        loss, grads = eqx.filter_value_and_grad(compute_loss)(
            models, batch, subspace_projectors, slice_projectors, reg_weight
        )
        params = eqx.filter(models, eqx.is_array)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_models = eqx.apply_updates(models, updates)
        return new_models, new_opt_state, loss

    return train_step


def save_checkpoint(path: Union[str, Path], tree: PyTree) -> None:
    """Serializes a PyTree (model weights or optimizer state) via Equinox."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, tree)


def load_checkpoint(path: Union[str, Path], template: CheckpointT) -> CheckpointT:
    """Loads leaves into a freshly constructed template of matching structure."""
    return eqx.tree_deserialise_leaves(Path(path), template)


def _last_epoch_index(checkpoint_dir: Path) -> int:
    """Highest epoch number among existing weight checkpoints, or 0."""
    indices = []
    for path in checkpoint_dir.glob("pretrain_model_ep*.eqx"):
        suffix = path.stem.removeprefix("pretrain_model_ep")
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=0)


def train(
    models: Models,
    dataloader: Iterable[Dict[str, np.ndarray]],
    *,
    latent_dim: int,
    loss_cfg: LossConfig,
    num_epochs: int,
    learning_rate: float,
    key: PRNGKeyArray,
    checkpoint_dir: Union[str, Path],
    log_every: int = 50,
    resume: bool = False,
    val_dataloader: Optional[Iterable[Dict[str, np.ndarray]]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Models:
    """
    Runs offline pretraining and returns the trained (encoder, predictor).

    The combined (encoder, predictor) pair is written to `checkpoint_dir`
    after every epoch as `pretrain_model_ep{n}.eqx` plus a rolling
    `pretrain_model_latest.eqx` (and `pretrain_model_best.eqx` on validation
    improvement), with the optimizer state in `pretrain_optstate_latest.eqx`
    and the frozen loss projectors in `projectors.eqx`.

    With `resume=True`, weights, optimizer state, and projectors are restored
    from the rolling files and epoch numbering continues from the highest
    existing epoch checkpoint; `num_epochs` more epochs are then trained.

    Arguments:
      models: Tuple containing (encoder, predictor) Equinox modules
      dataloader: Iterable yielding transition batches
      latent_dim: Dimensionality of latent embedding representation
      loss_cfg: Loss configuration specifying subspace and slice parameters
      num_epochs: Total number of pretraining epochs to run
      learning_rate: Optimizer learning rate
      key: PRNG key for projector initialization
      checkpoint_dir: Directory path for saving output checkpoints
      log_every: Frequency of logging metrics to stdout and wandb
      resume: Flag indicating whether to resume from existing checkpoints
      val_dataloader: Optional validation dataloader for evaluation
      config_dict: Optional dictionary of hyperparameters logged to wandb

    Returns:
      Trained (encoder, predictor) Equinox model tuple
    """
    checkpoint_dir = Path(checkpoint_dir)

    wandb.init(
        project="jepamania",
        mode="offline",
        config=config_dict or {},
    )

    subspace_dim = loss_cfg.subspace_dim or latent_dim // loss_cfg.num_subspaces
    key_proj, _ = jax.random.split(key)
    # Frozen random orthogonal bases; the freshly generated values are only
    # templates when resuming — the serialized ones are restored below, so a
    # resumed run keeps its original loss surface regardless of `key`.
    subspace_projectors, slice_projectors = generate_projectors(
        key_proj,
        latent_dim=latent_dim,
        num_subspaces=loss_cfg.num_subspaces,
        subspace_dim=subspace_dim,
        num_slices=loss_cfg.num_slices,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(models, eqx.is_array))

    projectors_path = checkpoint_dir / "projectors.eqx"
    start_epoch = 0
    if resume:
        model_path = checkpoint_dir / "pretrain_model_latest.eqx"
        latest_opt_path = checkpoint_dir / "pretrain_optstate_latest.eqx"
        for path in (model_path, latest_opt_path, projectors_path):
            if not path.exists():
                raise FileNotFoundError(f"Cannot resume: missing {path}")
        models = load_checkpoint(model_path, models)
        opt_state = load_checkpoint(latest_opt_path, opt_state)
        subspace_projectors, slice_projectors = load_checkpoint(
            projectors_path, (subspace_projectors, slice_projectors)
        )
        start_epoch = _last_epoch_index(checkpoint_dir)
        print(f"Resumed from {model_path} at epoch {start_epoch}")
    else:
        save_checkpoint(projectors_path, (subspace_projectors, slice_projectors))

    train_step = make_train_step(
        optimizer, subspace_projectors, slice_projectors, loss_cfg.reg_weight
    )

    @eqx.filter_jit
    def eval_step(
        models_eval: Models,
        batch_eval: Batch,
    ) -> Float[Array, ""]:
        return compute_loss(
            models_eval,
            batch_eval,
            subspace_projectors,
            slice_projectors,
            loss_cfg.reg_weight,
        )

    best_val_loss = float("inf")
    if resume and val_dataloader is not None:
        best_path = checkpoint_dir / "pretrain_model_best.eqx"
        if best_path.exists():
            best_models = load_checkpoint(best_path, models)
            val_losses = [eval_step(best_models, b) for b in val_dataloader]
            if val_losses:
                best_val_loss = float(jnp.mean(jnp.stack(val_losses)))
                print(
                    f"Restored best validation loss from {best_path}: "
                    f"{best_val_loss:.6f}"
                )

    last_epoch = start_epoch + num_epochs
    global_step = 0
    try:
        for epoch in range(start_epoch, last_epoch):
            # Keep losses as device scalars; a float() every step would block the
            # host on each result and defeat JAX async dispatch.
            epoch_losses = []
            epoch_start = time.time()

            for batch in dataloader:
                models, opt_state, loss = train_step(models, opt_state, batch)
                epoch_losses.append(loss)
                global_step += 1

                if global_step == 1 or global_step % log_every == 0:
                    print(
                        f"epoch {epoch + 1}/{last_epoch} | step {global_step} | "
                        f"loss {float(loss):.6f}"
                    )
                    wandb.log(
                        {
                            "train/step_loss": float(loss),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

            if not epoch_losses:
                raise RuntimeError(
                    "DataLoader yielded no batches; check data_dir and batch_size."
                )

            mean_loss = float(jnp.mean(jnp.stack(epoch_losses)))
            duration = time.time() - epoch_start

            val_loss_str = ""
            val_mean_loss = None
            if val_dataloader is not None:
                val_losses = []
                for val_batch in val_dataloader:
                    val_losses.append(eval_step(models, val_batch))
                if val_losses:
                    val_mean_loss = float(jnp.mean(jnp.stack(val_losses)))
                    val_loss_str = f" | val loss {val_mean_loss:.6f}"

            print(
                f"epoch {epoch + 1}/{last_epoch} done | "
                f"mean loss {mean_loss:.6f}{val_loss_str} | "
                f"{len(epoch_losses)} batches in {duration:.1f}s"
            )

            log_dict = {
                "train/epoch_loss": mean_loss,
                "epoch": epoch + 1,
                "duration_s": duration,
            }
            if val_mean_loss is not None:
                log_dict["val/epoch_loss"] = val_mean_loss
            wandb.log(log_dict, step=global_step)

            # Weights and optimizer state live in separate files so deployment
            # ships weights-only while resume can restore the Adam moments.
            save_checkpoint(
                checkpoint_dir / f"pretrain_model_ep{epoch + 1}.eqx", models
            )
            save_checkpoint(checkpoint_dir / "pretrain_model_latest.eqx", models)
            save_checkpoint(checkpoint_dir / "pretrain_optstate_latest.eqx", opt_state)

            if val_mean_loss is not None and val_mean_loss < best_val_loss:
                best_val_loss = val_mean_loss
                save_checkpoint(checkpoint_dir / "pretrain_model_best.eqx", models)
    finally:
        wandb.finish()

    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Sub-JEPA pretraining")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory of HDF5 shards",
    )
    parser.add_argument(
        "--encoder",
        choices=["vit", "conv", "lidar"],
        default="vit",
        help="Encoder backbone (must match the checkpoint when resuming)",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the rolling checkpoints in --checkpoint-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if DEFAULT_CONFIG.exists():
        print(f"Loading model config from {DEFAULT_CONFIG}")
    else:
        print(
            f"Warning: model config {DEFAULT_CONFIG} not found; "
            "using structured defaults"
        )
    cfg = load_config(str(DEFAULT_CONFIG))

    if DEFAULT_TRAIN_CONFIG.exists():
        print(f"Loading train config from {DEFAULT_TRAIN_CONFIG}")
    else:
        print(
            f"Warning: train config {DEFAULT_TRAIN_CONFIG} not found; using defaults"
        )
    train_cfg = load_train_config(str(DEFAULT_TRAIN_CONFIG))

    epochs = train_cfg.pretrain.epochs
    batch_size = train_cfg.pretrain.batch_size
    lr = train_cfg.pretrain.lr
    seed = train_cfg.pretrain.seed

    obs_type = "lidar" if args.encoder == "lidar" else "screen"
    dataset = SlidingWindowDataset(
        data_dir=args.data_dir,
        history_len=IMG_HIST_LEN,
        rollout_len=train_cfg.pretrain.rollout_len,
        discretize_actions=True,
        obs_type=obs_type,
        max_cache_bytes=int(train_cfg.pretrain.max_cache_gb * 1024**3),
    )
    if len(dataset) == 0:
        raise SystemExit(f"No valid transitions found in {args.data_dir}")
    print(f"Indexed {len(dataset)} transitions across {len(dataset.shards)} shards")

    train_dataset, val_dataset = dataset.split(
        val_ratio=train_cfg.pretrain.val_ratio, seed=seed
    )
    print(
        f"Dataset split by episode: {len(train_dataset)} train transitions, "
        f"{len(val_dataset)} validation transitions"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=train_cfg.pretrain.num_workers,
        seed=seed,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=train_cfg.pretrain.num_workers,
        seed=seed,
    )

    key = jax.random.PRNGKey(seed)
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
        "train_cfg": OmegaConf.to_container(train_cfg, resolve=True),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }

    train(
        (encoder, predictor),
        train_dataloader,
        latent_dim=cfg.encoder.latent_dim,
        loss_cfg=train_cfg.loss,
        num_epochs=epochs,
        learning_rate=lr,
        key=key_train,
        checkpoint_dir=args.checkpoint_dir,
        log_every=train_cfg.pretrain.log_every,
        resume=args.resume,
        val_dataloader=val_dataloader,
        config_dict=config_dict,
    )
    print(f"Training complete. Checkpoints written to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
