"""RL fine-tuning pipeline for joint adaptation of Sub-JEPA models and value head."""
# ruff: noqa: E402

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

TRAIN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from core.config import IMG_HIST_LEN, SubJepaConfig, load_config
from core.dynamics import MLPPredictor, MLPValueHead
from core.encoders import ConvEncoder, LidarEncoder, ViTEncoder, load_models_auto
from jaxtyping import Array, Float
from omegaconf import OmegaConf

import wandb
from train.config import TrainConfig, load_train_config
from train.dataloader import DataLoader, SlidingWindowDataset
from train.loss import generate_projectors, sub_jepa_loss
from train.pretrain import (
    _rollout_latent,
    load_checkpoint,
    save_checkpoint,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_CONFIG = TRAIN_ROOT.parent / "core" / "config.yaml"
DEFAULT_TRAIN_CONFIG = TRAIN_ROOT / "config.yaml"
DEFAULT_DATA_DIR = TRAIN_ROOT.parent / "win-client" / "data"
DEFAULT_CHECKPOINT = (
    TRAIN_ROOT.parent / "checkpoints" / "pretrain" / "pretrain_model_latest.eqx"
)
DEFAULT_OUTPUT_DIR = TRAIN_ROOT.parent / "checkpoints" / "finetune"

Encoder = Union[ViTEncoder, ConvEncoder, LidarEncoder]
RLModels = Tuple[Encoder, MLPPredictor, MLPValueHead]
Batch = Mapping[str, Union[np.ndarray, Array]]

Projectors = Tuple[
    Float[Array, "num_subspaces latent_dim subspace_dim"],
    Float[Array, "num_subspaces subspace_dim num_slices"],
]


def _extract_obs(
    encoder: Encoder, batch: Batch, is_target: bool = False
) -> Dict[str, Any]:
    if is_target:
        obs_raw = batch["obs_stack_targets"][:, -1]
        telem_raw = batch["telemetry_targets"][:, -1]
    else:
        obs_raw = batch["obs_stack_t"]
        telem_raw = batch["telemetry_t"]

    if isinstance(encoder, LidarEncoder):
        return {
            "lidar": obs_raw.astype(jnp.float32),
            "telemetry": telem_raw,
        }
    else:
        return {
            "screen": obs_raw.astype(jnp.float32) / 255.0,
            "telemetry": telem_raw,
        }


def compute_value_loss(
    value_head: MLPValueHead,
    encoder: Encoder,
    batch: Batch,
    gamma: float,
    stop_grad_encoder: bool = True,
) -> Float[Array, ""]:
    """Computes K-step discounted return value loss."""
    obs_t = _extract_obs(encoder, batch, is_target=False)
    obs_target = _extract_obs(encoder, batch, is_target=True)

    z_t = jax.vmap(encoder)(obs_t)
    if stop_grad_encoder:
        z_t = jax.lax.stop_gradient(z_t)
    z_target = jax.lax.stop_gradient(jax.vmap(encoder)(obs_target))

    v_t = jax.vmap(value_head)(z_t)
    v_target = jax.lax.stop_gradient(jax.vmap(value_head)(z_target))

    rewards = batch["rewards_seq"].astype(jnp.float32)
    k_steps = rewards.shape[1]
    discounts = gamma ** jnp.arange(k_steps, dtype=jnp.float32)
    disc_reward = jnp.sum(rewards * discounts, axis=1)

    target_return = disc_reward + (gamma**k_steps) * v_target
    return jnp.mean(optax.huber_loss(v_t, target_return))


def make_warmup_step(optimizer: optax.GradientTransformation, gamma: float):
    @eqx.filter_jit
    def warmup_step(
        value_head: MLPValueHead,
        frozen_encoder: Encoder,
        opt_state: optax.OptState,
        batch: Batch,
    ) -> Tuple[MLPValueHead, optax.OptState, Float[Array, ""]]:
        loss, grads = eqx.filter_value_and_grad(compute_value_loss)(
            value_head, frozen_encoder, batch, gamma, True
        )
        params = eqx.filter(value_head, eqx.is_array)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_value_head = eqx.apply_updates(value_head, updates)
        return new_value_head, new_opt_state, loss

    return warmup_step


def compute_joint_loss(
    models: RLModels,
    batch: Batch,
    subspace_projectors: Float[Array, "num_subspaces latent_dim subspace_dim"],
    slice_projectors: Float[Array, "num_subspaces subspace_dim num_slices"],
    reg_weight: float,
    gamma: float,
    value_weight: float,
) -> Tuple[Float[Array, ""], Tuple[Float[Array, ""], Float[Array, ""]]]:
    encoder, predictor, value_head = models

    obs_t = _extract_obs(encoder, batch, is_target=False)
    obs_target = _extract_obs(encoder, batch, is_target=True)

    z_t = jax.vmap(encoder)(obs_t)
    z_target = jax.vmap(encoder)(obs_target)

    actions = batch["actions_seq"].astype(jnp.int32)
    z_pred = jax.vmap(lambda z0, acts: _rollout_latent(predictor, z0, acts))(
        z_t, actions
    )
    jepa_loss = sub_jepa_loss(
        z_pred, z_target, subspace_projectors, slice_projectors, reg_weight
    )

    v_t = jax.vmap(value_head)(z_t)

    v_target = jax.lax.stop_gradient(
        jax.vmap(value_head)(jax.lax.stop_gradient(z_target))
    )

    rewards = batch["rewards_seq"].astype(jnp.float32)
    k_steps = rewards.shape[1]
    discounts = gamma ** jnp.arange(k_steps, dtype=jnp.float32)
    disc_reward = jnp.sum(rewards * discounts, axis=1)
    target_return = disc_reward + (gamma**k_steps) * v_target

    val_loss = jnp.mean(optax.huber_loss(v_t, target_return))

    total_loss = jepa_loss + value_weight * val_loss
    return total_loss, (jepa_loss, val_loss)


def _get_param_labels(models: RLModels):
    encoder, predictor, value_head = models
    return (
        jax.tree.map(lambda _: "enc_pred", encoder),
        jax.tree.map(lambda _: "enc_pred", predictor),
        jax.tree.map(lambda _: "val_head", value_head),
    )


def make_joint_step(
    optimizer: optax.GradientTransformation,
    subspace_projectors: Float[Array, "num_subspaces latent_dim subspace_dim"],
    slice_projectors: Float[Array, "num_subspaces subspace_dim num_slices"],
    reg_weight: float,
    gamma: float,
    value_weight: float,
):
    @eqx.filter_jit
    def joint_step(
        models: RLModels,
        opt_state: optax.OptState,
        batch: Batch,
    ) -> Tuple[
        RLModels, optax.OptState, Float[Array, ""], Float[Array, ""], Float[Array, ""]
    ]:
        (total_loss, (jepa_loss, val_loss)), grads = eqx.filter_value_and_grad(
            compute_joint_loss, has_aux=True
        )(
            models,
            batch,
            subspace_projectors,
            slice_projectors,
            reg_weight,
            gamma,
            value_weight,
        )
        params = eqx.filter(models, eqx.is_array)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_models = eqx.apply_updates(models, updates)
        return new_models, new_opt_state, total_loss, jepa_loss, val_loss

    return joint_step


def _load_projectors(
    checkpoint: Path,
    key: jax.Array,
    latent_dim: int,
    loss_cfg,
) -> Projectors:
    """Loads the frozen loss projectors written next to the model checkpoint."""
    projectors_path = checkpoint.parent / "projectors.eqx"
    if not projectors_path.exists():
        pretrain_fallback = checkpoint.parent.parent / "pretrain" / "projectors.eqx"
        if pretrain_fallback.exists():
            projectors_path = pretrain_fallback
        else:
            raise FileNotFoundError(
                f"Frozen loss projectors not found at {projectors_path} or "
                f"{pretrain_fallback}. Fine-tuning must reuse the projectors "
                "saved during pretraining."
            )
    subspace_dim = loss_cfg.subspace_dim or latent_dim // loss_cfg.num_subspaces
    template = generate_projectors(
        key,
        latent_dim=latent_dim,
        num_subspaces=loss_cfg.num_subspaces,
        subspace_dim=subspace_dim,
        num_slices=loss_cfg.num_slices,
    )
    return load_checkpoint(projectors_path, template)


def train_rl(
    data_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    value_head_path: Optional[Path],
    model_cfg: SubJepaConfig,
    train_cfg: TrainConfig,
) -> RLModels:
    """Runs one stateless RL fine-tuning pass over the given dataset.

    Loads a combined (encoder, predictor) checkpoint, then either loads the
    given value head (skipping bootstrap) or random-initializes one and
    bootstraps it on the frozen encoder. Both are then fine-tuned jointly,
    with `ft_model_latest.eqx` and `ft_value_head_latest.eqx` written to
    `output_dir` at the end.

    Arguments:
      data_dir: Path to reward-labeled HDF5 dataset directory
      output_dir: Destination directory for the fine-tuned checkpoints
      checkpoint: Combined (encoder, predictor) Equinox checkpoint; the frozen
        loss projectors are loaded from `projectors.eqx` alongside it
      value_head_path: Optional value head checkpoint; when given, the
        bootstrap warmup phase is skipped
      model_cfg: Model architecture configuration
      train_cfg: Training configuration (finetune and loss sections are used)

    Returns:
      Tuple containing fine-tuned (encoder, predictor, value_head) models
    """
    ft_cfg = train_cfg.finetune
    loss_cfg = train_cfg.loss
    latent_dim = model_cfg.encoder.latent_dim

    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(ft_cfg.seed)
    key_models, key_val, key_proj = jax.random.split(key, 3)

    (encoder, predictor), detected_type = load_models_auto(
        checkpoint, key_models, model_cfg.encoder, model_cfg.predictor
    )
    logging.info(f"Loaded Sub-JEPA models ({detected_type} encoder) from {checkpoint}")
    obs_type = "lidar" if detected_type == "lidar" else "screen"

    subspace_projectors, slice_projectors = _load_projectors(
        checkpoint, key_proj, latent_dim, loss_cfg
    )
    save_checkpoint(
        output_dir / "projectors.eqx", (subspace_projectors, slice_projectors)
    )

    logging.info(f"Loading dataset from {data_dir} (obs_type={obs_type})...")
    dataset = SlidingWindowDataset(
        data_dir=data_dir,
        history_len=IMG_HIST_LEN,
        rollout_len=ft_cfg.rollout_len,
        discretize_actions=True,
        obs_type=obs_type,
        load_rewards=True,
        max_cache_bytes=int(ft_cfg.max_cache_gb * 1024**3),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No valid transitions found in {data_dir}")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=ft_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=ft_cfg.num_workers,
        seed=ft_cfg.seed,
    )

    wandb.init(
        project="jepamania",
        mode="offline",
        config={
            "cfg": OmegaConf.to_container(model_cfg, resolve=True),
            "train_cfg": OmegaConf.to_container(train_cfg, resolve=True),
            "checkpoint": str(checkpoint),
            "value_head": str(value_head_path) if value_head_path else None,
        },
    )
    global_step = 0
    try:
        if value_head_path is not None:
            value_head = load_checkpoint(
                value_head_path, MLPValueHead(model_cfg.value_head, key=key_val)
            )
            logging.info(
                f"Loaded value head from {value_head_path}; skipping bootstrap."
            )
        else:
            value_head = MLPValueHead(model_cfg.value_head, key=key_val)
            logging.info(
                f"--- Bootstrapping value head ({ft_cfg.warmup_epochs} epochs) ---"
            )
            warmup_opt = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(ft_cfg.lr_warmup),
            )
            warmup_opt_state = warmup_opt.init(eqx.filter(value_head, eqx.is_array))
            warmup_step = make_warmup_step(warmup_opt, ft_cfg.gamma)

            for epoch in range(ft_cfg.warmup_epochs):
                t0 = time.time()
                losses = []
                for batch in dataloader:
                    value_head, warmup_opt_state, loss = warmup_step(
                        value_head, encoder, warmup_opt_state, batch
                    )
                    losses.append(loss)
                    global_step += 1
                    if global_step % ft_cfg.log_every == 0:
                        wandb.log({"warmup/step_loss": float(loss)}, step=global_step)
                mean_loss = float(jnp.mean(jnp.stack(losses)))
                logging.info(
                    f"[Warmup Epoch {epoch + 1}/{ft_cfg.warmup_epochs}] "
                    f"Value Loss: {mean_loss:.6f} ({time.time() - t0:.1f}s)"
                )
                wandb.log({"warmup/epoch_loss": mean_loss}, step=global_step)

            save_checkpoint(output_dir / "ft_value_head_latest.eqx", value_head)

        models: RLModels = (encoder, predictor, value_head)

        logging.info(
            f"--- Starting Joint Fine-Tuning ({ft_cfg.joint_epochs} epochs, "
            f"lr_enc={ft_cfg.lr_enc}, lr_val={ft_cfg.lr_val}) ---"
        )
        joint_opt = optax.multi_transform(
            {
                "enc_pred": optax.chain(
                    optax.clip_by_global_norm(1.0), optax.adamw(ft_cfg.lr_enc)
                ),
                "val_head": optax.chain(
                    optax.clip_by_global_norm(1.0), optax.adamw(ft_cfg.lr_val)
                ),
            },
            _get_param_labels,
        )
        joint_opt_state = joint_opt.init(eqx.filter(models, eqx.is_array))
        joint_step = make_joint_step(
            joint_opt,
            subspace_projectors,
            slice_projectors,
            loss_cfg.reg_weight,
            ft_cfg.gamma,
            ft_cfg.value_weight,
        )

        for epoch in range(ft_cfg.joint_epochs):
            t0 = time.time()
            total_losses, jepa_losses, val_losses = [], [], []
            for batch in dataloader:
                models, joint_opt_state, t_loss, j_loss, v_loss = joint_step(
                    models, joint_opt_state, batch
                )
                total_losses.append(t_loss)
                jepa_losses.append(j_loss)
                val_losses.append(v_loss)
                global_step += 1
                if global_step % ft_cfg.log_every == 0:
                    wandb.log(
                        {
                            "joint/step_total_loss": float(t_loss),
                            "joint/step_jepa_loss": float(j_loss),
                            "joint/step_value_loss": float(v_loss),
                        },
                        step=global_step,
                    )

            mean_t = float(jnp.mean(jnp.stack(total_losses)))
            mean_j = float(jnp.mean(jnp.stack(jepa_losses)))
            mean_v = float(jnp.mean(jnp.stack(val_losses)))
            logging.info(
                f"[Joint Epoch {epoch + 1}/{ft_cfg.joint_epochs}] "
                f"Total: {mean_t:.6f} | JEPA: {mean_j:.6f} | Val: {mean_v:.6f} "
                f"({time.time() - t0:.1f}s)"
            )
            wandb.log(
                {
                    "joint/epoch_total_loss": mean_t,
                    "joint/epoch_jepa_loss": mean_j,
                    "joint/epoch_value_loss": mean_v,
                },
                step=global_step,
            )

        save_checkpoint(output_dir / "ft_model_latest.eqx", (models[0], models[1]))
        save_checkpoint(output_dir / "ft_value_head_latest.eqx", models[2])
    finally:
        wandb.finish()
        if (
            "dataset" in locals()
            and hasattr(dataset, "_pool")
            and dataset._pool is not None
        ):
            dataset._pool.cache.clear()
            dataset._pool.refcounts.clear()
        if "dataloader" in locals():
            del dataloader
        if "dataset" in locals():
            del dataset

    logging.info(f"Fine-tuning complete. Checkpoints written to {output_dir}")
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sub-JEPA RL Fine-Tuner")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Reward-labeled HDF5 shards dir",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Combined (encoder, predictor) eqx checkpoint",
    )
    parser.add_argument(
        "--value-head",
        type=Path,
        default=None,
        help="Optional value head eqx checkpoint; skips bootstrap when given",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output checkpoints directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = load_config(str(DEFAULT_CONFIG))
    train_cfg = load_train_config(str(DEFAULT_TRAIN_CONFIG))

    train_rl(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        value_head_path=args.value_head,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )


if __name__ == "__main__":
    main()
