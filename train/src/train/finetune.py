"""RL fine-tuning pipeline for joint adaptation of Sub-JEPA models and value head."""
# ruff: noqa: E402

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

TRAIN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from core.config import (
    IMG_HIST_LEN,
    LAMBDA_RETURN_DECAY,
    OBSERVED_ROLLOUT_LEN,
    SubJepaConfig,
    load_config,
)
from core.dynamics import MLPPredictor, MLPValueHead
from core.encoders import ConvEncoder, LidarEncoder, ViTEncoder, load_models_auto
from jaxtyping import Array, Float
from omegaconf import OmegaConf

import wandb
from train.config import TrainConfig, load_train_config
from train.dataloader import DataLoader, SlidingWindowDataset
from train.loss import generate_projectors, sub_jepa_loss
from train.pretrain import (
    _rollout_latent_sequence,
    load_checkpoint,
    save_checkpoint,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_CONFIG = TRAIN_ROOT.parent / "core" / "config.yaml"
DEFAULT_TRAIN_CONFIG = TRAIN_ROOT / "config.yaml"
DEFAULT_DATA_DIR = TRAIN_ROOT.parent / "win-client" / "data" / "rl" / "rollouts"
DEFAULT_CHECKPOINT = (
    TRAIN_ROOT.parent / "checkpoints" / "finetune" / "ft_model_latest.eqx"
)
DEFAULT_OUTPUT_DIR = TRAIN_ROOT.parent / "checkpoints" / "finetune"

Encoder = Union[ViTEncoder, ConvEncoder, LidarEncoder]
RLModels = Tuple[Encoder, MLPPredictor, MLPValueHead]
Batch = Mapping[str, Union[np.ndarray, Array]]

Projectors = Tuple[
    Float[Array, "num_subspaces latent_dim subspace_dim"],
    Float[Array, "num_subspaces subspace_dim num_slices"],
]


def extract_obs(
    encoder: Encoder, batch: Batch, is_target: bool = False
) -> Dict[str, Any]:
    """Extracts and normalizes single-step observation dictionaries from batch data.

    Arguments:
      encoder: Vision, Conv, or Lidar observation encoder instance
      batch: Batch mapping containing observation stack and telemetry arrays
      is_target: Flag indicating whether to extract target frame at K_obs step

    Returns:
      Dictionary containing normalized sensor observations (telemetry + screen/lidar)
    """
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


_extract_obs = extract_obs


def _compute_truncated_lambda_return(
    rewards_seq: Float[Array, "batch steps"],
    values_seq: Float[Array, "batch steps"],
    mask_seq: Float[Array, "batch steps"],
    gamma: float,
    lambda_decay: float,
) -> Float[Array, "batch"]:
    """Computes M-step truncated TD(lambda) targets via reverse scan."""
    # Terminal return initializes to the final value estimate values_seq[:, -1].
    # Step k: G_k = r_k + gamma * mask_{k+1} * ((1-lambda)*V_{k+1} + lambda*G_{k+1}).
    terminal_value = values_seq[:, -1]

    def backward_step(next_return, step_data):
        reward, value, mask = step_data
        current_return = reward + gamma * mask * (
            (1.0 - lambda_decay) * value + lambda_decay * next_return
        )
        return current_return, current_return

    # Transpose to (steps, batch) and reverse along step axis for reverse scan
    rewards_rev = jnp.swapaxes(rewards_seq, 0, 1)[::-1]
    values_rev = jnp.swapaxes(values_seq, 0, 1)[::-1]
    mask_rev = jnp.swapaxes(mask_seq, 0, 1)[::-1]

    initial_return, _ = jax.lax.scan(
        backward_step,
        terminal_value,
        (rewards_rev, values_rev, mask_rev),
    )
    return initial_return


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

    obs_t = extract_obs(encoder, batch, is_target=False)
    obs_target = extract_obs(encoder, batch, is_target=True)

    z_t = jax.vmap(encoder)(obs_t)
    z_target = jax.vmap(encoder)(obs_target)

    # Roll forward across all available action tokens
    # (observed + imagined steps up to IMAGINED_ROLLOUT_LEN)
    actions = batch["actions_seq"].astype(jnp.int32)
    z_pred_sequence = jax.vmap(
        lambda z0, acts: _rollout_latent_sequence(predictor, z0, acts)
    )(z_t, actions)

    # Sub-JEPA geometry loss evaluates strictly against the
    # observed ground-truth target latent at step K_obs
    observed_idx = min(z_pred_sequence.shape[1], OBSERVED_ROLLOUT_LEN) - 1
    z_pred_observed = z_pred_sequence[:, observed_idx]
    jepa_loss = sub_jepa_loss(
        z_pred_observed, z_target, subspace_projectors, slice_projectors, reg_weight
    )

    # Evaluate value network across all predicted latents to bootstrap future returns
    v_t = jax.vmap(value_head)(z_t)
    v_pred_sequence = jax.vmap(jax.vmap(value_head))(
        jax.lax.stop_gradient(z_pred_sequence)
    )
    v_target_observed = jax.lax.stop_gradient(
        jax.vmap(value_head)(jax.lax.stop_gradient(z_target))
    )

    rewards = jnp.asarray(batch["rewards_seq"], dtype=jnp.float32)
    mask = jnp.asarray(batch.get("mask_seq", jnp.ones_like(rewards)), dtype=jnp.float32)
    lambda_return_target = _compute_truncated_lambda_return(
        rewards, v_pred_sequence, mask, gamma, LAMBDA_RETURN_DECAY
    )

    # Value loss combines base state TD(lambda)
    # error with observed K_obs latent consistency loss
    v_pred_consistency = v_pred_sequence[:, observed_idx]
    val_loss = 0.5 * jnp.mean(
        optax.huber_loss(v_t, lambda_return_target)
    ) + 0.5 * jnp.mean(optax.huber_loss(v_pred_consistency, v_target_observed))

    total_loss = jepa_loss + value_weight * val_loss
    return total_loss, (jepa_loss, val_loss)


def get_param_labels(models: RLModels):
    encoder, predictor, value_head = models
    return (
        jax.tree.map(lambda _: "enc_pred", encoder),
        jax.tree.map(lambda _: "enc_pred", predictor),
        jax.tree.map(lambda _: "val_head", value_head),
    )


_get_param_labels = get_param_labels


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


def load_projectors(
    checkpoint: Path,
    key: jax.Array,
    latent_dim: int,
    loss_cfg,
) -> Projectors:
    """Loads the frozen loss projectors written next to the model checkpoint."""
    projectors_path = checkpoint.parent / "projectors.eqx"
    if not projectors_path.exists():
        raise FileNotFoundError(
            f"Frozen loss projectors must be next to the model: {projectors_path}"
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


_load_projectors = load_projectors


def train_rl(
    data_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    value_head_path: Path,
    model_cfg: SubJepaConfig,
    train_cfg: TrainConfig,
    episode_ids: Sequence[int] | None = None,
    step_offset: int = 0,
) -> tuple[RLModels, int]:
    """Run one joint online update from an existing fine-tuned checkpoint.

    Arguments:
      data_dir: Path to reward-labeled HDF5 dataset directory
      output_dir: Destination directory for the fine-tuned checkpoints
      checkpoint: Combined (encoder, predictor) Equinox checkpoint; the frozen
        loss projectors are loaded from `projectors.eqx` alongside it
      value_head_path: Required value head checkpoint
      model_cfg: Model architecture configuration
      train_cfg: Training configuration (finetune and loss sections are used)
      episode_ids: Completed rollout episodes selected for this update
      step_offset: Global step offset from prior iterations

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

    subspace_projectors, slice_projectors = load_projectors(
        checkpoint, key_proj, latent_dim, loss_cfg
    )
    save_checkpoint(
        output_dir / "projectors.eqx", (subspace_projectors, slice_projectors)
    )

    logging.info(
        "Loading %s selected episodes from %s", len(episode_ids or ()), data_dir
    )
    dataset = SlidingWindowDataset(
        data_dir=data_dir,
        history_len=IMG_HIST_LEN,
        rollout_len=ft_cfg.rollout_len,
        discretize_actions=True,
        obs_type=obs_type,
        load_rewards=True,
        max_cache_bytes=int(ft_cfg.max_cache_gb * 1024**3),
        episode_ids=episode_ids,
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

    manage_wandb = wandb.run is None
    if manage_wandb:
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
    global_step = step_offset
    try:
        value_head = load_checkpoint(
            value_head_path, MLPValueHead(model_cfg.value_head, key=key_val)
        )

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
            get_param_labels,
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
        if manage_wandb:
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
    return models, global_step


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
        required=True,
        help="Existing value-head checkpoint",
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
