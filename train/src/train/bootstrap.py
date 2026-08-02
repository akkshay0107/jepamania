"""Supervised value-head bootstrapping and joint fine-tuning pipeline.

This module loads a pretrained Sub-JEPA world model checkpoint along with its frozen
subspace projectors, either bootstraps a new value head or loads an existing one, and
performs supervised value warmup and joint fine-tuning on pre-recorded dataset rollouts.
It outputs the starting checkpoints required by the online RL loop.
"""
# ruff: noqa: E402

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

TRAIN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from core.config import (
    IMG_HIST_LEN,
    LAMBDA_RETURN_DECAY,
    OBSERVED_ROLLOUT_LEN,
    SubJepaConfig,
    load_config,
)
from core.dynamics import MLPPolicyHead, MLPPredictor, MLPValueHead
from core.encoders import ConvEncoder, LidarEncoder, ViTEncoder, load_models_auto
from jaxtyping import Array, Float
from omegaconf import OmegaConf

from train.config import TrainConfig, load_train_config
from train.dataloader import DataLoader, SlidingWindowDataset
from train.finetune import (
    JointModels,
    _compute_truncated_lambda_return,
    extract_obs,
    get_param_labels,
    load_projectors,
    make_joint_step,
)
from train.policy import (
    make_policy_dataset,
    select_policy_sft_episodes,
    train_policy_sft,
)
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
DEFAULT_DATA_DIR = TRAIN_ROOT.parent / "win-client" / "data" / "rl" / "bootstrap"
DEFAULT_SSL_HUMAN_DIR = TRAIN_ROOT.parent / "win-client" / "data" / "ssl" / "human"
DEFAULT_CHECKPOINT = (
    TRAIN_ROOT.parent / "checkpoints" / "pretrain" / "pretrain_model_latest.eqx"
)
DEFAULT_OUTPUT_DIR = TRAIN_ROOT.parent / "checkpoints" / "finetune"

Encoder = Union[ViTEncoder, ConvEncoder, LidarEncoder]
Predictor = MLPPredictor
ValueModels = Tuple[Encoder, Predictor, MLPValueHead]
SFTModels = Tuple[Encoder, Predictor, MLPValueHead, MLPPolicyHead]
Batch = Mapping[str, Union[np.ndarray, Array]]


def compute_value_loss(
    value_head: MLPValueHead,
    encoder: Encoder,
    batch: Batch,
    gamma: float,
    stop_grad_encoder: bool = False,
    predictor: Optional[Predictor] = None,
) -> Float[Array, ""]:
    """Computes K-step discounted return value loss on z_t and predicted states."""
    obs_t = extract_obs(encoder, batch, is_target=False)
    obs_target = extract_obs(encoder, batch, is_target=True)

    z_t = jax.vmap(encoder)(obs_t)
    z_target = jax.vmap(encoder)(obs_target)

    if stop_grad_encoder:
        z_t = jax.lax.stop_gradient(z_t)
        z_target = jax.lax.stop_gradient(z_target)

    v_t = jax.vmap(value_head)(z_t)
    v_target = jax.lax.stop_gradient(jax.vmap(value_head)(z_target))

    rewards = jnp.asarray(batch["rewards_seq"], dtype=jnp.float32)
    mask = jnp.asarray(batch.get("mask_seq", jnp.ones_like(rewards)), dtype=jnp.float32)

    if predictor is not None and "actions_seq" in batch:
        actions = batch["actions_seq"].astype(jnp.int32)
        z_pred_sequence = jax.vmap(
            lambda z0, acts: _rollout_latent_sequence(predictor, z0, acts)
        )(z_t, actions)
        v_pred_sequence = jax.vmap(jax.vmap(value_head))(
            jax.lax.stop_gradient(z_pred_sequence)
        )
        lambda_return_target = _compute_truncated_lambda_return(
            rewards, v_pred_sequence, mask, gamma, LAMBDA_RETURN_DECAY
        )
        observed_idx = min(z_pred_sequence.shape[1], OBSERVED_ROLLOUT_LEN) - 1
        v_pred_consistency = v_pred_sequence[:, observed_idx]
        loss_t = jnp.mean(optax.huber_loss(v_t, lambda_return_target))
        loss_pred = jnp.mean(optax.huber_loss(v_pred_consistency, v_target))
        return 0.5 * loss_t + 0.5 * loss_pred

    k_steps = rewards.shape[1]
    discounts = gamma ** jnp.arange(k_steps, dtype=jnp.float32)
    disc_reward = jnp.sum(rewards * discounts, axis=1)
    target_return = disc_reward + (gamma**k_steps) * v_target
    return jnp.mean(optax.huber_loss(v_t, target_return))


def make_warmup_step(optimizer: optax.GradientTransformation, gamma: float):
    """Creates a JIT-compiled value-head warmup step function."""

    @eqx.filter_jit
    def warmup_step(
        value_head: MLPValueHead,
        frozen_encoder: Encoder,
        frozen_predictor: Optional[MLPPredictor],
        opt_state: optax.OptState,
        batch: Batch,
    ) -> Tuple[MLPValueHead, optax.OptState, Float[Array, ""]]:
        loss, grads = eqx.filter_value_and_grad(compute_value_loss)(
            value_head, frozen_encoder, batch, gamma, True, frozen_predictor
        )
        params = eqx.filter(value_head, eqx.is_array)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_value_head = eqx.apply_updates(value_head, updates)
        return new_value_head, new_opt_state, loss

    return warmup_step


def train_bootstrap(
    data_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    value_head_path: Optional[Path],
    model_cfg: SubJepaConfig,
    train_cfg: TrainConfig,
    ssl_human_dir: Path = DEFAULT_SSL_HUMAN_DIR,
) -> SFTModels:
    """Run supervised value bootstrapping and joint fine-tuning over historical data.

    Arguments:
      data_dir: Path to reward-labeled HDF5 dataset directory
      output_dir: Destination directory for the fine-tuned checkpoints
      checkpoint: Combined (encoder, predictor) Equinox checkpoint; the frozen
        subspace projectors are loaded from `projectors.eqx` alongside it
      value_head_path: Optional value head checkpoint; when omitted, a random
        value head is bootstrapped on the frozen encoder and predictor
      model_cfg: Model architecture configuration
      train_cfg: Training configuration (bootstrap and loss sections are used)
      ssl_human_dir: Human SSL HDF5 directory used for policy behavior cloning

    Returns:
      Tuple containing the fine-tuned encoder, predictor, value head, and
      behavior-cloned policy head
    """
    boot_cfg = train_cfg.bootstrap
    loss_cfg = train_cfg.loss
    latent_dim = model_cfg.encoder.latent_dim

    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(boot_cfg.seed)
    key_models, key_val, key_proj, key_policy, slice_rng = jax.random.split(key, 5)

    (encoder, predictor), detected_type = load_models_auto(
        checkpoint, key_models, model_cfg.encoder, model_cfg.predictor
    )
    logging.info(f"Loaded Sub-JEPA models ({detected_type} encoder) from {checkpoint}")
    obs_type = "lidar" if detected_type == "lidar" else "screen"

    subspace_projectors = load_projectors(checkpoint, key_proj, latent_dim, loss_cfg)
    save_checkpoint(output_dir / "projectors.eqx", subspace_projectors)

    logging.info(f"Loading dataset from {data_dir} (obs_type={obs_type})...")
    dataset = SlidingWindowDataset(
        data_dir=data_dir,
        history_len=IMG_HIST_LEN,
        rollout_len=boot_cfg.rollout_len,
        discretize_actions=True,
        obs_type=obs_type,
        load_rewards=True,
        max_cache_bytes=int(boot_cfg.max_cache_gb * 1024**3),
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No valid transitions found in {data_dir}")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=boot_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=boot_cfg.num_workers,
        seed=boot_cfg.seed,
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
    global_step = 0
    policy_dataset = None
    try:
        if value_head_path is not None:
            value_head = load_checkpoint(
                value_head_path, MLPValueHead(model_cfg.value_head, key=key_val)
            )
            logging.info(
                f"Loaded value head from {value_head_path}; skipping bootstrap warmup."
            )
        else:
            value_head = MLPValueHead(model_cfg.value_head, key=key_val)
            logging.info(
                f"--- Bootstrapping value head ({boot_cfg.warmup_epochs} epochs) ---"
            )
            warmup_opt = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(boot_cfg.lr_warmup),
            )
            warmup_opt_state = warmup_opt.init(eqx.filter(value_head, eqx.is_array))
            warmup_step = make_warmup_step(warmup_opt, boot_cfg.gamma)

            for epoch in range(boot_cfg.warmup_epochs):
                t0 = time.time()
                losses = []
                for batch in dataloader:
                    value_head, warmup_opt_state, loss = warmup_step(
                        value_head, encoder, predictor, warmup_opt_state, batch
                    )
                    losses.append(loss)
                    global_step += 1
                    if global_step % boot_cfg.log_every == 0:
                        wandb.log({"warmup/step_loss": float(loss)}, step=global_step)
                mean_loss = float(jnp.mean(jnp.stack(losses)))
                logging.info(
                    f"[Warmup Epoch {epoch + 1}/{boot_cfg.warmup_epochs}] "
                    f"Value Loss: {mean_loss:.6f} ({time.time() - t0:.1f}s)"
                )
                wandb.log({"warmup/epoch_loss": mean_loss}, step=global_step)

            save_checkpoint(output_dir / "ft_value_head_latest.eqx", value_head)

        models: JointModels = (encoder, predictor, value_head)

        if boot_cfg.joint_epochs > 0:
            logging.info(
                f"--- Starting Joint Fine-Tuning ({boot_cfg.joint_epochs} epochs, "
                f"lr_enc={boot_cfg.lr_enc}, lr_val={boot_cfg.lr_val}) ---"
            )
            joint_opt = optax.multi_transform(
                {
                    "enc_pred": optax.chain(
                        optax.clip_by_global_norm(1.0), optax.adamw(boot_cfg.lr_enc)
                    ),
                    "val_head": optax.chain(
                        optax.clip_by_global_norm(1.0), optax.adamw(boot_cfg.lr_val)
                    ),
                },
                get_param_labels,
            )
            joint_opt_state = joint_opt.init(eqx.filter(models, eqx.is_array))
            joint_step = make_joint_step(
                joint_opt,
                subspace_projectors,
                loss_cfg.num_slices,
                loss_cfg.reg_weight,
                boot_cfg.gamma,
                boot_cfg.value_weight,
            )

            for epoch in range(boot_cfg.joint_epochs):
                t0 = time.time()
                total_losses, jepa_losses, val_losses = [], [], []
                for batch in dataloader:
                    slice_rng, slice_key = jax.random.split(slice_rng)
                    models, joint_opt_state, t_loss, j_loss, v_loss = joint_step(
                        models, joint_opt_state, batch, slice_key
                    )
                    total_losses.append(t_loss)
                    jepa_losses.append(j_loss)
                    val_losses.append(v_loss)
                    global_step += 1
                    if global_step % boot_cfg.log_every == 0:
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
                    f"[Joint Epoch {epoch + 1}/{boot_cfg.joint_epochs}] "
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

        policy_episodes = select_policy_sft_episodes(
            data_dir,
            ssl_human_dir,
            min_frames=train_cfg.policy_sft.min_frames,
            max_frames=train_cfg.policy_sft.max_frames,
        )
        logging.info(
            "Selected %s successful human episodes for policy SFT",
            len(policy_episodes),
        )
        policy_dataset = make_policy_dataset(
            policy_episodes,
            obs_type=obs_type,
            rollout_len=boot_cfg.rollout_len,
            max_cache_bytes=int(boot_cfg.max_cache_gb * 1024**3),
        )
        policy_head = train_policy_sft(
            models[0],
            policy_dataset,
            train_cfg.policy_sft,
            model_cfg.policy_head,
            key_policy,
            output_dir / "ft_policy_head_latest.eqx",
        )
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
        if policy_dataset is not None:
            policy_dataset._pool.cache.clear()
            policy_dataset._pool.refcounts.clear()
            del policy_dataset

    logging.info(f"Bootstrapping complete. Checkpoints written to {output_dir}")
    return models[0], models[1], models[2], policy_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sub-JEPA Supervised Value Bootstrapper"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Reward-labeled HDF5 shards dir",
    )
    parser.add_argument(
        "--ssl-human-dir",
        type=Path,
        default=DEFAULT_SSL_HUMAN_DIR,
        help="Human SSL HDF5 shards used by policy behavior cloning",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Combined (encoder, predictor) eqx checkpoint from pretraining",
    )
    parser.add_argument(
        "--value-head",
        type=Path,
        default=None,
        help="Optional existing value-head checkpoint; skips warmup when given",
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

    train_bootstrap(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        value_head_path=args.value_head,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        ssl_human_dir=args.ssl_human_dir,
    )


if __name__ == "__main__":
    main()
