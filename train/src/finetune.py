"""RL fine-tuning pipeline for joint adaptation of Sub-JEPA models and value head."""
# ruff: noqa: E402

import argparse
import logging
import sys
import time
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))
from typing import Any, Dict, Mapping, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from core.config import (
    LATENT_DIM,
    EncoderConfig,
    LossConfig,
    PredictorConfig,
    ValueHeadConfig,
)
from core.dynamics import MLPPredictor, MLPValueHead
from core.encoders import ConvEncoder, LidarEncoder
from jaxtyping import Array, Float
from src.dataloader import DataLoader, SlidingWindowDataset
from src.loss import generate_projectors, sub_jepa_loss
from src.pretrain import (
    _rollout_latent,
    load_checkpoint,
    save_checkpoint,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

Encoder = Union[ConvEncoder, LidarEncoder]
RLModels = Tuple[Encoder, MLPPredictor, MLPValueHead]
Batch = Mapping[str, Union[np.ndarray, Array]]


def _extract_obs(
    encoder: Encoder, batch: Batch, is_target: bool = False
) -> Dict[str, Any]:
    key = "obs_stack_target" if is_target else "obs_stack_t"
    telem_key = "telemetry_target" if is_target else "telemetry_t"

    if isinstance(encoder, LidarEncoder):
        return {
            "lidar": batch[key].astype(jnp.float32),
            "telemetry": batch[telem_key],
        }
    else:
        return {
            "screen": batch[key].astype(jnp.float32) / 255.0,
            "telemetry": batch[telem_key],
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


def train_rl(
    data_dir: Path,
    ssl_checkpoint: Path,
    output_dir: Path,
    value_head_checkpoint: Union[Path, None] = None,
    warmup_epochs: int = 5,
    joint_epochs: int = 5,
    gamma: float = 0.990,
    lr_warmup: float = 1e-3,
    lr_enc: float = 1e-5,
    lr_val: float = 3e-4,
    value_weight: float = 1.0,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    obs_type: str = "screen",
) -> RLModels:
    """Executes cyclic RL fine-tuning for Sub-JEPA models and value head.

    Arguments:
      data_dir: Path to rollout or bootstrap HDF5 dataset directory
      ssl_checkpoint: Path to pretrained Sub-JEPA SSL Equinox checkpoint
      output_dir: Destination path for saving fine-tuned checkpoints
      value_head_checkpoint: Optional path to warm-started value head checkpoint
      warmup_epochs: Number of value head warmup epochs before joint fine-tuning
      joint_epochs: Number of epochs for joint encoder-predictor-value fine-tuning
      gamma: Discount factor for K-step return value estimation
      lr_warmup: Learning rate for value head warmup phase
      lr_enc: Learning rate for Sub-JEPA encoder and predictor during joint tuning
      lr_val: Learning rate for value head during joint tuning
      value_weight: Loss weighting factor applied to value head Huber loss
      batch_size: Transition batch size for dataloader
      num_workers: Background prefetching worker count
      seed: Random seed for dataset shuffling and initialization
      obs_type: Observation modality ('screen' or 'lidar')

    Returns:
      Tuple containing fine-tuned (encoder, predictor, value_head) Equinox models
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(seed)

    logging.info(f"Loading dataset from {data_dir} (load_rewards=True)...")
    dataset = SlidingWindowDataset(
        data_dir=data_dir,
        history_len=4,
        rollout_len=5,
        discretize_actions=False,
        obs_type=obs_type,
        load_rewards=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No valid transitions found in {data_dir}")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        seed=seed,
    )

    logging.info(f"Loading pre-trained SSL models from {ssl_checkpoint}...")
    key_enc, key_pred, key_val, key_proj = jax.random.split(key, 4)
    if obs_type == "lidar":
        encoder = LidarEncoder(EncoderConfig(), key=key_enc)
    else:
        encoder = ConvEncoder(EncoderConfig(), key=key_enc)
    predictor = MLPPredictor(PredictorConfig(), key=key_pred)

    template = (encoder, predictor)
    encoder, predictor = load_checkpoint(ssl_checkpoint, template)

    value_head = MLPValueHead(ValueHeadConfig(), key=key_val)
    if value_head_checkpoint is not None:
        path = Path(value_head_checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"Specified value head checkpoint not found: {path}"
            )
        logging.info(f"Loading warm-started value head from {path}...")
        value_head = load_checkpoint(path, value_head)

    if warmup_epochs > 0:
        logging.info(f"--- Starting Value Head Warmup ({warmup_epochs} epochs) ---")
        warmup_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr_warmup),
        )
        warmup_opt_state = warmup_opt.init(eqx.filter(value_head, eqx.is_array))
        warmup_step = make_warmup_step(warmup_opt, gamma)

        for epoch in range(warmup_epochs):
            t0 = time.time()
            losses = []
            for batch in dataloader:
                value_head, warmup_opt_state, loss = warmup_step(
                    value_head, encoder, warmup_opt_state, batch
                )
                losses.append(loss)
            mean_loss = float(jnp.mean(jnp.stack(losses)))
            logging.info(
                f"[Warmup Epoch {epoch + 1}/{warmup_epochs}] "
                f"Value Loss: {mean_loss:.6f} ({time.time() - t0:.1f}s)"
            )

        save_checkpoint(output_dir / "rl_warmup_value_head.eqx", value_head)

    models: RLModels = (encoder, predictor, value_head)

    if joint_epochs > 0:
        logging.info(
            f"--- Starting Joint Fine-Tuning ({joint_epochs} epochs, "
            f"lr_enc={lr_enc}, lr_val={lr_val}) ---"
        )
        loss_cfg = LossConfig()
        subspace_dim = loss_cfg.subspace_dim or LATENT_DIM // loss_cfg.num_subspaces
        subspace_projectors, slice_projectors = generate_projectors(
            key_proj,
            latent_dim=LATENT_DIM,
            num_subspaces=loss_cfg.num_subspaces,
            subspace_dim=subspace_dim,
            num_slices=loss_cfg.num_slices,
        )

        joint_opt = optax.multi_transform(
            {
                "enc_pred": optax.chain(
                    optax.clip_by_global_norm(1.0), optax.adamw(lr_enc)
                ),
                "val_head": optax.chain(
                    optax.clip_by_global_norm(1.0), optax.adamw(lr_val)
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
            gamma,
            value_weight,
        )

        for epoch in range(joint_epochs):
            t0 = time.time()
            total_losses, jepa_losses, val_losses = [], [], []
            for batch in dataloader:
                models, joint_opt_state, t_loss, j_loss, v_loss = joint_step(
                    models, joint_opt_state, batch
                )
                total_losses.append(t_loss)
                jepa_losses.append(j_loss)
                val_losses.append(v_loss)

            mean_t = float(jnp.mean(jnp.stack(total_losses)))
            mean_j = float(jnp.mean(jnp.stack(jepa_losses)))
            mean_v = float(jnp.mean(jnp.stack(val_losses)))
            logging.info(
                f"[Joint Epoch {epoch + 1}/{joint_epochs}] "
                f"Total: {mean_t:.6f} | JEPA: {mean_j:.6f} | Val: {mean_v:.6f} "
                f"({time.time() - t0:.1f}s)"
            )
            save_checkpoint(output_dir / f"rl_joint_epoch_{epoch + 1}.eqx", models)
            save_checkpoint(output_dir / "rl_joint_latest.eqx", models)
            save_checkpoint(
                output_dir / "rl_joint_latest_subjepa.eqx", (models[0], models[1])
            )
            save_checkpoint(output_dir / "rl_joint_latest_value_head.eqx", models[2])

    save_checkpoint(output_dir / "rl_joint_latest.eqx", models)
    save_checkpoint(output_dir / "rl_joint_latest_subjepa.eqx", (models[0], models[1]))
    save_checkpoint(output_dir / "rl_joint_latest_value_head.eqx", models[2])

    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sub-JEPA RL Fine-Tuner")
    parser.add_argument("--data-dir", type=Path, required=True, help="HDF5 shards dir")
    parser.add_argument(
        "--ssl-checkpoint",
        type=Path,
        required=True,
        help="Pretrained SSL eqx checkpoint",
    )
    parser.add_argument(
        "--value-head-checkpoint",
        type=Path,
        default=None,
        help="Optional warm-start value head eqx checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/rl"),
        help="Output checkpoints directory",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--joint-epochs", type=int, default=5, help="Joint epochs")
    parser.add_argument(
        "--obs-type", type=str, default="screen", choices=["screen", "lidar"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rl(
        data_dir=args.data_dir,
        ssl_checkpoint=args.ssl_checkpoint,
        value_head_checkpoint=args.value_head_checkpoint,
        output_dir=args.output_dir,
        warmup_epochs=args.warmup_epochs,
        joint_epochs=args.joint_epochs,
        obs_type=args.obs_type,
    )


if __name__ == "__main__":
    main()
