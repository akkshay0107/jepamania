"""RL fine-tuning pipeline for joint adaptation of Sub-JEPA models and value head."""
# ruff: noqa: E402

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
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
from train.loss import generate_subspace_projectors, sub_jepa_loss
from train.policy import (
    make_policy_dataset,
    policy_nll,
    select_policy_sft_episodes,
)
from train.pretrain import (
    _rollout_latent_sequence,
    load_checkpoint,
    save_checkpoint,
)
from train.priority_dataloader import EpisodeSummary, read_completed_episode_summaries

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
DEFAULT_BOOTSTRAP_DATA_DIR = (
    TRAIN_ROOT.parent / "win-client" / "data" / "rl" / "bootstrap"
)
DEFAULT_SSL_HUMAN_DIR = TRAIN_ROOT.parent / "win-client" / "data" / "ssl" / "human"

Encoder = Union[ViTEncoder, ConvEncoder, LidarEncoder]
ValueModels = Tuple[Encoder, MLPPredictor, MLPValueHead]
RLModels = Tuple[Encoder, MLPPredictor, MLPValueHead, MLPPolicyHead]
JointModels = Union[ValueModels, RLModels]
Batch = Mapping[str, Union[np.ndarray, Array]]

SubspaceProjectors = Float[Array, "num_subspaces latent_dim subspace_dim"]


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
    models: JointModels,
    batch: Batch,
    subspace_projectors: SubspaceProjectors,
    num_slices: int,
    slice_key: jax.Array,
    reg_weight: float,
    gamma: float,
    value_weight: float,
) -> Tuple[Float[Array, ""], Tuple[Float[Array, ""], Float[Array, ""]]]:
    encoder, predictor, value_head = models[:3]

    obs_t = extract_obs(encoder, batch, is_target=False)
    obs_target = extract_obs(encoder, batch, is_target=True)

    z_t = jax.vmap(encoder)(obs_t)
    z_target = jax.vmap(encoder)(obs_target)

    actions = batch["actions_seq"].astype(jnp.int32)
    z_pred_sequence = jax.vmap(
        lambda z0, acts: _rollout_latent_sequence(predictor, z0, acts)
    )(z_t, actions)

    # Sub-JEPA geometry loss evaluates strictly against the
    # observed ground-truth target latent at step K_obs
    observed_idx = min(z_pred_sequence.shape[1], OBSERVED_ROLLOUT_LEN) - 1
    z_pred_observed = z_pred_sequence[:, observed_idx]
    jepa_loss = sub_jepa_loss(
        z_pred_observed,
        z_target,
        subspace_projectors,
        num_slices,
        slice_key,
        reg_weight,
    )

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

    v_pred_consistency = v_pred_sequence[:, observed_idx]
    val_loss = 0.5 * jnp.mean(
        optax.huber_loss(v_t, lambda_return_target)
    ) + 0.5 * jnp.mean(optax.huber_loss(v_pred_consistency, v_target_observed))

    total_loss = jepa_loss + value_weight * val_loss
    return total_loss, (jepa_loss, val_loss)


def get_param_labels(models: JointModels):
    encoder, predictor, value_head = models[:3]
    labels = (
        jax.tree.map(lambda _: "enc_pred", encoder),
        jax.tree.map(lambda _: "enc_pred", predictor),
        jax.tree.map(lambda _: "val_head", value_head),
    )
    if len(models) == 4:
        return labels + (jax.tree.map(lambda _: "policy_head", models[3]),)
    return labels


_get_param_labels = get_param_labels


def make_joint_step(
    optimizer: optax.GradientTransformation,
    subspace_projectors: SubspaceProjectors,
    num_slices: int,
    reg_weight: float,
    gamma: float,
    value_weight: float,
):
    @eqx.filter_jit
    def joint_step(
        models: JointModels,
        opt_state: optax.OptState,
        batch: Batch,
        slice_key: jax.Array,
    ) -> Tuple[
        JointModels,
        optax.OptState,
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, ""],
    ]:
        (total_loss, (jepa_loss, val_loss)), grads = eqx.filter_value_and_grad(
            compute_joint_loss, has_aux=True
        )(
            models,
            batch,
            subspace_projectors,
            num_slices,
            slice_key,
            reg_weight,
            gamma,
            value_weight,
        )
        params = eqx.filter(models, eqx.is_array)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_models = eqx.apply_updates(models, updates)
        return new_models, new_opt_state, total_loss, jepa_loss, val_loss

    return joint_step


@dataclass(frozen=True)
class RunningAdvantageStd:
    count: float = 0.0
    total: float = 0.0
    squared_total: float = 0.0

    @property
    def value(self) -> float:
        if self.count <= 1.0:
            return 1.0
        mean = self.total / self.count
        variance = max(self.squared_total / self.count - mean * mean, 1e-6)
        return math.sqrt(variance)

    def update(
        self, count: float, total: float, squared_total: float
    ) -> "RunningAdvantageStd":
        return RunningAdvantageStd(
            self.count + count,
            self.total + total,
            self.squared_total + squared_total,
        )


def advantage_weights(
    advantages: Float[Array, "batch"],
    running_std: float,
    max_weight: float,
) -> Float[Array, "batch"]:
    """Filter non-positive advantages and clip exponential CRR weights."""
    scale = jnp.maximum(jnp.asarray(running_std, dtype=jnp.float32), 1e-6)
    max_log_weight = jnp.log(jnp.asarray(max_weight, dtype=jnp.float32))
    clipped_exponent = jnp.minimum(advantages / scale, max_log_weight)
    return jnp.where(
        advantages > 0.0,
        jnp.exp(clipped_exponent),
        0.0,
    )


def combine_policy_losses(
    online_loss: Float[Array, ""],
    human_loss: Float[Array, ""],
    human_anchor_ratio: float,
    has_online_policy_data: float,
) -> Float[Array, ""]:
    """Mix online and human losses, retaining the full anchor without online data."""
    online_mix = has_online_policy_data * (1.0 - human_anchor_ratio)
    anchor_mix = jnp.where(has_online_policy_data > 0.0, human_anchor_ratio, 1.0)
    return online_mix * online_loss + anchor_mix * human_loss


def compute_online_policy_loss(
    models: RLModels,
    batch: Batch,
    gamma: float,
    running_advantage_std: float,
    max_advantage_weight: float,
) -> tuple[
    Float[Array, ""],
    tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]],
]:
    """Compute latent-stopped, advantage-weighted NLL for executed actions."""
    encoder, predictor, value_head, policy_head = models
    obs_t = extract_obs(encoder, batch, is_target=False)
    latents = jax.lax.stop_gradient(jax.vmap(encoder)(obs_t))
    actions = jnp.asarray(batch["actions_seq"], dtype=jnp.int32)
    predicted_latents = jax.vmap(
        lambda latent, action_seq: _rollout_latent_sequence(
            predictor, latent, action_seq
        )
    )(latents, actions)
    predicted_values = jax.vmap(jax.vmap(value_head))(
        jax.lax.stop_gradient(predicted_latents)
    )
    current_values = jax.vmap(value_head)(latents)
    rewards = jnp.asarray(batch["rewards_seq"], dtype=jnp.float32)
    masks = jnp.asarray(
        batch.get("mask_seq", jnp.ones_like(rewards)), dtype=jnp.float32
    )
    lambda_returns = _compute_truncated_lambda_return(
        rewards,
        predicted_values,
        masks,
        gamma,
        LAMBDA_RETURN_DECAY,
    )
    advantages = jax.lax.stop_gradient(lambda_returns - current_values)
    weights = advantage_weights(advantages, running_advantage_std, max_advantage_weight)
    logits = jax.vmap(policy_head.raw_logits)(latents)
    nll = optax.softmax_cross_entropy_with_integer_labels(logits, actions[:, 0])
    positive = advantages > 0.0
    positive_count = jnp.sum(positive)
    loss = jnp.sum(weights * nll) / jnp.maximum(positive_count, 1.0)
    positive_advantages = jnp.where(positive, advantages, 0.0)
    return loss, (
        positive_count,
        jnp.sum(positive_advantages),
        jnp.sum(jnp.square(positive_advantages)),
    )


def compute_online_loss(
    models: RLModels,
    replay_batch: Batch,
    online_policy_batch: Batch,
    human_anchor_batch: Batch,
    subspace_projectors: SubspaceProjectors,
    num_slices: int,
    slice_key: jax.Array,
    reg_weight: float,
    gamma: float,
    value_weight: float,
    policy_weight: float,
    human_anchor_ratio: float,
    running_advantage_std: float,
    max_advantage_weight: float,
    has_online_policy_data: float,
):
    joint_loss, (jepa_loss, value_loss) = compute_joint_loss(
        models,
        replay_batch,
        subspace_projectors,
        num_slices,
        slice_key,
        reg_weight,
        gamma,
        value_weight,
    )
    online_loss, advantage_stats = compute_online_policy_loss(
        models,
        online_policy_batch,
        gamma,
        running_advantage_std,
        max_advantage_weight,
    )
    human_loss = policy_nll(models[3], models[0], human_anchor_batch)
    combined_policy_loss = combine_policy_losses(
        online_loss,
        human_loss,
        human_anchor_ratio,
        has_online_policy_data,
    )
    total_loss = joint_loss + policy_weight * combined_policy_loss
    return total_loss, (
        jepa_loss,
        value_loss,
        online_loss,
        human_loss,
        combined_policy_loss,
        advantage_stats,
    )


def make_online_step(
    optimizer: optax.GradientTransformation,
    subspace_projectors: SubspaceProjectors,
    num_slices: int,
    reg_weight: float,
    gamma: float,
    value_weight: float,
    policy_weight: float,
    human_anchor_ratio: float,
    max_advantage_weight: float,
):
    @eqx.filter_jit
    def step(
        models: RLModels,
        opt_state: optax.OptState,
        replay_batch: Batch,
        online_policy_batch: Batch,
        human_anchor_batch: Batch,
        running_advantage_std: float,
        has_online_policy_data: float,
        slice_key: jax.Array,
    ):
        (loss, aux), gradients = eqx.filter_value_and_grad(
            compute_online_loss, has_aux=True
        )(
            models,
            replay_batch,
            online_policy_batch,
            human_anchor_batch,
            subspace_projectors,
            num_slices,
            slice_key,
            reg_weight,
            gamma,
            value_weight,
            policy_weight,
            human_anchor_ratio,
            running_advantage_std,
            max_advantage_weight,
            has_online_policy_data,
        )
        updates, opt_state = optimizer.update(
            gradients, opt_state, eqx.filter(models, eqx.is_array)
        )
        return eqx.apply_updates(models, updates), opt_state, loss, aux

    return step


def qualifying_online_episode_ids(
    summaries: Sequence[EpisodeSummary],
    selected_episode_ids: Sequence[int],
    incumbent_median_frames: float | None,
) -> tuple[int, ...]:
    """Select completed MPC laps strictly faster than the incumbent median."""
    if incumbent_median_frames is None:
        return ()
    selected = set(int(episode_id) for episode_id in selected_episode_ids)
    return tuple(
        summary.episode_id
        for summary in summaries
        if summary.episode_id in selected
        and summary.source == "mpc_driver"
        and summary.termination == "done"
        and summary.transition_count < incumbent_median_frames
    )


def _merge_batches(primary: Batch, anchor: Batch) -> dict[str, np.ndarray]:
    return {
        name: np.concatenate(
            [np.asarray(primary[name]), np.asarray(anchor[name])], axis=0
        )
        for name in primary.keys() & anchor.keys()
    }


def _next_cycling(loader: DataLoader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def mixed_replay_batches(
    online_dataset: SlidingWindowDataset,
    bootstrap_dataset: SlidingWindowDataset,
    batch_size: int,
    bootstrap_ratio: float,
    num_workers: int,
    seed: int,
):
    """Yield replay batches containing approximately the requested anchor ratio."""
    bootstrap_size = max(1, int(round(batch_size * bootstrap_ratio)))
    online_size = max(1, batch_size - bootstrap_size)
    online_loader = DataLoader(
        online_dataset,
        online_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        seed=seed,
    )
    bootstrap_loader = DataLoader(
        bootstrap_dataset,
        bootstrap_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        seed=seed + 1,
    )
    bootstrap_iterator = iter(bootstrap_loader)
    for online_batch in online_loader:
        bootstrap_batch, bootstrap_iterator = _next_cycling(
            bootstrap_loader, bootstrap_iterator
        )
        yield _merge_batches(online_batch, bootstrap_batch)


def load_projectors(
    checkpoint: Path,
    key: jax.Array,
    latent_dim: int,
    loss_cfg,
) -> SubspaceProjectors:
    """Load the frozen subspace projectors written next to the checkpoint."""
    projectors_path = checkpoint.parent / "projectors.eqx"
    if not projectors_path.exists():
        raise FileNotFoundError(
            f"Frozen subspace projectors must be next to the model: {projectors_path}"
        )
    subspace_dim = loss_cfg.subspace_dim or latent_dim // loss_cfg.num_subspaces
    template = generate_subspace_projectors(
        key,
        latent_dim=latent_dim,
        num_subspaces=loss_cfg.num_subspaces,
        subspace_dim=subspace_dim,
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
    policy_head_path: Path,
    episode_ids: Sequence[int] | None = None,
    step_offset: int = 0,
    incumbent_median_frames: float | None = None,
    bootstrap_data_dir: Path = DEFAULT_BOOTSTRAP_DATA_DIR,
    ssl_human_dir: Path = DEFAULT_SSL_HUMAN_DIR,
) -> tuple[RLModels, int]:
    """Run one four-component online update with conservative policy regression.

    Arguments:
      data_dir: Path to reward-labeled HDF5 dataset directory
      output_dir: Destination directory for the fine-tuned checkpoints
        checkpoint: Combined (encoder, predictor) Equinox checkpoint; the frozen
        subspace projectors are loaded from `projectors.eqx` alongside it
      value_head_path: Required value head checkpoint
      policy_head_path: Policy checkpoint used to initialize the policy head
      model_cfg: Model architecture configuration
      train_cfg: Training configuration (finetune and loss sections are used)
      episode_ids: Completed rollout episodes selected for this update
      step_offset: Global step offset from prior iterations
      incumbent_median_frames: Successful incumbent median used to qualify faster laps
      bootstrap_data_dir: Historical reward-labeled data mixed into replay
      ssl_human_dir: Successful human demonstrations used as the policy anchor

    Returns:
      Fine-tuned (encoder, predictor, value_head, policy_head) and global step
    """
    ft_cfg = train_cfg.finetune
    loss_cfg = train_cfg.loss
    latent_dim = model_cfg.encoder.latent_dim

    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(ft_cfg.seed)
    key_models, key_val, key_policy, key_proj, slice_rng = jax.random.split(key, 5)

    (encoder, predictor), detected_type = load_models_auto(
        checkpoint, key_models, model_cfg.encoder, model_cfg.predictor
    )
    logging.info(f"Loaded Sub-JEPA models ({detected_type} encoder) from {checkpoint}")
    obs_type = "lidar" if detected_type == "lidar" else "screen"

    subspace_projectors = load_projectors(checkpoint, key_proj, latent_dim, loss_cfg)
    save_checkpoint(output_dir / "projectors.eqx", subspace_projectors)

    selected_episode_ids = tuple(episode_ids or ())
    logging.info(
        "Loading %s selected episodes from %s", len(selected_episode_ids), data_dir
    )
    online_dataset = SlidingWindowDataset(
        data_dir=data_dir,
        history_len=IMG_HIST_LEN,
        rollout_len=ft_cfg.rollout_len,
        discretize_actions=True,
        obs_type=obs_type,
        load_rewards=True,
        max_cache_bytes=int(ft_cfg.max_cache_gb * 1024**3),
        episode_ids=selected_episode_ids,
    )
    if len(online_dataset) == 0:
        raise RuntimeError(f"No valid transitions found in {data_dir}")
    bootstrap_dataset = SlidingWindowDataset(
        data_dir=bootstrap_data_dir,
        history_len=IMG_HIST_LEN,
        rollout_len=ft_cfg.rollout_len,
        discretize_actions=True,
        obs_type=obs_type,
        load_rewards=True,
        max_cache_bytes=int(ft_cfg.max_cache_gb * 1024**3),
    )
    if len(bootstrap_dataset) == 0:
        raise RuntimeError(f"No bootstrap transitions found in {bootstrap_data_dir}")

    summaries = read_completed_episode_summaries(data_dir)
    qualifying_ids = qualifying_online_episode_ids(
        summaries, selected_episode_ids, incumbent_median_frames
    )
    online_policy_dataset = SlidingWindowDataset(
        data_dir=data_dir,
        history_len=IMG_HIST_LEN,
        rollout_len=ft_cfg.rollout_len,
        discretize_actions=True,
        obs_type=obs_type,
        load_rewards=True,
        max_cache_bytes=int(ft_cfg.max_cache_gb * 1024**3),
        episode_ids=qualifying_ids,
    )
    human_episodes = select_policy_sft_episodes(
        bootstrap_data_dir,
        ssl_human_dir,
        min_frames=train_cfg.policy_sft.min_frames,
        max_frames=train_cfg.policy_sft.max_frames,
    )
    human_dataset = make_policy_dataset(
        human_episodes,
        obs_type=obs_type,
        rollout_len=ft_cfg.rollout_len,
        max_cache_bytes=int(ft_cfg.max_cache_gb * 1024**3),
    )
    if len(human_dataset) == 0:
        raise RuntimeError("No successful-human policy anchor transitions found")

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
                "policy_head": str(policy_head_path),
            },
        )
    global_step = step_offset
    try:
        value_head = load_checkpoint(
            value_head_path, MLPValueHead(model_cfg.value_head, key=key_val)
        )
        policy_head = load_checkpoint(
            policy_head_path,
            MLPPolicyHead(model_cfg.policy_head, key=key_policy),
        )
        models: RLModels = (encoder, predictor, value_head, policy_head)

        logging.info(
            f"--- Starting Joint Fine-Tuning ({ft_cfg.joint_epochs} epochs, "
            f"lr_enc={ft_cfg.lr_enc}, lr_val={ft_cfg.lr_val}) ---"
        )
        online_optimizer = optax.multi_transform(
            {
                "enc_pred": optax.chain(
                    optax.clip_by_global_norm(1.0), optax.adamw(ft_cfg.lr_enc)
                ),
                "val_head": optax.chain(
                    optax.clip_by_global_norm(1.0), optax.adamw(ft_cfg.lr_val)
                ),
                "policy_head": optax.chain(
                    optax.clip_by_global_norm(1.0),
                    optax.adamw(ft_cfg.lr_policy),
                ),
            },
            get_param_labels,
        )
        opt_state = online_optimizer.init(eqx.filter(models, eqx.is_array))
        online_step = make_online_step(
            online_optimizer,
            subspace_projectors,
            loss_cfg.num_slices,
            loss_cfg.reg_weight,
            ft_cfg.gamma,
            ft_cfg.value_weight,
            ft_cfg.policy_weight,
            ft_cfg.human_anchor_ratio,
            ft_cfg.max_advantage_weight,
        )
        human_loader = DataLoader(
            human_dataset,
            batch_size=ft_cfg.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=ft_cfg.num_workers,
            seed=ft_cfg.seed + 2,
        )
        online_policy_loader = (
            DataLoader(
                online_policy_dataset,
                batch_size=ft_cfg.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=ft_cfg.num_workers,
                seed=ft_cfg.seed + 3,
            )
            if len(online_policy_dataset) > 0
            else None
        )
        running_std = RunningAdvantageStd()

        for epoch in range(ft_cfg.joint_epochs):
            t0 = time.time()
            epoch_metrics: dict[str, list[Array]] = {}
            human_iterator = iter(human_loader)
            online_policy_iterator = (
                iter(online_policy_loader) if online_policy_loader is not None else None
            )
            replay_batches = mixed_replay_batches(
                online_dataset,
                bootstrap_dataset,
                ft_cfg.batch_size,
                ft_cfg.bootstrap_anchor_ratio,
                ft_cfg.num_workers,
                ft_cfg.seed + epoch,
            )
            for replay_batch in replay_batches:
                human_batch, human_iterator = _next_cycling(
                    human_loader, human_iterator
                )
                has_online = online_policy_loader is not None
                if has_online:
                    online_policy_batch, online_policy_iterator = _next_cycling(
                        online_policy_loader, online_policy_iterator
                    )
                else:
                    online_policy_batch = human_batch
                slice_rng, slice_key = jax.random.split(slice_rng)
                models, opt_state, total_loss, aux = online_step(
                    models,
                    opt_state,
                    replay_batch,
                    online_policy_batch,
                    human_batch,
                    running_std.value,
                    float(has_online),
                    slice_key,
                )
                (
                    jepa_loss,
                    value_loss,
                    online_policy_loss,
                    human_policy_loss,
                    policy_loss,
                    advantage_stats,
                ) = aux
                running_std = running_std.update(
                    *(float(stat) for stat in advantage_stats)
                )
                metrics = {
                    "total_loss": total_loss,
                    "jepa_loss": jepa_loss,
                    "value_loss": value_loss,
                    "online_policy_loss": online_policy_loss,
                    "human_policy_loss": human_policy_loss,
                    "policy_loss": policy_loss,
                }
                for name, value in metrics.items():
                    epoch_metrics.setdefault(name, []).append(value)
                global_step += 1
                if global_step % ft_cfg.log_every == 0:
                    wandb.log(
                        {
                            f"online/step_{name}": float(value)
                            for name, value in metrics.items()
                        }
                        | {"online/running_advantage_std": running_std.value},
                        step=global_step,
                    )
            means = {
                name: float(jnp.mean(jnp.stack(values)))
                for name, values in epoch_metrics.items()
            }
            logging.info(
                "[Online Epoch %s/%s] Total %.6f | JEPA %.6f | Value %.6f | "
                "Policy %.6f | qualifying laps %s (%.1fs)",
                epoch + 1,
                ft_cfg.joint_epochs,
                means["total_loss"],
                means["jepa_loss"],
                means["value_loss"],
                means["policy_loss"],
                len(qualifying_ids),
                time.time() - t0,
            )
            wandb.log(
                {f"online/epoch_{name}": value for name, value in means.items()},
                step=global_step,
            )

        save_checkpoint(output_dir / "ft_model_latest.eqx", (models[0], models[1]))
        save_checkpoint(output_dir / "ft_value_head_latest.eqx", models[2])
        save_checkpoint(output_dir / "ft_policy_head_latest.eqx", models[3])
    finally:
        if manage_wandb:
            wandb.finish()
        for dataset in (
            online_dataset,
            bootstrap_dataset,
            online_policy_dataset,
            human_dataset,
        ):
            dataset._pool.cache.clear()
            dataset._pool.refcounts.clear()

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
        "--policy-head",
        type=Path,
        required=True,
        help="Existing policy-head checkpoint",
    )
    parser.add_argument(
        "--incumbent-median-frames",
        type=float,
        default=None,
        help="Median successful incumbent duration used for online policy filtering",
    )
    parser.add_argument(
        "--bootstrap-data-dir",
        type=Path,
        default=DEFAULT_BOOTSTRAP_DATA_DIR,
    )
    parser.add_argument(
        "--ssl-human-dir",
        type=Path,
        default=DEFAULT_SSL_HUMAN_DIR,
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
        policy_head_path=args.policy_head,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        incumbent_median_frames=args.incumbent_median_frames,
        bootstrap_data_dir=args.bootstrap_data_dir,
        ssl_human_dir=args.ssl_human_dir,
    )


if __name__ == "__main__":
    main()
