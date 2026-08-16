"""Policy behavior-cloning data selection, losses, metrics, and training."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence, Union, cast

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from core.actions import GAS_BRAKE_VALUES_JAX, NUM_GAS_BRAKE
from core.config import IMG_HIST_LEN, PolicyHeadConfig
from core.dynamics import MLPPolicyHead
from core.encoders import ConvEncoder, LidarEncoder, ViTEncoder
from core.interfaces import Encoder as EncoderProtocol
from jaxtyping import Array, Float

import wandb
from train.config import PolicySFTConfig
from train.dataloader import DataLoader, SlidingWindowDataset
from train.pretrain import save_checkpoint

ConcreteEncoder = Union[ViTEncoder, ConvEncoder, LidarEncoder]
Batch = Mapping[str, Union[np.ndarray, Array]]
EXCLUDED_TERMINATIONS = frozenset({"manual", "respawn", "stuck", "truncated"})


class RawPolicyHead(Protocol):
    def raw_logits(
        self, latent_state: Float[Array, "latent_dim"]
    ) -> Float[Array, "actions"]: ...


@dataclass(frozen=True)
class PolicyEpisode:
    file_path: Path
    episode_id: int
    frame_start: int
    frame_end: int
    source: str
    termination: str

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start

    @property
    def dataset_key(self) -> tuple[Path, int]:
        return self.file_path, self.episode_id


def _attr_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_policy_episode(
    file_path: Path,
    episode_name: str,
    group: h5py.Group,
    *,
    is_bootstrap: bool,
    min_frames: int,
    max_frames: int,
    rewards: h5py.Dataset | None,
    episode_ids: np.ndarray,
) -> PolicyEpisode | None:
    episode_id_text = episode_name.removeprefix("episode_")
    if not episode_id_text.isdigit():
        return None
    attrs = group.attrs
    if not {"frame_start", "frame_end", "source", "termination"}.issubset(attrs):
        return None
    frame_start = int(attrs["frame_start"])  # pyright: ignore[reportArgumentType]
    frame_end = int(attrs["frame_end"])  # pyright: ignore[reportArgumentType]
    episode_id = int(episode_id_text)
    if (
        frame_start < 0
        or frame_end <= frame_start
        or frame_end > len(episode_ids)
        or not np.all(episode_ids[frame_start:frame_end] == episode_id)
    ):
        return None
    source = _attr_text(attrs["source"]).lower()
    termination = _attr_text(attrs["termination"]).lower()
    frame_count = frame_end - frame_start
    if (
        source != "human"
        or termination in EXCLUDED_TERMINATIONS
        or frame_count < min_frames
        or frame_count > max_frames
    ):
        return None
    if is_bootstrap:
        if rewards is None or frame_end > len(rewards):
            return None
        terminal_reward = float(rewards[frame_end - 1])  # pyright: ignore[reportIndexIssue]
        if not np.isfinite(terminal_reward) or terminal_reward <= 0.0:
            return None
    elif termination != "done":
        return None
    return PolicyEpisode(
        file_path=file_path,
        episode_id=episode_id,
        frame_start=frame_start,
        frame_end=frame_end,
        source=source,
        termination=termination,
    )


def _select_from_directory(
    data_dir: Path,
    *,
    is_bootstrap: bool,
    min_frames: int,
    max_frames: int,
) -> list[PolicyEpisode]:
    selected: list[PolicyEpisode] = []
    if not data_dir.exists():
        return selected
    for file_path in sorted(data_dir.rglob("*.h5")):
        try:
            with h5py.File(file_path, "r") as file:
                if "metadata" not in file:
                    continue
                if "episode_id" not in file:
                    continue
                episode_ids = np.asarray(file["episode_id"], dtype=np.int32)
                metadata = cast(h5py.Group, file["metadata"])
                rewards = (
                    cast(h5py.Dataset, file["rewards"]) if "rewards" in file else None
                )
                for episode_name in metadata.keys():
                    if not episode_name.startswith("episode_"):
                        continue
                    episode = _read_policy_episode(
                        file_path,
                        episode_name,
                        cast(h5py.Group, metadata[episode_name]),
                        is_bootstrap=is_bootstrap,
                        min_frames=min_frames,
                        max_frames=max_frames,
                        rewards=rewards,
                        episode_ids=episode_ids,
                    )
                    if episode is not None:
                        selected.append(episode)
        except (OSError, KeyError, TypeError, ValueError, IndexError, OverflowError):
            continue
    return selected


def select_policy_sft_episodes(
    bootstrap_dir: str | Path,
    ssl_human_dir: str | Path,
    min_frames: int = 420,
    max_frames: int = 540,
) -> tuple[PolicyEpisode, ...]:
    """Select successful human laps using source-specific success criteria."""
    bootstrap = _select_from_directory(
        Path(bootstrap_dir),
        is_bootstrap=True,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    ssl_human = _select_from_directory(
        Path(ssl_human_dir),
        is_bootstrap=False,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    return tuple(bootstrap + ssl_human)


def make_policy_dataset(
    episodes: Sequence[PolicyEpisode],
    *,
    obs_type: str,
    rollout_len: int,
    max_cache_bytes: int,
) -> SlidingWindowDataset:
    data_files = tuple(dict.fromkeys(episode.file_path for episode in episodes))
    data_root = data_files[0].parent if data_files else Path(".")
    return SlidingWindowDataset(
        data_dir=data_root,
        history_len=IMG_HIST_LEN,
        rollout_len=rollout_len,
        discretize_actions=True,
        obs_type=obs_type,
        load_rewards=True,
        max_cache_bytes=max_cache_bytes,
        episode_keys=[episode.dataset_key for episode in episodes],
        data_files=data_files,
    )


def _policy_observation(encoder: EncoderProtocol, batch: Batch) -> dict[str, Array]:
    if isinstance(encoder, LidarEncoder):
        return {
            "lidar": jnp.asarray(batch["obs_stack_t"], dtype=jnp.float32),
            "telemetry": jnp.asarray(batch["telemetry_t"], dtype=jnp.float32),
        }
    return {
        "screen": jnp.asarray(batch["obs_stack_t"], dtype=jnp.float32) / 255.0,
        "telemetry": jnp.asarray(batch["telemetry_t"], dtype=jnp.float32),
    }


def policy_logits(
    policy_head: RawPolicyHead, encoder: EncoderProtocol, batch: Batch
) -> Float[Array, "batch actions"]:
    latents = jax.lax.stop_gradient(
        jax.vmap(encoder)(_policy_observation(encoder, batch))
    )
    return jax.vmap(policy_head.raw_logits)(latents)


def policy_nll(
    policy_head: RawPolicyHead, encoder: EncoderProtocol, batch: Batch
) -> Float[Array, ""]:
    """Compute categorical NLL for the first action aligned to each current state."""
    logits = policy_logits(policy_head, encoder, batch)
    targets = jnp.asarray(batch["actions_seq"][:, 0], dtype=jnp.int32)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))


def policy_batch_metrics(
    policy_head: RawPolicyHead, encoder: EncoderProtocol, batch: Batch
) -> dict[str, Float[Array, ""]]:
    logits = policy_logits(policy_head, encoder, batch)
    targets = jnp.asarray(batch["actions_seq"][:, 0], dtype=jnp.int32)
    predictions = jnp.argmax(logits, axis=-1)
    top_three = jax.lax.top_k(logits, 3)[1]
    probabilities = jax.nn.softmax(logits, axis=-1)
    target_controls = GAS_BRAKE_VALUES_JAX[targets % NUM_GAS_BRAKE]
    predicted_controls = GAS_BRAKE_VALUES_JAX[predictions % NUM_GAS_BRAKE]
    return {
        "nll": jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        ),
        "top1_accuracy": jnp.mean(predictions == targets),
        "top3_accuracy": jnp.mean(jnp.any(top_three == targets[:, None], axis=1)),
        "entropy": jnp.mean(
            -jnp.sum(probabilities * jax.nn.log_softmax(logits), axis=-1)
        ),
        "steer_accuracy": jnp.mean(
            predictions // NUM_GAS_BRAKE == targets // NUM_GAS_BRAKE
        ),
        "gas_accuracy": jnp.mean(predicted_controls[:, 0] == target_controls[:, 0]),
        "brake_accuracy": jnp.mean(predicted_controls[:, 1] == target_controls[:, 1]),
    }


def make_policy_sft_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def step(
        policy_head: MLPPolicyHead,
        encoder: EncoderProtocol,
        opt_state: optax.OptState,
        batch: Batch,
    ) -> tuple[MLPPolicyHead, optax.OptState, Float[Array, ""]]:
        loss, gradients = eqx.filter_value_and_grad(policy_nll)(
            policy_head, encoder, batch
        )
        updates, opt_state = optimizer.update(
            gradients, opt_state, eqx.filter(policy_head, eqx.is_array)
        )
        return eqx.apply_updates(policy_head, updates), opt_state, loss

    return step


def train_policy_sft(
    encoder: ConcreteEncoder,
    dataset: SlidingWindowDataset,
    cfg: PolicySFTConfig,
    policy_cfg: PolicyHeadConfig,
    key: Array,
    checkpoint_path: Path,
) -> MLPPolicyHead:
    """Train the policy head and retain the lowest epoch training NLL.

    Arguments:
      encoder: Frozen observation encoder used to produce policy latents
      dataset: Successful human transition dataset
      cfg: Policy SFT optimization and logging configuration
      policy_cfg: Policy-head architecture configuration
      key: PRNG key used to initialize the policy head
      checkpoint_path: Destination for the best policy-head checkpoint

    Returns:
      The policy head restored from the lowest-NLL checkpoint
    """
    if len(dataset) == 0:
        raise RuntimeError("No successful human transitions available for policy SFT")
    policy_head = MLPPolicyHead(policy_cfg, key=key)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        seed=0,
    )
    batches_per_epoch = len(dataloader)
    total_steps = max(1, cfg.epochs * batches_per_epoch)
    warmup_steps = int(total_steps * 0.10)
    decay_steps = max(1, total_steps - warmup_steps)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr_max,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=cfg.lr_min,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule),
    )
    opt_state = optimizer.init(eqx.filter(policy_head, eqx.is_array))
    step = make_policy_sft_step(optimizer)
    best_nll = float("inf")
    global_step = 0
    for epoch in range(cfg.epochs):
        start = time.time()
        totals: dict[str, list[Array]] = {}
        for batch in dataloader:
            policy_head, opt_state, _ = step(policy_head, encoder, opt_state, batch)
            metrics = policy_batch_metrics(policy_head, encoder, batch)
            for name, value in metrics.items():
                totals.setdefault(name, []).append(value)
            global_step += 1
            if global_step == 1 or global_step % cfg.log_every == 0:
                current_lr = float(np.asarray(lr_schedule(global_step)))
                wandb.log(
                    {
                        f"policy_sft/step_{name}": float(value)
                        for name, value in metrics.items()
                    }
                    | {"policy_sft/learning_rate": current_lr},
                    step=global_step,
                )
        means = {
            name: float(jnp.mean(jnp.stack(values))) for name, values in totals.items()
        }
        if means["nll"] < best_nll:
            best_nll = means["nll"]
            save_checkpoint(checkpoint_path, policy_head)
        logging.info(
            "[Policy SFT %s/%s] NLL %.6f | top-1 %.3f | top-3 %.3f | "
            "entropy %.3f | steer %.3f | gas %.3f | brake %.3f (%.1fs)",
            epoch + 1,
            cfg.epochs,
            means["nll"],
            means["top1_accuracy"],
            means["top3_accuracy"],
            means["entropy"],
            means["steer_accuracy"],
            means["gas_accuracy"],
            means["brake_accuracy"],
            time.time() - start,
        )
        wandb.log({f"policy_sft/epoch_{name}": value for name, value in means.items()})
    return eqx.tree_deserialise_leaves(checkpoint_path, policy_head)
