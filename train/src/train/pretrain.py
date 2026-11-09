"""Offline SubJEPA updates, reproducible sampling, and model checkpoints."""

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from core.actions import ACTION_GRID_NP
from core.config import SubJepaConfig, load_config
from core.dynamics import MLPPredictor
from core.encoders import ViTEncoder, load_world_model
from core.interfaces import Encoder, Predictor
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import OmegaConf

from train.config import LossConfig, TrainConfig, load_train_config
from train.dataloader import SlidingWindowDataset
from train.loss import generate_subspace_projectors, sub_jepa_components

if TYPE_CHECKING:
    from typing import Annotated as Float
    from typing import Annotated as Int
else:
    from jaxtyping import Float, Int

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "core/config.yaml"
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "train/config.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "win-client/data"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints/pretrain"

type Models = tuple[Encoder, Predictor]
type Batch = Mapping[str, np.ndarray | Array]
type Metrics = dict[str, Float[Array, ""]]


class LearnerState[ModelT, TargetT](NamedTuple):
    models: ModelT
    optimizer: optax.OptState
    targets: TargetT
    bases: Float[Array, "subspaces latent_dim subspace_dim"]
    key: PRNGKeyArray
    updates: Int[Array, ""]


def _rollout_latent_sequence(
    predictor: Predictor,
    z0: Float[Array, "state_dim"],
    actions: Float[Array, "steps 3"],
) -> Float[Array, "steps state_dim"]:
    def step(state, action):
        next_state = predictor(state, action)
        return next_state, next_state

    return jax.lax.scan(step, z0, actions)[1]


def encode(
    encoder: Encoder, frames: Array | np.ndarray, telemetry: Array | np.ndarray
) -> Float[Array, "*leading state_dim"]:
    """Encode a batch or sequence while transferring compact uint8 frames."""
    pixels = jnp.asarray(frames, dtype=jnp.float32) / 255.0
    telemetry = jnp.asarray(telemetry, dtype=jnp.float32)
    flat = jax.vmap(encoder)(
        {
            "screen": pixels.reshape(-1, *pixels.shape[-3:]),
            "telemetry": telemetry.reshape(-1, telemetry.shape[-1]),
        }
    )
    return flat.reshape(*pixels.shape[:-3], flat.shape[-1])


def encode_rollout(models: Models, batch: Batch):
    encoder, predictor = models
    roots = encode(encoder, batch["obs_stack_t"], batch["telemetry_t"])
    targets = encode(encoder, batch["obs_stack_targets"], batch["telemetry_targets"])
    actions = jnp.asarray(batch["actions_seq"][:, : targets.shape[1]], jnp.float32)
    predicted = jax.vmap(_rollout_latent_sequence, in_axes=(None, 0, 0))(
        predictor, roots, actions
    )
    return roots, predicted, targets


def ssl_components(
    predicted: Float[Array, "batch steps state_dim"],
    targets: Float[Array, "batch steps state_dim"],
    bases: Float[Array, "subspaces latent_dim subspace_dim"],
    num_slices: int,
    key: PRNGKeyArray,
    mask: np.ndarray | Array | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    return sub_jepa_components(
        predicted.reshape(-1, predicted.shape[-1]),
        targets.reshape(-1, targets.shape[-1]),
        bases,
        num_slices,
        key,
        None if mask is None else jnp.asarray(mask).reshape(-1),
    )


def compute_loss(
    models: Models,
    batch: Batch,
    subspace_projectors: Float[Array, "subspaces latent_dim subspace_dim"],
    num_slices: int,
    reg_weight: float,
    slice_key: PRNGKeyArray,
) -> Float[Array, ""]:
    _, predicted, targets = encode_rollout(models, batch)
    prediction, regularization = ssl_components(
        predicted,
        targets,
        subspace_projectors,
        num_slices,
        slice_key,
        batch.get("target_mask_seq"),
    )
    return prediction + reg_weight * regularization


def make_train_step(optimizer, subspace_projectors, num_slices, reg_weight):
    """Build the scalar SSL step used by the saved diagnostic and local checks."""

    @eqx.filter_jit
    def train_step(models, opt_state, batch, slice_key):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(
            models, batch, subspace_projectors, num_slices, reg_weight, slice_key
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(models, eqx.is_array)
        )
        return eqx.apply_updates(models, updates), opt_state, loss

    return train_step


def alpha_for(batch: Batch, reference_weight: float) -> float:
    batch_size, steps = batch["actions_seq"].shape[:2]
    return reference_weight * 120 / (batch_size * steps)


def ssl_metrics(predicted, targets, batch, bases, key, loss: LossConfig) -> Metrics:
    prediction, regularization = ssl_components(
        predicted, targets, bases, loss.num_slices, key
    )
    alpha = alpha_for(batch, loss.reg_weight)
    return {
        "ssl": prediction + alpha * regularization,
        "prediction": prediction,
        "regularization": regularization,
    }


def make_update[ModelT, TargetT](
    optimizer: optax.GradientTransformation,
    objective: Callable,
    update_targets: Callable[[ModelT, TargetT], TargetT],
):
    """Differentiate once and commit weights, optimizer and targets together."""

    @eqx.filter_jit
    def step(state: LearnerState[ModelT, TargetT], *batches: Batch):
        key, subkey = jax.random.split(state.key)
        (loss, metrics), gradients = eqx.filter_value_and_grad(objective, has_aux=True)(
            state.models, state.targets, state.bases, subkey, *batches
        )
        norm = optax.global_norm(gradients)
        params, static = eqx.partition(state.models, eqx.is_array)
        target_params, target_static = eqx.partition(state.targets, eqx.is_array)
        updates, optimizer_state = optimizer.update(gradients, state.optimizer, params)
        models = eqx.apply_updates(state.models, updates)
        accepted = (
            jnp.isfinite(loss)
            & jnp.isfinite(norm)
            & jnp.isfinite(optax.global_norm(updates))
        )

        def apply(_):
            return (
                eqx.filter(models, eqx.is_array),
                optimizer_state,
                eqx.filter(update_targets(models, state.targets), eqx.is_array),
                state.updates + 1,
            )

        def reject(_):
            return params, state.optimizer, target_params, state.updates

        params, optimizer_state, target_params, count = jax.lax.cond(
            accepted, apply, reject, operand=None
        )
        models = eqx.combine(params, static)
        targets = eqx.combine(target_params, target_static)
        return (
            LearnerState(models, optimizer_state, targets, state.bases, key, count),
            {**metrics, "loss": loss, "grad_norm": norm, "accepted": accepted},
        )

    return step


def make_optimizer(models, schedules: tuple[optax.ScalarOrSchedule, ...]):
    labels = tuple(
        jax.tree.map(lambda _: f"group_{index}", eqx.filter(model, eqx.is_array))
        for index, model in enumerate(models)
    )
    transforms = {
        f"group_{index}": optax.adamw(
            schedule,
            weight_decay=1e-4,
            mask=lambda params: jax.tree.map(lambda value: value.ndim >= 2, params),
        )
        for index, schedule in enumerate(schedules)
    }
    return optax.chain(
        optax.clip_by_global_norm(10), optax.partition(transforms, labels)
    )


def new_state(models, optimizer, targets, key, latent_dim: int, loss: LossConfig):
    key, basis_key = jax.random.split(key)
    bases = generate_subspace_projectors(
        basis_key,
        latent_dim,
        loss.num_subspaces,
        loss.subspace_dim or latent_dim // loss.num_subspaces,
    )
    return LearnerState(
        models,
        optimizer.init(eqx.filter(models, eqx.is_array)),
        targets,
        bases,
        jax.random.key_data(key),
        jnp.asarray(0, dtype=jnp.int32),
    )


def save_checkpoint(path: str | Path, tree: PyTree) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temp = Path(file.name)
    try:
        with file:
            eqx.tree_serialise_leaves(file, tree)
            file.flush()
            os.fsync(file.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load_checkpoint[T](path: str | Path, template: T) -> T:
    return eqx.tree_deserialise_leaves(Path(path), template)


def log_json(path: Path, metrics: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, allow_nan=False) + "\n")


def legacy_splits(root: Path, rollout_len: int, val_ratio: float, seed: int):
    paths = [root] if root.is_file() else sorted(root.rglob("*.h5"))
    groups: dict[str, list[Path]] = {}
    provenance = {}
    for path in paths:
        with h5py.File(path, "r") as file:
            if "episode_id" not in file or "action_ids" in file:
                continue
            metadata = cast(h5py.Group | None, file.get("metadata"))
            first = None if metadata is None else next(iter(metadata.values()), None)
            attrs = {} if first is None else first.attrs
            source = str(attrs.get("source", path.parent.name))
            groups.setdefault(source, []).append(path.resolve())
            provenance[str(path.resolve())] = {
                "source": source,
                "recorded_map_name": str(attrs.get("map_name", "unknown")),
                "recorded_map_uid": str(attrs.get("map_uid", "unknown")),
            }

    train, validation = [], []
    for paths in groups.values():
        dataset = SlidingWindowDataset(
            root, rollout_len=rollout_len, data_files=paths, view="ssl"
        )
        if len(dataset.episode_keys) >= 2 and val_ratio > 0:
            fit, held_out = dataset.split(val_ratio=val_ratio, seed=seed)
            train.append(fit)
            validation.append(held_out)
        elif len(dataset):
            train.append(dataset)
    return train, validation, provenance


def sample_legacy(
    groups: Sequence[SlidingWindowDataset], size: int, rng: np.random.Generator
) -> tuple[Batch, list[list[int]]]:
    # Rotate the remainder across groups while keeping each batch close to balanced.
    counts = np.full(len(groups), size // len(groups), dtype=np.intp)
    counts[rng.permutation(len(groups))[: size % len(groups)]] += 1
    batches, indices = [], []
    for group, count in zip(groups, counts, strict=True):
        chosen = rng.integers(len(group), size=count)
        indices.append(chosen.tolist())
        if count:
            batches.append(group.read_batch(chosen))
    return {
        name: np.concatenate([batch[name] for batch in batches]) for name in batches[0]
    }, indices


def make_ssl_objective(loss: LossConfig):
    def objective(models, _targets, bases, key, batch):
        _, predictions, actual = encode_rollout(models, batch)
        metrics = ssl_metrics(predictions, actual, batch, bases, key, loss)
        return metrics["ssl"], metrics

    return objective


def run_pretrain(args, model_cfg: SubJepaConfig, config: TrainConfig) -> None:
    """Train from episode-split legacy recordings and export E/F weights."""
    cfg = config.pretrain
    if cfg.updates <= 0 or cfg.batch_size <= 0 or cfg.log_every <= 0:
        raise ValueError("Updates, batch size and logging interval must be positive.")
    if not 0 <= cfg.val_ratio < 1:
        raise ValueError("Validation ratio must be in [0, 1).")
    train, validation, sources = legacy_splits(
        args.data_dir, cfg.rollout_len, cfg.val_ratio, cfg.seed
    )
    if not train:
        raise ValueError(f"No complete SSL windows in {args.data_dir}.")

    key, encoder_key, predictor_key = jax.random.split(jax.random.key(cfg.seed), 3)
    models = (
        ViTEncoder(model_cfg.encoder, encoder_key),
        MLPPredictor(model_cfg.predictor, predictor_key),
    )
    if args.initialize:
        models = load_world_model(
            args.initialize, encoder_key, model_cfg.encoder, model_cfg.predictor
        )
    schedules = tuple(
        optax.warmup_cosine_decay_schedule(
            0.0, lr, min(500, cfg.updates - 1), cfg.updates, end_value=lr * 0.1
        )
        for lr in (cfg.lr_enc, cfg.lr_pred)
    )
    optimizer = make_optimizer(models, schedules)
    state = new_state(
        models, optimizer, (), key, model_cfg.encoder.latent_dim, config.loss
    )
    objective = make_ssl_objective(config.loss)
    update = make_update(optimizer, objective, lambda _models, targets: targets)
    evaluate = eqx.filter_jit(objective)
    out = args.checkpoint_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run.json"
    checkpoint = out / "learner.eqx"
    manifest = json.loads(
        json.dumps(
            {
                "training": OmegaConf.to_container(config, resolve=True),
                "model": OmegaConf.to_container(model_cfg, resolve=True),
                "action_table": ACTION_GRID_NP.tolist(),
                "sources": sources,
                "train": [group.episode_keys for group in train],
                "validation": [group.episode_keys for group in validation],
            }
        )
    )
    if args.resume:
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                "Run configuration or episode split differs from the checkpoint."
            )
        state, start = load_checkpoint(checkpoint, (state, 0))
    else:
        if manifest_path.exists():
            raise FileExistsError(f"Use --resume or a new run directory: {out}")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        start = 0

    validation_batch = (
        sample_legacy(validation, cfg.batch_size, np.random.default_rng(cfg.seed))[0]
        if validation
        else None
    )
    for attempt in range(start, cfg.updates):
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, attempt]))
        batch, indices = sample_legacy(train, cfg.batch_size, rng)
        state, metrics = update(state, batch)
        report_step = (attempt + 1) % cfg.log_every == 0 or attempt + 1 == cfg.updates
        if not bool(metrics["accepted"]):
            log_json(
                out / "metrics.jsonl",
                {"attempt": attempt + 1, "rejected_batch": indices},
            )
        elif report_step:
            report = {name: float(value) for name, value in metrics.items()}
            report.update(
                {
                    "updates": int(state.updates),
                    "batch_size": cfg.batch_size,
                    "rollout_len": cfg.rollout_len,
                    "N": cfg.batch_size * cfg.rollout_len,
                    "alpha": alpha_for(batch, config.loss.reg_weight),
                }
            )
            if validation_batch is not None:
                _, val_metrics = evaluate(
                    state.models,
                    (),
                    state.bases,
                    jax.random.key(cfg.seed),
                    validation_batch,
                )
                report.update(
                    {
                        f"validation_{name}": float(value)
                        for name, value in val_metrics.items()
                    }
                )
            log_json(out / "metrics.jsonl", report)
            print(json.dumps(report), flush=True)
        if report_step:
            save_checkpoint(checkpoint, (state, attempt + 1))
    save_checkpoint(out / "world_model.eqx", state.models)
    save_checkpoint(out / "projectors.eqx", state.bases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--initialize", type=Path)
    start.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_pretrain(
        args, load_config(str(args.model_config)), load_train_config(args.config)
    )


if __name__ == "__main__":
    main()
