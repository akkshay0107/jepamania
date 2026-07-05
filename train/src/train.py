"""
Offline Sub-JEPA pretraining loop.

Optimizes an (Encoder, Predictor) pair jointly: the encoder embeds the
observation at time t and at t+K, the predictor rolls the latent forward
through the K recorded action tokens, and the Sub-JEPA loss matches the
rolled-out latent against the target latent while regularizing the target
distribution to prevent collapse.
"""

import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from core.config import LossConfig
from core.interfaces import Encoder, Predictor
from core.loss import generate_projectors, sub_jepa_loss
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

# (encoder, predictor) — both are Equinox modules, so the tuple is a PyTree.
Models = Tuple[Encoder, Predictor]
# Batches cross the JIT boundary: host-side NumPy in, traced Arrays inside.
# Mapping (read-only, covariant) lets both views satisfy the same signature.
Batch = Mapping[str, Union[np.ndarray, Array]]

# Checkpoints hold either the (encoder, predictor) models or an optax state.
CheckpointT = TypeVar("CheckpointT")


def _rollout_latent(
    predictor: Predictor,
    z0: Float[Array, "latent_dim"],
    actions: Int[Array, "K"],
) -> Float[Array, "latent_dim"]:
    """Rolls a single latent forward through K action tokens."""

    def step(
        z: Float[Array, "latent_dim"], action: Int[Array, ""]
    ) -> Tuple[Float[Array, "latent_dim"], None]:
        return predictor(z, action), None

    z_final, _ = jax.lax.scan(step, z0, actions)
    return z_final


def compute_loss(
    models: Models,
    batch: Batch,
    subspace_projectors: Float[Array, "num_subspaces latent_dim subspace_dim"],
    slice_projectors: Float[Array, "num_subspaces subspace_dim num_slices"],
    reg_weight: float,
) -> Float[Array, ""]:
    encoder, predictor = models

    # uint8 -> float32 here so the cast runs on-device after the (4x smaller)
    # uint8 transfer, conserving PCIe bandwidth.
    obs_t = {
        "screen": batch["obs_stack_t"].astype(jnp.float32) / 255.0,
        "telemetry": batch["telemetry_t"],
    }
    obs_target = {
        "screen": batch["obs_stack_target"].astype(jnp.float32) / 255.0,
        "telemetry": batch["telemetry_target"],
    }
    actions = batch["actions_seq"].astype(jnp.int32)

    z_t = jax.vmap(encoder)(obs_t)
    z_target = jax.vmap(encoder)(obs_target)
    z_pred = jax.vmap(lambda z0, acts: _rollout_latent(predictor, z0, acts))(
        z_t, actions
    )

    return sub_jepa_loss(
        z_pred, z_target, subspace_projectors, slice_projectors, reg_weight
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
    for path in checkpoint_dir.glob("subjepa_epoch_*.eqx"):
        suffix = path.stem.removeprefix("subjepa_epoch_")
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
) -> Models:
    """Runs offline pretraining and returns the trained (encoder, predictor).

    Checkpoints are written to `checkpoint_dir` after every epoch as
    `subjepa_epoch_{n}.eqx` plus a rolling `subjepa_latest.eqx`, with the
    optimizer state alongside in matching `*_optstate.eqx` files and the
    frozen loss projectors in `subjepa_projectors.eqx`.

    With `resume=True`, weights, optimizer state, and projectors are restored
    from the rolling files and epoch numbering continues from the highest
    existing epoch checkpoint; `num_epochs` more epochs are then trained.
    """
    checkpoint_dir = Path(checkpoint_dir)

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

    projectors_path = checkpoint_dir / "subjepa_projectors.eqx"
    start_epoch = 0
    if resume:
        latest_path = checkpoint_dir / "subjepa_latest.eqx"
        latest_opt_path = checkpoint_dir / "subjepa_latest_optstate.eqx"
        for path in (latest_path, latest_opt_path, projectors_path):
            if not path.exists():
                raise FileNotFoundError(f"Cannot resume: missing {path}")
        models = load_checkpoint(latest_path, models)
        opt_state = load_checkpoint(latest_opt_path, opt_state)
        subspace_projectors, slice_projectors = load_checkpoint(
            projectors_path, (subspace_projectors, slice_projectors)
        )
        start_epoch = _last_epoch_index(checkpoint_dir)
        print(f"Resumed from {latest_path} at epoch {start_epoch}")
    else:
        save_checkpoint(projectors_path, (subspace_projectors, slice_projectors))

    train_step = make_train_step(
        optimizer, subspace_projectors, slice_projectors, loss_cfg.reg_weight
    )

    last_epoch = start_epoch + num_epochs
    global_step = 0
    for epoch in range(start_epoch, last_epoch):
        # Keep losses as device scalars; a float() every step would block the
        # host on each result and defeat JAX async dispatch.
        epoch_losses = []
        epoch_start = time.time()

        for batch in dataloader:
            models, opt_state, loss = train_step(models, opt_state, batch)
            epoch_losses.append(loss)
            global_step += 1

            if global_step % log_every == 0:
                print(
                    f"epoch {epoch + 1}/{last_epoch} | step {global_step} | "
                    f"loss {float(loss):.6f}"
                )

        if not epoch_losses:
            raise RuntimeError(
                "DataLoader yielded no batches; check data_dir and batch_size."
            )

        mean_loss = float(jnp.mean(jnp.stack(epoch_losses)))
        duration = time.time() - epoch_start
        print(
            f"epoch {epoch + 1}/{last_epoch} done | "
            f"mean loss {mean_loss:.6f} | "
            f"{len(epoch_losses)} batches in {duration:.1f}s"
        )

        # Weights and optimizer state live in separate files so deployment
        # ships weights-only while resume can restore the Adam moments.
        save_checkpoint(checkpoint_dir / f"subjepa_epoch_{epoch + 1}.eqx", models)
        save_checkpoint(checkpoint_dir / "subjepa_latest.eqx", models)
        save_checkpoint(
            checkpoint_dir / f"subjepa_epoch_{epoch + 1}_optstate.eqx", opt_state
        )
        save_checkpoint(checkpoint_dir / "subjepa_latest_optstate.eqx", opt_state)

    return models
