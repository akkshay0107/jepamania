"""Joint Sub-JEPA, reward, termination, and scalar value updates from replay."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from core.dynamics import RewardValueHead
from core.encoders import DrivingModels, ViTEncoder
from jaxtyping import Array, Bool, Float

from train.config import FinetuneConfig, LossConfig
from train.pretrain import (
    Batch,
    LearnerState,
    encode,
    encode_rollout,
    make_optimizer,
    make_update,
    ssl_metrics,
)

type TargetModels = tuple[ViTEncoder, RewardValueHead]
type OnlineState = LearnerState[DrivingModels, TargetModels]


def return_targets(
    rewards: Float[Array, "batch steps"],
    dones: Bool[Array, "batch steps"],
    tail: Float[Array, "batch"],
    gamma: float,
) -> Float[Array, "batch steps"]:
    def step(continuation, transition):
        reward, done = transition
        value = reward + jnp.where(done, 0.0, gamma * continuation)
        return value, value

    _, returns = jax.lax.scan(step, tail, (rewards.T, dones.T), reverse=True)
    return returns.T


def tail_value(targets: TargetModels, batch: Batch) -> Float[Array, "batch"]:
    encoder, head = targets
    states = encode(
        encoder, batch["obs_stack_targets"][:, -1], batch["telemetry_targets"][:, -1]
    )
    return jax.lax.stop_gradient(eqx.filter_vmap(head.state_value)(states))


def action_losses(
    rewards: Float[Array, "... actions"],
    logits: Float[Array, "... actions"],
    batch: Batch,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    actions = jnp.asarray(batch["action_ids"])[..., None]
    reward = jnp.take_along_axis(rewards, actions, axis=-1)[..., 0]
    logit = jnp.take_along_axis(logits, actions, axis=-1)[..., 0]
    done = jnp.asarray(batch["dones"], dtype=logit.dtype)
    return (
        jnp.mean((reward - batch["rewards"]) ** 2),
        jnp.mean(optax.sigmoid_binary_cross_entropy(logit, done)),
    )


def make_joint_objective(config: FinetuneConfig, loss: LossConfig):
    """Build the objective for observed transitions and recorded action rollouts.

    Arguments:
      config: Discount and joint training settings.
      loss: Sub-JEPA regularization settings.

    Returns:
      A loss function that also reports its components.
    """
    use_ssl = config.ablation != "rl-only"
    freeze_encoder = config.ablation == "frozen-encoder"
    gamma = config.gamma

    def objective(
        models: DrivingModels,
        targets: TargetModels,
        bases,
        key,
        real: Batch,
        legacy: Batch | None,
        online: Batch | None,
    ):
        encoder, predictor, head = models
        if freeze_encoder:
            weights, static = eqx.partition(encoder, eqx.is_array)
            encoder = eqx.combine(jax.tree.map(jax.lax.stop_gradient, weights), static)

        roots = encode(encoder, real["obs_stack_t"], real["telemetry_t"])
        rewards, logits, values = eqx.filter_vmap(head)(roots)
        reward, termination = action_losses(rewards, logits, real)
        target = jnp.asarray(real["rewards"]) + jnp.where(
            real["dones"], 0.0, gamma * tail_value(targets, real)
        )
        value = jnp.mean((values - target) ** 2)

        ssl = {}
        legacy_key, online_key = jax.random.split(key)
        if use_ssl and legacy is not None:
            _, predicted, actual = encode_rollout((encoder, predictor), legacy)
            ssl = ssl_metrics(predicted, actual, legacy, bases, legacy_key, loss)

        if online is not None:
            online_root, predicted, actual = encode_rollout(
                (encoder, predictor), online
            )
            if use_ssl:
                online_ssl = ssl_metrics(
                    predicted, actual, online, bases, online_key, loss
                )
                ssl = (
                    {
                        name: 0.5 * (ssl[name] + term)
                        for name, term in online_ssl.items()
                    }
                    if ssl
                    else online_ssl
                )

            states = jnp.concatenate((online_root[:, None], predicted[:, :-1]), axis=1)
            rewards, logits, values = eqx.filter_vmap(eqx.filter_vmap(head))(states)
            rollout_reward, rollout_termination = action_losses(rewards, logits, online)

            dones = jnp.asarray(online["dones"], dtype=jnp.bool_)
            tail = tail_value(targets, online)
            returns = return_targets(jnp.asarray(online["rewards"]), dones, tail, gamma)
            imagined_values = eqx.filter_vmap(eqx.filter_vmap(head.state_value))(
                jax.lax.stop_gradient(predicted)
            )
            values = jnp.concatenate((values[:, :1], imagined_values[:, :-1]), axis=1)
            value_errors = (values - returns) ** 2
            tail_errors = jnp.where(
                dones[:, -1], 0.0, (imagined_values[:, -1] - tail) ** 2
            )
            rollout_value = (value_errors.sum() + tail_errors.sum()) / (
                value_errors.size + jnp.count_nonzero(~dones[:, -1])
            )

            reward = 0.5 * (reward + rollout_reward)
            termination = 0.5 * (termination + rollout_termination)
            value = 0.5 * (value + rollout_value)

        total = reward + termination + value + ssl.get("ssl", 0.0)
        return total, {
            **ssl,
            "reward": reward,
            "termination": termination,
            "value": value,
        }

    return objective


def update_targets(models: DrivingModels, targets: TargetModels) -> TargetModels:
    encoder, _, head = models
    live = eqx.filter((encoder, head), eqx.is_array)
    old, static = eqx.partition(targets, eqx.is_array)
    averaged = jax.tree.map(
        lambda target, online: 0.995 * target + 0.005 * online, old, live
    )
    return eqx.combine(averaged, static)


def build_joint_learner(
    models: DrivingModels, config: FinetuneConfig, loss: LossConfig
):
    if config.ablation not in {"joint", "frozen-encoder", "rl-only"}:
        raise ValueError("Finetune ablation must be joint, frozen-encoder, or rl-only.")
    schedules = (
        0.0
        if config.ablation == "frozen-encoder"
        else optax.linear_schedule(0.0, config.lr_enc, 1000),
        optax.linear_schedule(0.0, config.lr_pred, 1000),
        optax.linear_schedule(0.0, config.lr_head, 1000),
    )
    optimizer = make_optimizer(models, schedules)
    objective = make_joint_objective(config, loss)
    return optimizer, make_update(optimizer, objective, update_targets)
