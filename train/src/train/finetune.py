"""Joint SubJEPA, reward, and Double DQN updates from recorded game transitions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from core.dynamics import RewardQHead
from core.encoders import DrivingModels, ViTEncoder

from train.config import FinetuneConfig, LossConfig
from train.loss import double_dqn_target, select_action, terminal_loss
from train.pretrain import (
    Batch,
    LearnerState,
    encode,
    encode_rollout,
    make_optimizer,
    make_update,
    ssl_metrics,
)

type TargetModels = tuple[ViTEncoder, RewardQHead]
type OnlineState = LearnerState[DrivingModels, TargetModels]


def read_states(head: RewardQHead, states):
    reward, q = jax.vmap(head)(states.reshape(-1, states.shape[-1]))
    shape = (*states.shape[:-1], q.shape[-1])
    return reward.reshape(shape), q.reshape(shape)


def td_targets(head, targets, live_next, batch: Batch, gamma: float):
    target_encoder, target_head = targets
    _, live_q = read_states(head, jax.lax.stop_gradient(live_next))
    target_next = encode(
        target_encoder, batch["obs_stack_targets"], batch["telemetry_targets"]
    )
    if batch["action_ids"].ndim == 1:
        target_next = target_next[:, 0]
    _, target_q = read_states(target_head, target_next)
    return double_dqn_target(
        jnp.asarray(batch["rewards"]),
        jnp.asarray(batch["dones"]),
        live_q,
        target_q,
        gamma,
    )


def reward_and_td(head, states, batch: Batch, targets):
    reward, q = read_states(head, states)
    ids = jnp.asarray(batch["action_ids"])
    reward_error = select_action(reward, ids) - batch["rewards"]
    selected_q = select_action(q, ids)
    return (
        jnp.mean(optax.huber_loss(reward_error)),
        jnp.mean(optax.huber_loss(selected_q - targets)),
        jnp.mean(selected_q),
    )


def make_joint_objective(config: FinetuneConfig, loss: LossConfig):
    use_ssl = config.ablation != "rl-only"
    freeze_encoder = config.ablation == "frozen-encoder"

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
        legacy_key, online_key, terminal_key = jax.random.split(key, 3)
        roots = encode(encoder, real["obs_stack_t"], real["telemetry_t"])
        real_next = encode(
            encoder, real["obs_stack_targets"][:, 0], real["telemetry_targets"][:, 0]
        )
        real_target = td_targets(head, targets, real_next, real, config.gamma)
        reward, td, q_mean = reward_and_td(head, roots, real, real_target)
        terminal, leakage = terminal_loss(
            predictor,
            head,
            roots,
            jnp.asarray(real["action_ids"]),
            jnp.asarray(real["dones"]),
            terminal_key,
        )

        zero = jnp.zeros((), dtype=roots.dtype)
        ssl = {
            "ssl": zero,
            "prediction": zero,
            "regularization": zero,
            "weighted_regularization": zero,
        }
        if use_ssl and legacy is not None:
            _, predicted, actual = encode_rollout((encoder, predictor), legacy)
            ssl = ssl_metrics(predicted, actual, legacy, bases, legacy_key, loss)

        metrics = {
            "reward_real": reward,
            "td_real": td,
            "q_mean": q_mean,
            "td_target_mean": jnp.mean(real_target),
        }
        if online is not None:
            online_root, predicted, actual = encode_rollout(
                (encoder, predictor), online
            )
            if use_ssl:
                online_ssl = ssl_metrics(
                    predicted, actual, online, bases, online_key, loss
                )
                ssl = (
                    online_ssl
                    if legacy is None
                    else {
                        name: 0.5 * (ssl[name] + value)
                        for name, value in online_ssl.items()
                    }
                )
            rollout_states = jnp.concatenate(
                (online_root[:, None], predicted[:, :-1]), axis=1
            )
            rollout_target = td_targets(head, targets, actual, online, config.gamma)
            rollout_reward, rollout_td, _ = reward_and_td(
                head, rollout_states, online, rollout_target
            )
            metrics.update(reward_rollout=rollout_reward, td_rollout=rollout_td)
            reward = 0.5 * (reward + rollout_reward)
            td = 0.5 * (td + rollout_td)
        total = ssl["ssl"] + reward + td + 0.1 * terminal
        return total, {
            **ssl,
            **metrics,
            "reward": reward,
            "td": td,
            "terminal": terminal,
            "terminal_leakage": leakage,
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
