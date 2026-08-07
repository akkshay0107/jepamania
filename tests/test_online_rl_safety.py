# ruff: noqa: E402

import sys
from pathlib import Path
from typing import cast

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from train.config import FinetuneConfig
from train.dataloader import SlidingWindowDataset
from train.finetune import (
    RLModels,
    _compute_truncated_lambda_return,
    advantage_weights,
    compute_online_policy_loss,
)
from win_client import env_patches

from rl_loop import (
    EvaluationResult,
    _validate_rollout_files,
    parse_args,
    should_promote_checkpoint,
)


def _write_rollout(path: Path, frame_count: int = 8) -> None:
    with h5py.File(path, "w") as file:
        file.create_dataset("episode_id", data=np.zeros(frame_count, dtype=np.int32))
        file.create_dataset(
            "actions",
            data=np.arange(frame_count * 3, dtype=np.float32).reshape(frame_count, 3),
        )
        file.create_dataset(
            "rewards", data=np.arange(1, frame_count + 1, dtype=np.float32)
        )
        file.create_dataset(
            "observations/screen",
            data=np.arange(frame_count * 4, dtype=np.uint8).reshape(frame_count, 2, 2),
        )
        file.create_dataset(
            "observations/telemetry",
            data=np.arange(frame_count * 2, dtype=np.float32).reshape(frame_count, 2),
        )


def test_terminal_windows_keep_reward_but_disable_bootstrap(tmp_path: Path) -> None:
    rollout_path = tmp_path / "rollout.h5"
    _write_rollout(rollout_path)

    dataset = SlidingWindowDataset(
        rollout_path,
        history_len=2,
        rollout_len=5,
        discretize_actions=False,
        obs_type="screen",
        load_rewards=True,
        include_terminal_windows=True,
    )
    terminal_sample = dataset[-1]

    np.testing.assert_array_equal(
        terminal_sample["mask_seq"], np.array([1, 1, 1, 1, 0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        terminal_sample["target_mask_seq"],
        np.array([1, 1, 1, 1, 0], dtype=np.float32),
    )
    assert terminal_sample["rewards_seq"][-1] == 8.0
    assert terminal_sample["actions_seq"].shape == (5, 3)

    pretraining_dataset = SlidingWindowDataset(
        rollout_path,
        history_len=2,
        rollout_len=5,
        discretize_actions=True,
        obs_type="screen",
    )
    assert len(dataset) == 4
    assert len(pretraining_dataset) == 3


def test_lambda_target_wrapper_blocks_future_value_gradients() -> None:
    rewards = jnp.zeros((1, 3))
    masks = jnp.ones((1, 3))

    def target(values: jax.Array) -> jax.Array:
        return jax.lax.stop_gradient(
            _compute_truncated_lambda_return(rewards, values, masks, 0.99, 0.95)
        )

    values = jnp.array([[1.0, 2.0, 3.0]])
    gradients = jax.grad(lambda value: jnp.sum(target(value)))(values)
    np.testing.assert_allclose(gradients, 0.0)


class _FakeEncoder:
    def __call__(self, observations):
        return jnp.asarray([observations["telemetry"][0]])


class _FakePredictor:
    def __call__(self, latent, _action):
        return latent


class _ZeroValue:
    def __call__(self, _latent):
        return jnp.asarray(0.0)


class _TelemetryPolicy:
    def raw_logits(self, latent):
        return jnp.array([0.0, -1.0 - latent[0]])


def test_policy_loss_uses_total_positive_weight_for_normalization() -> None:
    models = cast(
        RLModels, (_FakeEncoder(), _FakePredictor(), _ZeroValue(), _TelemetryPolicy())
    )
    batch = {
        "obs_stack_t": np.zeros((2, 1, 1, 1), dtype=np.uint8),
        "telemetry_t": np.array([[0.0], [1.0]], dtype=np.float32),
        "actions_seq": np.zeros((2, 1), dtype=np.int32),
        "rewards_seq": np.array([[1.0], [2.0]], dtype=np.float32),
        "mask_seq": np.zeros((2, 1), dtype=np.float32),
    }

    loss, _ = compute_online_policy_loss(models, batch, 0.99, 1.0, 20.0)
    nll = optax.softmax_cross_entropy_with_integer_labels(
        jnp.array([[0.0, -1.0], [0.0, -2.0]]), jnp.zeros(2, dtype=jnp.int32)
    )
    weights = advantage_weights(jnp.array([1.0, 2.0]), 1.0, 20.0)
    expected = jnp.sum(weights * nll) / jnp.sum(weights)
    np.testing.assert_allclose(loss, expected)


def test_rollout_files_are_separate_and_legacy_flag_maps_to_training() -> None:
    args = parse_args(["--rollout-file", "legacy.h5"])
    assert args.training_rollout_file == Path("legacy.h5")
    assert args.evaluation_rollout_file != args.training_rollout_file

    _validate_rollout_files(Path("training.h5"), Path("evaluation.h5"))
    with pytest.raises(ValueError, match="must be different"):
        _validate_rollout_files(Path("same.h5"), Path("same.h5"))


def test_promotion_requires_success_parity_and_configured_margin() -> None:
    incumbent = EvaluationResult(success_count=10, median_successful_frames=100.0)
    assert not should_promote_checkpoint(
        incumbent,
        EvaluationResult(success_count=10, median_successful_frames=96.0),
        min_improvement_frames=5,
    )
    assert should_promote_checkpoint(
        incumbent,
        EvaluationResult(success_count=10, median_successful_frames=95.0),
        min_improvement_frames=5,
    )
    assert not should_promote_checkpoint(
        incumbent,
        EvaluationResult(success_count=9, median_successful_frames=90.0),
        min_improvement_frames=5,
    )
    assert FinetuneConfig().evaluation_episodes == 15
    assert FinetuneConfig().promotion_min_improvement_frames == 5


def test_environment_patch_failure_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_patches.platform, "system", lambda: "Windows")

    def fail_configuration() -> None:
        raise ImportError("TMRL unavailable")

    monkeypatch.setattr(
        env_patches, "configure_rtgym_action_history", fail_configuration
    )
    with pytest.raises(RuntimeError, match="not applied"):
        env_patches.apply_online_rl_patches()
