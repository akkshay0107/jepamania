"""Measure warmed-up encoding and CEM inference on the local driver."""

import time

import equinox as eqx
import jax
import numpy as np
from core.config import IMG_HIST_LEN, IMG_SIZE, TELEMETRY_FEATURES, load_config
from core.planners import CEMPlanner
from win_client.mpc_driver import (
    MPCDriver,
    _encode,
    make_driving_models,
    search_action,
)


def benchmark(call, iterations: int = 100):
    for _ in range(12):
        jax.block_until_ready(call())
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(call())
        elapsed.append(1000 * (time.perf_counter() - start))
    return np.percentile(elapsed, [50, 95, 99])


def main() -> None:
    config = load_config("core/config.yaml")
    init, obs_key = jax.random.split(jax.random.key(42))
    models = make_driving_models(config, init)
    encoder = models[0]
    planner = CEMPlanner(config.planner, 0.997)
    driver = MPCDriver(planner, 42)
    rng = np.random.default_rng(42)
    obs = {
        "screen": rng.integers(
            0, 256, (IMG_HIST_LEN, IMG_SIZE, IMG_SIZE), dtype=np.uint8
        ),
        "telemetry": np.zeros(TELEMETRY_FEATURES, dtype=np.float32),
    }
    device_obs = jax.device_put(obs)
    encode = eqx.filter_jit(_encode)
    plan = planner.empty_plan()
    driver.warmup(models)
    calls = {
        "device encoding": lambda: encode(encoder, device_obs),
        "device MPC + encoding": lambda: search_action(
            planner, models, device_obs, plan, obs_key
        ),
        "host MPC + transfer": lambda: driver.select(models, obs),
    }
    print(f"Backend {jax.default_backend()} | milliseconds after warmup")
    for name, call in calls.items():
        p50, p95, p99 = benchmark(call)
        print(f"{name:<24} p50={p50:6.2f} p95={p95:6.2f} p99={p99:6.2f}")


if __name__ == "__main__":
    main()
