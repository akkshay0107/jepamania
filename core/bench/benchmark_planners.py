import time

import equinox as eqx
import jax
import numpy as np
from core.config import PredictorConfig, ValueHeadConfig
from core.dynamics import MLPPredictor, MLPValueHead
from core.planners import BeamSearchPlanner, CEMPlanner, RandomShootingPlanner


def benchmark(func, args, num_warmup=10, num_runs=100):
    # Warmup to ensure fully compiled XLA execution
    for _ in range(num_warmup):
        res = func(*args)
        jax.block_until_ready(res)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        res = func(*args)
        jax.block_until_ready(res)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    return np.array(latencies)


def main():
    print("Setting up models for benchmarking...")
    key = jax.random.PRNGKey(0)
    key_pred, key_val, key_plan = jax.random.split(key, 3)

    pred_cfg = PredictorConfig()
    predictor = MLPPredictor(pred_cfg, key_pred)

    val_cfg = ValueHeadConfig()
    value_head = MLPValueHead(val_cfg, key_val)

    # Pure value-guided objective: just use the predicted value
    def objective_fn(latent):
        return value_head(latent)

    latent_state = jax.random.normal(key_plan, (pred_cfg.latent_dim,))

    print(
        f"\n{'Planner':<20} | {'Seq_Len':<10} | {'Params':<30} | {'Mean (ms)':<10} |"
        f" {'Max (ms)':<10}"
    )
    print("-" * 90)

    rs_configs = [
        {"seq": 10, "samples": 100},
        {"seq": 10, "samples": 250},
        {"seq": 10, "samples": 500},
        {"seq": 10, "samples": 1000},
        {"seq": 10, "samples": 2000},
    ]

    for cfg in rs_configs:
        planner = RandomShootingPlanner(
            predictor=predictor,
            objective_fn=objective_fn,
            sequence_len=cfg["seq"],
            num_samples=cfg["samples"],
        )

        @eqx.filter_jit
        def run_rs_planner(state, p_key):
            return planner(state, key=p_key)

        latencies = benchmark(run_rs_planner, (latent_state, key_plan))
        params_str = f"samples={cfg['samples']}"
        print(
            f"{'RandomShooting':<20} | {cfg['seq']:<10} | {params_str:<30} |"
            f" {np.mean(latencies):<10.2f} | {np.max(latencies):<10.2f}"
        )

    beam_configs = [
        {"seq": 10, "width": 2},
        {"seq": 10, "width": 3},
        {"seq": 10, "width": 5},
        {"seq": 10, "width": 8},
        {"seq": 10, "width": 10},
    ]

    for cfg in beam_configs:
        planner = BeamSearchPlanner(
            predictor=predictor,
            objective_fn=objective_fn,
            sequence_len=cfg["seq"],
            beam_width=cfg["width"],
        )

        @eqx.filter_jit
        def run_beam_planner(state):
            return planner(state)

        latencies = benchmark(run_beam_planner, (latent_state,))
        params_str = f"width={cfg['width']}"
        print(
            f"{'BeamSearch':<20} | {cfg['seq']:<10} | {params_str:<30} |"
            f" {np.mean(latencies):<10.2f} | {np.max(latencies):<10.2f}"
        )

    cem_configs = [
        {"seq": 10, "iters": 2, "samples": 50},
        {"seq": 10, "iters": 3, "samples": 50},
        {"seq": 10, "iters": 3, "samples": 100},
        {"seq": 10, "iters": 3, "samples": 200},
        {"seq": 10, "iters": 5, "samples": 100},
    ]

    for cfg in cem_configs:
        planner = CEMPlanner(
            predictor=predictor,
            objective_fn=objective_fn,
            sequence_len=cfg["seq"],
            num_iters=cfg["iters"],
            num_samples=cfg["samples"],
            num_elites=max(1, cfg["samples"] // 4),
            alpha=0.1,
        )

        @eqx.filter_jit
        def run_cem_planner(state, p_key):
            return planner(state, key=p_key)

        latencies = benchmark(run_cem_planner, (latent_state, key_plan))
        params_str = f"iters={cfg['iters']}, samples={cfg['samples']}"
        print(
            f"{'CEM':<20} | {cfg['seq']:<10} | {params_str:<30} |"
            f" {np.mean(latencies):<10.2f} | {np.max(latencies):<10.2f}"
        )


if __name__ == "__main__":
    main()
