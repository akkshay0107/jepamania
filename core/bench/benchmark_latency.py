import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
from core.config import (
    IMG_HIST_LEN,
    TELEMETRY_FEATURES,
    EncoderConfig,
    PredictorConfig,
)

from core import MLPPredictor, ViTEncoder


def benchmark(func, args, num_warmup=100, num_runs=1000):
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
    key = jax.random.PRNGKey(0)
    key_encoder, key_predictor, key_screen, key_telemetry, key_latent = (
        jax.random.split(key, 5)
    )

    enc_cfg = EncoderConfig()
    encoder_vit = ViTEncoder(enc_cfg, key_encoder)

    pred_cfg = PredictorConfig()
    predictor = MLPPredictor(pred_cfg, key_predictor)

    screen = jax.random.normal(key_screen, (IMG_HIST_LEN, 64, 64))
    telemetry = jax.random.normal(key_telemetry, (TELEMETRY_FEATURES,))

    latent_state = jax.random.normal(key_latent, (predictor.state_dim,))
    action = jnp.zeros(3, dtype=jnp.float32)

    @eqx.filter_jit
    def run_vit_encoder(screen_data, telemetry_data):
        return encoder_vit({"screen": screen_data, "telemetry": telemetry_data})

    @eqx.filter_jit
    def run_predictor(latent, action_val):
        return predictor(latent, action_val)

    print("Benchmarking ViT Encoder...")
    vit_enc_latencies = benchmark(run_vit_encoder, (screen, telemetry))

    print("Benchmarking Predictor...")
    pred_latencies = benchmark(run_predictor, (latent_state, action))

    def print_stats(name, latencies):
        print(f"\n--- {name} Latency Stats ---")
        print(f"Mean: {np.mean(latencies):.4f} ms")
        print(f"Std:  {np.std(latencies):.4f} ms")
        print(f"Min:  {np.min(latencies):.4f} ms")
        print(f"Max:  {np.max(latencies):.4f} ms")
        print(f"P95:  {np.percentile(latencies, 95):.4f} ms")
        print(f"P99:  {np.percentile(latencies, 99):.4f} ms")

    print_stats("ViT Encoder", vit_enc_latencies)
    print_stats("Predictor", pred_latencies)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(vit_enc_latencies, bins=50, color="green", alpha=0.7)
    plt.title("ViT Encoder Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(pred_latencies, bins=50, color="orange", alpha=0.7)
    plt.title("Predictor Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    out_dir = Path(__file__).parent.parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "latency_distribution.png"
    plt.savefig(output_path)
    print(f"\nSaved latency distribution plot to {output_path}")


if __name__ == "__main__":
    main()
