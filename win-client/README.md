# Trackmania Windows Client (`win-client`)

Windows runtime and data collection interface for JEPAMania using Openplanet and `rtgym`.

## Entry Points

### 1. Data Recording (`record.py`)
Record session HDF5 shards from human play or a pretrained SAC agent:

```bash
# Record human gameplay (keyboard/gamepad)
uv run --package win-client python record.py --mode human

# Record pretrained SAC RL policy rollouts
uv run --package win-client python record.py --mode agent
```

### 2. Autonomous MPC Speedrunning (`run.py`)
Deploy the pretrained Sub-JEPA latent world model to drive autonomously in real time:

```bash
uv run --package win-client python run.py \
  --checkpoint-path checkpoints/subjepa_best.eqx \
  --encoder-type vit \
  --planner-type cem
```

#### Optional CLI Flags for `run.py`:
- `--checkpoint-path`: Path to pretrained Sub-JEPA `.eqx` checkpoint.
- `--value-head-path`: Path to pretrained `MLPValueHead` `.eqx` checkpoint.
- `--encoder-type`: Choose from `vit`, `conv`, or `lidar` (default: value set in `settings.yaml`).
- `--planner-type`: Choose from `cem`, `beam`, or `random` (default: value set in `settings.yaml`).
- `--record-rollouts`: Record real-time MPC rollouts to HDF5 shards for iterative pretraining.
