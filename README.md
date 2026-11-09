# JEPAMania: Autonomous Driving using World Models

**JEPAMania** is an autonomous driving and speedrunning framework for **Trackmania**. Rather than decoding pixel reconstructions or relying on expensive frame-by-frame simulators, JEPAMania learns an action-conditioned latent geometry of track dynamics. Real-time driving is achieved by searching over candidate control sequences in latent space using fast MPC planners. The framework is built using JAX and Equinox, using the [Sub-JEPA](https://github.com/intcomp/Sub-JEPA) idea to prevent representation collapse in the latents.

---

## 1. Project Architecture & Overview

JEPAMania separates functional deep learning components into decoupled workspace modules (`core`, `train`, `win-client`) managed via `uv`.

### Core Pipeline

1. **Observation & Telemetry Capture (`win-client`)**  
   Screen frames and nine telemetry values (speed, gear, RPM, and two previous controls) are streamed from Trackmania via Openplanet and `rtgym`.
2. **Latent Encoding (`core.encoders`)**  
   A convolutional stem turns four grayscale frames into a 4×4 grid of visual tokens. A two-block, width-128 ViT processes these tokens and speed/gear/RPM telemetry. Its 128 learned values are packed with six known control-history values into a 134-value state.
3. **Action-Conditioned Latent Rollout (`core.dynamics`)**  
   A residual MLP predicts changes to the learned state from gas, brake, steering, and known history, then shifts history exactly. Sub-JEPA scores the 128 learned values using available future observations. SSL uses recorded physical controls; CEM searches the existing action grid.
4. **Real-Time MPC Planning (`core.planners`)**  
   Categorical CEM scores a short latent rollout by its predicted rewards and the best Q-value at its endpoint. The driver executes the first action and replans from the next real observation.
5. **Online Learning (`train` and `rl_loop.py`)**
   The learner continues SubJEPA alongside reward prediction and Double DQN. A shared readout predicts reward and Q for each of the 54 actions. The game uses one frozen model snapshot per episode while the learner updates separately.
---

## 2. Environment Setup & Installation

JEPAMania requires **Python 3.13+** and uses **`uv`** for workspace management. It is recommended to have Windows for recording frames from Trackmania (installable through Steam). The pretrain phase can be done on a separate host.

### Prerequisites
1. **Install `uv`**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **GPU Acceleration (Linux Host for Training)**:
   Ensure CUDA and cuDNN are installed for JAX GPU execution.
3. **Windows Host / Client (Trackmania Execution)**:
   Required for running live data recording and autonomous driving via **Openplanet** and `rtgym`.

### Workspace Initialization

Clone the repository and synchronize all workspace dependencies (`core`, `train`, `win-client`):

```bash
git clone https://github.com/akkshay0107/jepamania.git
cd jepamania

# Sync and install all packages in editable mode
uv sync
```

---

## 3. Running the Code: End-to-End Workflows

Every pipeline stage is accessible either via root orchestrator scripts or targeted package commands.

### Step 1: Data Recording (`win-client/record.py`)
Record session HDF5 shards containing observations, actions, telemetry, and rewards.

```bash
# Record human gameplay (keyboard or gamepad)
uv run --package win-client python win-client/record.py --mode human

# Record rollouts driven by a pretrained SAC agent policy
uv run --package win-client python win-client/record.py --mode agent
```

### Step 2. Pretrain the encoder and dynamics

```bash
uv run python -m train.pretrain \
  --data-dir win-client/data \
  --checkpoint-dir checkpoints/pretrain
```

Use `--initialize path/to/world_model.eqx` to warm-start E/F or `--resume` with the original configuration to resume the same run. `world_model.eqx` and `projectors.eqx` are the inputs for online learning. `learner.eqx` also saves the optimizer and random stream.

The defaults use 24 complete five-step windows per batch, balanced by legacy source. `loss.reg_weight` specifies the coefficient at 120 encoded targets. The learner scales it by the actual number of targets in each window group.

### Step 3. Collect and learn online on Windows

Enable TMRL's virtual gamepad so the game can execute the steering and throttle levels in the 54-action table.

```bash
uv run python rl_loop.py \
  --initialize checkpoints/pretrain/world_model.eqx \
  --data-dir win-client/data \
  --run-dir checkpoints/online
```

The first 4,096 decisions fill replay with held random actions. Joint updates then use recorded rewards and action IDs, with legacy and online windows sharing the SSL objective. The default reward charges 0.001 per step, adds 1 for a finish, and subtracts 1 for a genuine game failure. The 120-second limit is a collection cut, so the learner bootstraps from its final observation.

Use `--resume` in place of `--initialize` to reopen the run. Keep the `replay/` shards with the learner checkpoint. The run records model snapshots and separate evaluation summaries. `finetune.ablation` can be set to `frozen-encoder` or `rl-only` for matched trials from the same E/F initialization.

### Step 4. Drive with a frozen snapshot

```bash
uv run python win-client/run.py \
  --checkpoint-path checkpoints/online/driver.eqx \
  --controller mpc
```

The driver export includes the action table, preprocessing, model settings, and planner settings. Choose `--controller q` to evaluate greedy Q at the same cadence. Measure local inference after warmup with `uv run python core/bench/benchmark_planners.py`.

## 4. Configuration

[core/config.yaml](core/config.yaml) sets the encoder, predictor, and planner. The default CEM uses a five-step horizon, 128 candidates, four iterations, and 16 elites. Keep the planning horizon at or below the trained rollout horizon.

[train/config.yaml](train/config.yaml) sets the SSL and joint learning rates, regularizer, update and decision budgets, and evaluation frequency. The default trial evaluates every 10,000 accepted decisions.

### Windows Runtime & Client Deployment ([win-client/settings.yaml](win-client/settings.yaml))

Controls the human and SAC data recorders.

| Parameter | Default | Tuning Guidance |
| :--- | :--- | :--- |
| `agent.exploration.ou_noise_sigma`| `0.05` | Volatility of Ornstein-Uhlenbeck exploration noise during agent data recording. |
| `agent.filter.steer_deadzone` | `0.015` | Absolute threshold below which high-frequency steering jitter is zeroed out. |
| `episode_monitor.stuck_speed_kmh` | `3.0` | Speed threshold below which the car is considered stuck. |
| `episode_monitor.stuck_window_frames`| `80` | Frames below `stuck_speed_kmh` before triggering an automatic episode reset (`80` frames ≈ 4s at 20 Hz). |
| `episode_monitor.max_frames_per_episode`| `1800` | Hard reset limit per episode (`1800` frames = 1.5 minutes at 20 Hz). |

---

## 5. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
