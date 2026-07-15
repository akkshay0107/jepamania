# JEPAMania: Autonomous Driving using World Models

**JEPAMania** is an autonomous driving and speedrunning framework for **Trackmania**. Rather than decoding pixel reconstructions or relying on expensive frame-by-frame simulators, JEPAMania learns an action-conditioned latent geometry of track dynamics. Real-time driving is achieved by searching over candidate control sequences in latent space using fast MPC planners. The framework is built using JAX and Equinox, using the [Sub-JEPA](https://github.com/intcomp/Sub-JEPA) idea to prevent representation collapse in the latents.

---

## 1. Project Architecture & Overview

JEPAMania separates functional deep learning components into decoupled workspace modules (`core`, `train`, `win-client`) managed via `uv`.

### Core Pipeline

1. **Observation & Telemetry Capture (`win-client`)**  
   Screen frames (RGB) or LiDAR raycasts alongside 9-float vehicle telemetry (speed, gear, RPM, previous steering/throttle actions) are streamed from Trackmania at 20 Hz via Openplanet and `rtgym`.
2. **Latent Encoding (`core.encoders`)**  
   An observation encoder (`ViTEncoder`, `ConvEncoder`, or `LidarEncoder`) compresses visual stacks and telemetry into a compact latent embedding vector $z_t \in \mathbb{R}^{192}$.
3. **Action-Conditioned Latent Rollout (`core.dynamics`)**  
   An autoregressive dynamics predictor (`MLPPredictor`) rolls forward future latent states $z_{t+k}$ across a $K$-step planning horizon given candidate action sequences without frame decoding.
4. **Real-Time MPC Planning (`core.planners`)**  
   A Model-Predictive Control planner (**Cross-Entropy Method / CEM**, **Beam Search**, or **Random Rollouts**) evaluates candidate latent trajectories against fine-tuned value heads (`MLPValueHead`) or smoothness penalties to select the optimal control action within 5 milliseconds.
5. **Cyclic Self-Improvement (`train` & `rl_loop.py`)**  
   Recorded gameplay shards feed self-supervised **Sub-JEPA Sliced-Subspace Pretraining** and downstream joint **Value/Policy Fine-Tuning** to continually improve driving performance.

### Key Architectural Highlights

- **Sliced-Subspace Regularization**:  
  To prevent representational collapse without contrastive negative pairs, Sub-JEPA projects embeddings across $M$ orthogonal subspaces and $S$ 1D slices, matching predicted latents against target latents stably.
- **Real-Time Latent Planning**:  
  Executes up to 128 parallel rollout trajectories per control tick using JAX `vmap`/`jit` for low-latency decision making.
- **Closed-Loop Cyclic RL**:  
  Interleaves online MPC rollout collection with value head bootstrapping and joint latent policy fine-tuning.

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

### Step 2: Self-Supervised Sub-JEPA Pretraining (`train/src/pretrain.py`)
Pretrain the joint `(Encoder, Predictor)` world model on recorded HDF5 shards.

```bash
# Start fresh pretraining with Vision Transformer (ViT) encoder
uv run --package train python train/src/pretrain.py \
  --encoder vit \
  --data-dir win-client/data \
  --checkpoint-dir checkpoints/pretrain

# Resume training from existing checkpoints
uv run --package train python train/src/pretrain.py \
  --encoder vit \
  --resume \
  --checkpoint-dir checkpoints/pretrain
```

### Step 3: Downstream RL & Value Head Fine-Tuning (`train/src/finetune.py`)
Fine-tune the pretrained Sub-JEPA latent model and train a discounted return value head on reward-labeled rollouts.

```bash
# Bootstrap value head on frozen encoder, then fine-tune jointly
uv run --package train python train/src/finetune.py \
  --data-dir win-client/data/rl \
  --checkpoint checkpoints/pretrain/pretrain_model_best.eqx \
  --output-dir checkpoints/finetune
```

### Step 4: Real-Time Autonomous MPC Driving (`win-client/run.py`)
Deploy the trained latent world model to drive autonomously in live Trackmania.

```bash
# Run Cross-Entropy Method (CEM) MPC driver
uv run --package win-client python win-client/run.py \
  --checkpoint-path checkpoints/finetune/ft_model_latest.eqx \
  --value-head-path checkpoints/finetune/ft_value_head_latest.eqx \
  --planner-type cem
```

*Supported `--planner-type` options:* `cem`, `beam`, `random`. Add `--record-rollouts` to simultaneously record live MPC trajectories to disk.

### Step 5: Cyclic Online RL Loop (`rl_loop.py`)
Orchestrate alternating phases of MPC rollout collection and joint model fine-tuning automatically:

```bash
uv run python rl_loop.py \
  --pretrain-checkpoint checkpoints/pretrain/pretrain_model_best.eqx \
  --bootstrap-dir win-client/data/rl/bootstrap \
  --rollouts-dir win-client/data/rl/rollouts \
  --checkpoints-dir checkpoints/rl \
  --num-iterations 5 \
  --episodes-per-iter 5
```

### Step 6: Checkpoint Export (`tar`)
Package required configs and model checkpoints into a compressed archive for deployment across machines:

```bash
# Export checkpoints and configs into a compressed tar archive
tar -czvf train_checkpoints.tar.gz checkpoints/ core/config.yaml train/config.yaml win-client/settings.yaml

# Extract checkpoint archive on target deployment host
tar -xzvf train_checkpoints.tar.gz
```

---

## 4. Tweaking Configurations

JEPAMania uses hierarchical YAML configuration files cleanly separated by concern. Rather than editing code directly, modify the parameters in the configuration files below to customize architectures, training objectives, and runtime behavior.

### Core Architecture & Planner Settings ([core/config.yaml](core/config.yaml))

Controls neural network dimensions, transformer backbones, and default latent planning parameters.

| Parameter | Default | Tuning Guidance |
| :--- | :--- | :--- |
| `encoder.latent_dim` | `192` | Latent vector size ($d$). Must match `predictor.latent_dim`. Increase to `256` or `384` for higher visual feature capacity. |
| `encoder.transformer.num_layers` | `3` | Number of ViT self-attention blocks used for spatial observation encoding. |
| `encoder.transformer.num_heads` | `4` | Number of attention heads in each ViT block. |
| `predictor.hidden_dim` | `256` | MLP hidden layer width for latent transition dynamics prediction. |
| `planner.type` | `"beam"` | Default MPC planner algorithm (`"cem"`, `"beam"`, or `"random"`). |
| `planner.sequence_len` | `10` | MPC planning horizon ($K$ steps forward into the future). |
| `planner.beam_width` | `6` | Number of concurrent top trajectories tracked during Beam Search optimization. |

---

### Training & Fine-Tuning Hyperparameters ([train/config.yaml](train/config.yaml))

Defines Sub-JEPA sliced-subspace regularizers, learning rates, schedules, and dataloader settings.

| Parameter | Default | Tuning Guidance |
| :--- | :--- | :--- |
| `loss.num_subspaces` | `16` | Number of orthogonal subspace projections ($M$) for Sub-JEPA loss calculation. |
| `loss.num_slices` | `16` | Number of 1D slices ($S$) per subspace. |
| `loss.reg_weight` | `0.5` | Variance regularization weight preventing representation collapse. Increase if latents shrink toward zero. |
| `pretrain.epochs` | `10` | Number of self-supervised pretraining epochs over recorded shards. |
| `pretrain.batch_size` | `256` | Transition batch size during pretraining. |
| `pretrain.lr` | `3.0e-4` | Peak AdamW learning rate for joint encoder/predictor pretraining. |
| `pretrain.rollout_len` | `5` | Autoregressive transition steps predicted during pretraining. |
| `finetune.warmup_epochs` | `3` | Epochs spent bootstrapping the value head on frozen encoder representations. |
| `finetune.joint_epochs` | `5` | Epochs for joint end-to-end fine-tuning of encoder, predictor, and value head. |
| `finetune.lr_enc` | `1.0e-5` | Small learning rate for backbone fine-tuning to preserve learned world dynamics. |
| `finetune.lr_val` | `3.0e-4` | Learning rate for the Huber discounted return value head. |
| `finetune.value_weight` | `0.5` | Relative loss multiplier between Sub-JEPA predictive loss and Huber value return loss. |

---

### Windows Runtime & Client Deployment ([win-client/settings.yaml](win-client/settings.yaml))

Manages real-time telemetry filtering, exploration noise, stuck reset monitoring, and live MPC execution.

| Parameter | Default | Tuning Guidance |
| :--- | :--- | :--- |
| `agent.exploration.ou_noise_sigma`| `0.05` | Volatility of Ornstein-Uhlenbeck exploration noise during agent data recording. |
| `agent.filter.steer_deadzone` | `0.015` | Absolute threshold below which high-frequency steering jitter is zeroed out. |
| `episode_monitor.stuck_speed_kmh` | `3.0` | Speed threshold below which the car is considered stuck. |
| `episode_monitor.stuck_window_frames`| `80` | Frames below `stuck_speed_kmh` before triggering an automatic episode reset (`80` frames ≈ 4s at 20 Hz). |
| `episode_monitor.max_frames_per_episode`| `1800` | Hard reset limit per episode (`1800` frames = 1.5 minutes at 20 Hz). |
| `mpc.planner_type` | `"cem"` | Online MPC algorithm (`"cem"`, `"beam"`, or `"random"`). |
| `mpc.num_samples` | `128` | Number of candidate action trajectories sampled per CEM iteration. |
| `mpc.num_iters` | `3` | Number of distribution refinement iterations in CEM. |
| `mpc.num_elites` | `16` | Top-performing trajectories averaged to update sampling distributions in CEM. |

---

## 5. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
