# Embodied-Aki Architecture

## Overview

Embodied-Aki is a **Hierarchical-End-to-End Hybrid VLA (Vision-Language-Action) Framework** designed for general-purpose embodied AI. The architecture combines the interpretability of hierarchical decision-making with the efficiency of end-to-end learning.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Language Instructions                      │
│                  "Pick up the red block"                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM Planner (System 2)                     │
│  - Task decomposition                                        │
│  - Long-horizon planning                                     │
│  - Constraint checking                                       │
│                                                               │
│  Output: Subgoal sequence [g1, g2, ..., gn]                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              VLA Backbone (Qwen2.5-VL)                       │
│  - Visual encoder (ViT)                                      │
│  - Language model (LLM)                                      │
│  - Multi-modal fusion                                        │
│                                                               │
│  Input: Images + Instructions                                │
│  Output: Hidden states (condition features)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              World Model (RSSM)                              │
│  - Environment dynamics prediction                           │
│  - Imagination-based planning                                │
│  - State estimation                                          │
│                                                               │
│  Components:                                                 │
│  - Encoder: obs → latent                                     │
│  - RSSM: latent dynamics                                     │
│  - Decoder: latent → obs                                     │
│  - Reward predictor                                          │
│  - Value function                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Policy Head (System 1)                          │
│  Options:                                                    │
│  1. Diffusion Policy                                         │
│     - Iterative denoising                                    │
│     - Multi-modal action distribution                        │
│                                                               │
│  2. ACT (Action Chunking Transformer)                        │
│     - Autoregressive generation                              │
│     - Query-based attention                                  │
│                                                               │
│  Output: Action sequence [a1, a2, ..., aT]                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Cross-Model Reviewer                              │
│  - Safety validation                                         │
│  - Feasibility checking                                      │
│  - Consistency verification                                  │
│                                                               │
│  Verdicts: APPROVED | NEEDS_REVISION | REJECTED | UNSAFE    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Robot Execution                                 │
│  - Action execution on physical/simulated robot              │
│  - Feedback collection                                       │
│  - Closed-loop control                                       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. VLA Backbone (`vla_backbone.py`)

**Purpose**: Multi-modal feature extraction and fusion

**Architecture**:
- **Visual Encoder**: ViT-based image encoder
- **Language Model**: Qwen2.5-VL backbone
- **Action Embedder**: Maps proprioception to VLM space
- **Action Head**: Predicts action tokens from hidden states

**Key Features**:
- LoRA adapters for efficient fine-tuning
- Support for multiple VLM backbones (Qwen, InternVL)
- Flexible action chunk size configuration

**Input/Output**:
```python
Input:
  - images: [B, T, C, H, W]
  - instructions: List[str]
  - proprioception: [B, proprio_dim] (optional)

Output:
  - action_predictions: [B, chunk_size, action_dim]
  - hidden_states: [B, hidden_dim]
  - loss: scalar
```

### 2. World Model (`world_model.py`)

**Purpose**: Learn environment dynamics for imagination-based planning

**Architecture**:
- **RSSM (Recurrent State-Space Model)**:
  - Deterministic state (h_t): Long-term memory
  - Stochastic state (z_t): Uncertainty modeling
  - Transition model: p(z_t | z_{t-1}, a_{t-1})
  - Representation model: q(z_t | h_t, obs_t)

- **Encoder/Decoder**:
  - CNN-based visual encoder
  - Proprioception encoder
  - Image decoder for reconstruction

- **Predictors**:
  - Reward predictor: r_t = r(h_t, z_t)
  - Value function: V(h_t, z_t)

**Key Features**:
- Dreamer v3-inspired architecture
- KL balancing for stable training
- Imagination horizon for planning

**Input/Output**:
```python
Input:
  - images: [B, T, C, H, W]
  - actions: [B, T, action_dim]
  - proprioception: [B, T, proprio_dim]

Output:
  - reconstructed_images: [B, T, C, H, W]
  - predicted_rewards: [B, T, 1]
  - kl_loss: scalar
  - trajectory: List[state_dict]
```

### 3. Policy Head (`policy_head.py`)

**Purpose**: Generate executable action sequences

#### Option A: Diffusion Policy

**Architecture**:
- Time embedding network
- Condition encoder
- Denoising MLP
- Reverse diffusion sampler

**Training**:
```python
# Forward pass (denoising)
predicted_noise = policy(noisy_actions, timestep, condition)
loss = MSE(predicted_noise, true_noise)
```

**Inference**:
```python
# Reverse diffusion
actions = randn()
for t in reversed(range(num_steps)):
    noise = policy(actions, t, condition)
    actions = denoise(actions, noise, t)
```

#### Option B: ACT Policy

**Architecture**:
- Learnable query embeddings
- Transformer decoder
- Autoregressive generation

**Training**:
```python
# Teacher forcing
predicted_actions = policy(condition, target_actions)
loss = MSE(predicted_actions, target_actions)
```

**Inference**:
```python
# Autoregressive generation
for i in range(num_queries):
    action = policy.generate_step(condition, i)
    update_query(action)
```

### 4. LLM Planner (`planner.py`)

**Purpose**: High-level task decomposition and planning

**Architecture**:
- Qwen2.5 LLM backbone
- Structured prompt engineering
- Value function for feasibility estimation

**Planning Process**:
1. Parse task description
2. Generate subgoal sequence
3. Validate feasibility
4. Self-correction (optional)

**Output Format**:
```json
{
  "task": "original task",
  "subgoals": [
    {
      "id": 1,
      "description": "Move arm to block",
      "preconditions": ["arm_free", "block_visible"],
      "expected_outcome": "arm_near_block",
      "success_criteria": "distance < 0.05m"
    }
  ],
  "estimated_steps": 5,
  "confidence": 0.85
}
```

### 5. Cross-Model Reviewer (`reviewer.py`)

**Purpose**: Adversarial safety and feasibility validation

**Architecture**:
- Safety classifier
- Feasibility estimator
- Issue detector (multi-label)

**Review Checks**:
1. **Safety**: Collision risk, excessive force, joint limits
2. **Feasibility**: Action smoothness, physical constraints
3. **Consistency**: Semantic alignment with instructions

**Verdicts**:
- `APPROVED`: Safe and feasible
- `NEEDS_REVISION`: Minor issues detected
- `REJECTED`: Infeasible actions
- `UNSAFE`: Safety violations

**Adversarial Review**:
```python
for round in range(num_rounds):
    report = reviewer.review(actions, plan, instructions)
    if report.verdict == APPROVED:
        break
    actions = executor.revise(actions, report.suggestions)
```

## Data Flow

### Training Pipeline

```
1. Load dataset (LeRobot/RLDS format)
   ↓
2. Preprocess (images, actions, instructions)
   ↓
3. Forward pass:
   - VLA Backbone → hidden_states
   - World Model → reconstruction + KL loss
   - Policy Head → action predictions
   ↓
4. Compute losses:
   - VLA loss (MSE)
   - World model loss (reconstruction + KL)
   - Policy loss (diffusion/ACT)
   ↓
5. Reviewer penalty (if enabled)
   ↓
6. Backward pass + o


















































































































































































































































