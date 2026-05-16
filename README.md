# SafeGated-PPO: Multi-Robot Feasibility-Gated PPO

HJ-reachability-based feasibility-gated PPO for safe reach-avoid navigation across three robot platforms, using state-based policies. Implements the gated policy optimization from the [ShieldVLA](https://arxiv.org/abs/2506.XXXXX) paper (Eq. 4–7).

## Method

Per-sample gated objective (Eq. 5):

$$\mathcal{L}(\theta) = \zeta_\theta(s) \, \mathcal{L}_\text{reward} + \beta \, (1 - \zeta_\theta(s)) \, \mathcal{L}_\text{safety}$$

where:
- **Gate** $\zeta_\theta(s) = \mathbf{1}[Q_c(s, \bar{a}_\theta(s)) > \delta]$ — evaluated at the *current* policy mean (Eq. 5)
- **Safety loss** $\mathcal{L}_\text{safety} = -Q_c(s, \bar{a}_\theta(s))$ — deterministic policy gradient through Q_c (Eq. 6)
- **Adaptive** $\beta$ **via dual ascent** $\beta \leftarrow \text{clamp}(\beta + \eta_\beta(J_C - c_\max),\; 0.5,\; 15)$ (Eq. 7)
- **HJ Bellman** for online Q_c: $Q_c(s,a) \leftarrow \min\bigl[g(s),\, \gamma \min_{a'} Q_c(s', a') + (1-\gamma) g(s)\bigr]$ (Eq. 4)

## Robots

| Robot | State dim | Action dim | Description |
|-------|-----------|------------|-------------|
| `turtlebot` | 7 | 2 (v, ω) | 2D differential-drive navigation in walled room with cylindrical obstacle |
| `drone` | 12 | 4 (vx, vy, vz, yaw_rate) | 3D quadrotor navigation with spherical obstacle |
| `fr3` | 21 | 7 (joint-pos Δ) | Franka FR3 7-DOF reach-avoid with cylindrical obstacle (MuJoCo physics) |

## Results (v4, 100 eval episodes)

| Robot | Checkpoint | Success ↑ | Safety (ep) ↑ | CSC ↓ |
|-------|-----------|-----------|---------------|-------|
| **TurtleBot** | iter100 (warmup, no gate) | 100% | 30% | 7.31 |
| | **iter250 (gated)** | **94%** | **85%** | **2.67** |
| **Drone** | iter100 (warmup, no gate) | 100% | 59% | 4.24 |
| | **iter250 (gated)** | **87%** | **87%** | **1.31** |
| **FR3** | iter200 (warmup, no gate) | 19% | 68% | 3.39 |
| | **iter400 (gated)** | **74%** | **74%** | **3.02** |

- **Success**: fraction of episodes reaching the goal
- **Safety (ep)**: fraction of episodes with *zero* constraint violations
- **CSC**: Cumulative Safety Cost — mean violation steps per episode (lower = better)

## Structure

```
state_gated_ppo/
├── base_env.py          # BaseNavEnv ABC + EnvPool vectorizer
├── models.py            # StatePolicy, SafetyCritic, gated_ppo_update()
├── train.py             # Unified gated PPO training (--robot flag)
├── train_hj.py          # HJ safety critic pretraining (SAC-style)
├── turtlebot/env.py     # TurtleBot 2D nav env
├── drone/env.py         # Drone 3D nav env
├── manipulator/env.py   # 3-DOF planar manipulator env
└── manipulator/fr3_env.py  # Franka FR3 7-DOF env (MuJoCo)
```

## Quick Start

```bash
# Train with gated PPO (online Q_c, no pretraining needed)
python train.py --robot turtlebot --iterations 300 --critic_warmup 100 --save_dir checkpoints/

python train.py --robot drone --iterations 300 --critic_warmup 100 --save_dir checkpoints/

python train.py --robot fr3 --iterations 800 --critic_warmup 200 --save_dir checkpoints/
```

Key hyperparameters:
- `--safety_delta 0.0` — gate threshold (Q_c > δ → safe)
- `--coef_min 0.5 --coef_max 15.0` — β clamp range
- `--cost_limit 3.0` — target CSC for dual ascent
- `--critic_warmup N` — pure PPO warmup iterations (no gating) for Q_c to calibrate

## Gating Logic

- **Safe** ($Q_c(s, \bar{a}_\theta(s)) > 0$): standard PPO clipped surrogate (reward maximization)
- **Unsafe** ($Q_c \leq 0$): deterministic policy gradient $-Q_c(s, \bar{a}_\theta(s))$ (safety recovery)
- Per-sample gating: each sample in the minibatch is independently gated
- Adaptive $\beta \in [0.5, 15]$ via dual ascent on episode cost vs. $c_\max = 3$
- Online HJ Bellman updates for Q_c during PPO training (twin-Q with Polyak averaging)
