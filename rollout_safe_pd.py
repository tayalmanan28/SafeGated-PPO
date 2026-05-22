"""
Rollout trajectories from (0, 1.5) to goal (0, -1.5) using:
  - PD controller toward goal as the reference controller
  - Switch to learned safe policy when Q_c(s, a_PD) < 0

Usage:
  python rollout_safe_pd.py --ckpt checkpoints/line_segment_brt/point_mass_brt_final.pth
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from point_mass_brt import (
    DoubleIntegrator2DEnv, GaussianPolicy, TwinQCritic
)


def pd_controller(state, goal, kp=2.0, kd=2.0):
    """Simple PD controller: a = kp*(goal - pos) - kd*vel"""
    pos = state[:2]
    vel = state[2:]
    a = kp * (goal - pos) - kd * vel
    return np.clip(a, -1.0, 1.0)


def rollout(env, critic, actor, device, goal, start, n_steps=400,
            kp=2.0, kd=2.0, threshold=0.0):
    """
    Roll out one trajectory with safety switching.
    Returns trajectory dict with positions, actions, which controller was active.
    """
    state = np.array(start, dtype=np.float32)
    env.state = state.copy()
    env.step_count = 0

    traj = {"pos": [state[:2].copy()], "vel": [state[2:].copy()],
            "actions": [], "controller": [], "safety_vals": [], "g_vals": []}

    for t in range(n_steps):
        # PD action
        a_pd = pd_controller(state, goal, kp=kp, kd=kd)

        # Evaluate Q_c for the PD action
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_t = torch.tensor(a_pd, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            qc_pd = critic.min_q(s_t, a_t).item()

        # Switch logic
        if qc_pd < threshold:
            # Unsafe! Use learned safe policy
            with torch.no_grad():
                a_safe = actor.get_dist(s_t).mean.squeeze(0).cpu().numpy()
            action = a_safe
            ctrl_label = "safe"
        else:
            action = a_pd
            ctrl_label = "pd"

        # Step
        next_state, g, done, info = env.step(action)

        traj["actions"].append(action.copy())
        traj["controller"].append(ctrl_label)
        traj["safety_vals"].append(qc_pd)
        traj["g_vals"].append(g)

        state = next_state
        traj["pos"].append(state[:2].copy())
        traj["vel"].append(state[2:].copy())

        # Check if reached goal
        if np.linalg.norm(state[:2] - goal) < 0.1 and np.linalg.norm(state[2:]) < 0.2:
            break

    traj["pos"] = np.array(traj["pos"])
    traj["vel"] = np.array(traj["vel"])
    traj["actions"] = np.array(traj["actions"])
    traj["safety_vals"] = np.array(traj["safety_vals"])
    traj["g_vals"] = np.array(traj["g_vals"])
    return traj


def rollout_pd_only(env, goal, start, n_steps=400, kp=2.0, kd=2.0):
    """Roll out with pure PD (no safety switching) for comparison."""
    state = np.array(start, dtype=np.float32)
    env.state = state.copy()
    env.step_count = 0

    traj = {"pos": [state[:2].copy()], "g_vals": []}

    for t in range(n_steps):
        a_pd = pd_controller(state, goal, kp=kp, kd=kd)
        next_state, g, done, info = env.step(a_pd)
        state = next_state
        traj["pos"].append(state[:2].copy())
        traj["g_vals"].append(g)

        if np.linalg.norm(state[:2] - goal) < 0.1 and np.linalg.norm(state[2:]) < 0.2:
            break

    traj["pos"] = np.array(traj["pos"])
    traj["g_vals"] = np.array(traj["g_vals"])
    return traj


def visualize_rollouts(trajs, pd_traj, goal, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: Trajectory plot ---
    ax = axes[0]

    # Obstacle
    circle = patches.Circle((0, 0), 0.5, fill=True, color="red", alpha=0.3)
    ax.add_patch(circle)
    # Anchor
    ax.plot(-2.0, 0.0, "bs", markersize=10, label="Anchor (-2,0)", zorder=5)
    # Start / Goal
    ax.plot(0, 1.5, "go", markersize=12, label="Start (0, 1.5)", zorder=5)
    ax.plot(0, -1.5, "r*", markersize=15, label="Goal (0, -1.5)", zorder=5)

    # Draw tangent lines from anchor to obstacle (shadow boundary)
    # Tangent from (-2,0) to circle at origin r=0.5:
    # angle = arcsin(0.5 / 2) = arcsin(0.25)
    ang = np.arcsin(0.5 / 2.0)
    base_angle = 0.0  # angle of line from anchor to center
    for sign in [1, -1]:
        theta = base_angle + sign * ang
        # Line from (-2, 0) in direction theta, extend to x=2
        dx = np.cos(theta)
        dy = np.sin(theta)
        t_end = 4.0 / dx if dx > 0 else 4.0
        ax.plot([-2, -2 + t_end * dx], [0, t_end * dy],
                "k--", alpha=0.3, linewidth=1)

    # PD-only trajectory
    ax.plot(pd_traj["pos"][:, 0], pd_traj["pos"][:, 1],
            "k-", alpha=0.4, linewidth=1.5, label="PD only (unsafe)")

    # Safe trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajs)))
    for i, traj in enumerate(trajs):
        pos = traj["pos"]
        ctrl = traj["controller"]

        # Color segments by controller
        for t in range(len(ctrl)):
            color = "blue" if ctrl[t] == "pd" else "orange"
            ax.plot(pos[t:t+2, 0], pos[t:t+2, 1], color=color,
                    linewidth=2, alpha=0.8)

        # Mark start/end
        ax.plot(pos[-1, 0], pos[-1, 1], "kx", markersize=8)

    # Legend patches
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="blue", linewidth=2, label="PD active"),
        Line2D([0], [0], color="orange", linewidth=2, label="Safe policy active"),
        Line2D([0], [0], color="black", linewidth=1.5, alpha=0.4, label="PD only (no safety)"),
        patches.Patch(facecolor="red", alpha=0.3, label="Obstacle (r=0.5)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectories: Start (0, 1.5) → Goal (0, -1.5)")
    ax.grid(True, alpha=0.3)

    # --- Right: Safety value over time ---
    ax2 = axes[1]
    ax2.axhline(0, color="red", linestyle="--", linewidth=1, label="Q_c = 0 threshold")

    # PD only g values
    ax2.plot(pd_traj["g_vals"], "k-", alpha=0.4, linewidth=1.5, label="PD only: g(s)")

    for i, traj in enumerate(trajs):
        t_arr = np.arange(len(traj["safety_vals"]))
        ax2.plot(t_arr, traj["safety_vals"], "b-", alpha=0.6, linewidth=1,
                 label="Q_c(s, a_PD)" if i == 0 else None)
        ax2.plot(t_arr, traj["g_vals"], "g-", alpha=0.6, linewidth=1,
                 label="g(s) (actual margin)" if i == 0 else None)

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Value")
    ax2.set_title("Safety value & margin over time")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Safety-Gated PD Control: Line-Segment BRT\n"
                 "Switches to safe policy when Q_c(s, a_PD) < 0", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved rollout visualization to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/line_segment_brt/point_mass_brt_final.pth")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_rollouts", type=int, default=5)
    p.add_argument("--n_steps", type=int, default=400)
    p.add_argument("--kp", type=float, default=2.0)
    p.add_argument("--kd", type=float, default=2.0)
    p.add_argument("--save_path", default="checkpoints/line_segment_brt/rollouts.png")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    S, A = 4, 2
    critic = TwinQCritic(S, A, hidden=256).to(device)
    actor = GaussianPolicy(S, A, hidden=256).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    critic.load_state_dict(ckpt["critic"])
    actor.load_state_dict(ckpt["actor"])
    critic.eval()
    actor.eval()
    print(f"Loaded checkpoint from {args.ckpt}")

    goal = np.array([0.0, -1.5])
    start = [0.0, 1.5, 0.0, 0.0]  # [x, y, vx, vy]

    # Rollouts with safety switching
    trajs = []
    for i in range(args.n_rollouts):
        env = DoubleIntegrator2DEnv(dt=0.05, max_steps=args.n_steps, seed=100 + i)
        # Add small perturbation to start for variety
        perturbed_start = start.copy()
        perturbed_start[0] += np.random.uniform(-0.3, 0.3)
        perturbed_start[1] += np.random.uniform(-0.1, 0.1)
        traj = rollout(env, critic, actor, device, goal, perturbed_start,
                       n_steps=args.n_steps, kp=args.kp, kd=args.kd)
        trajs.append(traj)
        violated = np.any(traj["g_vals"] < 0)
        print(f"  Traj {i}: steps={len(traj['actions'])}, "
              f"safe_switches={sum(1 for c in traj['controller'] if c == 'safe')}, "
              f"g_min={min(traj['g_vals']):.3f}, violated={violated}")

    # PD-only baseline (no safety)
    env_pd = DoubleIntegrator2DEnv(dt=0.05, max_steps=args.n_steps, seed=42)
    pd_traj = rollout_pd_only(env_pd, goal, start, n_steps=args.n_steps,
                              kp=args.kp, kd=args.kd)
    pd_violated = np.any(np.array(pd_traj["g_vals"]) < 0)
    print(f"  PD only: steps={len(pd_traj['g_vals'])}, "
          f"g_min={min(pd_traj['g_vals']):.3f}, violated={pd_violated}")

    visualize_rollouts(trajs, pd_traj, goal, args.save_path)


if __name__ == "__main__":
    main()
