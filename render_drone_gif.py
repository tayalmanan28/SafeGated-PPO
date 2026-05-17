"""Render drone navigation GIF from trained policy (egocentric + isometric views)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import art3d
import matplotlib.animation as animation
from drone.env import DroneNavEnv, OBSTACLE_RADIUS, GOAL_RADIUS
from models import StatePolicy

# Load checkpoint
ckpt_path = "/data/t-manantayal/SafeGatedVLA/state_gated_ppo_v5/drone/drone_best.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
policy = StatePolicy(12, 4, hidden=256)
policy.load_state_dict(ckpt["policy"])
policy.eval()

# Run episodes until we get a successful one that avoids obstacles
env = DroneNavEnv(seed=77)
for trial in range(50):
    env.rng = np.random.RandomState(100 + trial)
    state = env.reset()
    trajectory = [env.pos.copy()]
    done = False
    total_cost = 0
    reached = False
    while not done:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, _ = policy(s_t)
            action = mean.squeeze(0).numpy()
        state, reward, done, info = env.step(action)
        trajectory.append(env.pos.copy())
        total_cost += info["cost"]
        reached = info.get("reached", False)
    if reached and total_cost == 0:
        print(f"Trial {trial}: REACHED goal safely ({len(trajectory)} steps)")
        break
else:
    print("Warning: no clean trial found, using last")

trajectory = np.array(trajectory)
goal = env.goal.copy()
obs_pos = env.obs_pos.copy()
start = trajectory[0]

print(f"Start: {start}, Goal: {goal}, Obstacle: {obs_pos}")
print(f"Steps: {len(trajectory)}, Reached: {reached}, Cost: {total_cost}")


# ── Render isometric (3D) view ──────────────────────────────────────
def render_3d_gif(traj, goal, obs_pos, filename):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_zlim(0, 2.2)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Drone Navigation (Isometric) — Step {frame}/{len(traj)-1}')

        # Draw obstacle sphere
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        xs = obs_pos[0] + OBSTACLE_RADIUS * np.outer(np.cos(u), np.sin(v))
        ys = obs_pos[1] + OBSTACLE_RADIUS * np.outer(np.sin(u), np.sin(v))
        zs = obs_pos[2] + OBSTACLE_RADIUS * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.3, color='red')

        # Draw goal sphere
        xs = goal[0] + GOAL_RADIUS * np.outer(np.cos(u), np.sin(v))
        ys = goal[1] + GOAL_RADIUS * np.outer(np.sin(u), np.sin(v))
        zs = goal[2] + GOAL_RADIUS * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.3, color='green')

        # Trajectory up to current frame
        t = traj[:frame+1]
        ax.plot(t[:, 0], t[:, 1], t[:, 2], 'b-', linewidth=2, label='Drone path')
        ax.scatter(*t[-1], c='blue', s=80, marker='^', zorder=5, label='Drone')
        ax.scatter(*traj[0], c='cyan', s=60, marker='o', label='Start')
        ax.scatter(*goal, c='green', s=80, marker='*', label='Goal')

        ax.legend(loc='upper left', fontsize=8)
        ax.view_init(elev=25, azim=-60 + frame*0.3)

    frames = list(range(0, len(traj), 2)) + [len(traj)-1]
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    ani.save(filename, writer='pillow', fps=10)
    plt.close()
    print(f"Saved: {filename}")


# ── Render egocentric (top-down following drone) view ───────────────
def render_ego_gif(traj, goal, obs_pos, filename):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        drone_pos = traj[frame]
        # Center view on drone
        view_range = 2.5
        ax.set_xlim(drone_pos[0]-view_range, drone_pos[0]+view_range)
        ax.set_ylim(drone_pos[1]-view_range, drone_pos[1]+view_range)
        ax.set_aspect('equal')
        ax.set_title(f'Drone Navigation (Egocentric Top-Down) — Step {frame}/{len(traj)-1}\nAltitude: {drone_pos[2]:.2f}m')
        ax.set_xlabel('X'); ax.set_ylabel('Y')

        # Obstacle
        obs_circle = Circle(obs_pos[:2], OBSTACLE_RADIUS, color='red', alpha=0.4, label='Obstacle')
        ax.add_patch(obs_circle)

        # Goal
        goal_circle = Circle(goal[:2], GOAL_RADIUS, color='green', alpha=0.4, label='Goal')
        ax.add_patch(goal_circle)

        # Trail
        t = traj[:frame+1]
        ax.plot(t[:, 0], t[:, 1], 'b-', linewidth=1.5, alpha=0.6)

        # Drone marker (triangle pointing in direction of motion)
        ax.scatter(drone_pos[0], drone_pos[1], c='blue', s=120, marker='^', zorder=5, label='Drone')

        # Start
        ax.scatter(traj[0, 0], traj[0, 1], c='cyan', s=80, marker='o', label='Start')

        # Distance annotations
        d_goal = np.linalg.norm(drone_pos - goal)
        d_obs = np.linalg.norm(drone_pos - obs_pos) - OBSTACLE_RADIUS
        ax.text(0.02, 0.98, f'd_goal={d_goal:.2f}  d_obs={d_obs:.2f}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    frames = list(range(0, len(traj), 2)) + [len(traj)-1]
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    ani.save(filename, writer='pillow', fps=10)
    plt.close()
    print(f"Saved: {filename}")


out_dir = "/data/t-manantayal/SafeGatedVLA/state_gated_ppo_v5/drone"
render_3d_gif(trajectory, goal, obs_pos, f"{out_dir}/drone_isometric.gif")
render_ego_gif(trajectory, goal, obs_pos, f"{out_dir}/drone_egocentric.gif")
