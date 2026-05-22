"""
Learn the Backward Reachable Tube (BRT) for a 2D point mass (double integrator)
using Hamilton-Jacobi safety critic (Fisac et al., ICRA 2019).

System: Double integrator in 2D
  State: [x, y, vx, vy]  (position + velocity)
  Control: [ax, ay]       (acceleration, bounded)
  Dynamics:
    x_dot  = vx
    y_dot  = vy
    vx_dot = ax
    vy_dot = ay

Domain: [-2, 2] x [-2, 2] (position space)
Obstacle: circle of radius 0.5 centered at origin

The safety margin g(s) = ||pos|| - 0.5  (distance to obstacle boundary).
The learned Q-critic approximates the BRT:
  V(s) < 0 => state will inevitably reach the obstacle (inside BRT)
  V(s) >= 0 => a safe control exists to avoid the obstacle

Usage:
  python point_mass_brt.py --iterations 300
  python point_mass_brt.py --iterations 300 --visualize
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions import Normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ─── Network Definitions ─────────────────────────────────────────────

def mlp(dims, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Gaussian actor for continuous control."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp([state_dim, hidden, hidden, action_dim])
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = torch.tanh(self.net(state))
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, state):
        mean, std = self(state)
        return Normal(mean, std)


class TwinQCritic(nn.Module):
    """Twin-Q safety critic for HJ value function."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        inp = state_dim + action_dim
        self.q1 = mlp([inp, hidden, hidden, 1])
        self.q2 = mlp([inp, hidden, hidden, 1])

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def min_q(self, state, action):
        q1, q2 = self(state, action)
        return torch.min(q1, q2)


# ─── Environment ─────────────────────────────────────────────────────

class DoubleIntegrator2DEnv:
    """
    2D Double Integrator (point mass) with obstacle avoidance.

    State: [x, y, vx, vy]
    Action: [ax, ay] in [-1, 1]^2

    Domain: position in [-2, 2]^2, velocity in [-2, 2]^2
    Obstacle: circle at origin, radius 0.5
    """

    def __init__(self, dt: float = 0.05, max_steps: int = 200,
                 a_max: float = 1.0, seed: int = 0):
        self.dt = dt
        self.max_steps = max_steps
        self.a_max = a_max
        self.state_dim = 4  # [x, y, vx, vy]
        self.action_dim = 2  # [ax, ay]
        self.rng = np.random.default_rng(seed)

        # Obstacle definition
        self.obs_center = np.array([0.0, 0.0])
        self.obs_radius = 0.5

        # Domain bounds
        self.pos_bound = 2.0
        self.vel_bound = 2.0

        self.state = None
        self.step_count = 0

    def safety_margin(self, state: np.ndarray) -> float:
        """
        g(s) = min distance from origin to segment [A, P] - r_obstacle.
        A = (-2, 0) anchor, P = (x, y) point mass position.
        The entire line segment A-P must clear the obstacle.
        """
        x, y = state[0], state[1]
        # Anchor point
        ax, ay = -2.0, 0.0
        # Vector from A to P
        dx, dy = x - ax, y - ay  # (x+2, y)
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-12:
            # P is at A; distance from O to A = 2.0
            return 2.0 - self.obs_radius

        # Projection parameter: t* = (O - A) . (P - A) / |P - A|^2
        # O - A = (2, 0), P - A = (x+2, y)
        t_star = (2.0 * dx + 0.0 * dy) / seg_len_sq  # = 2*(x+2) / ((x+2)^2 + y^2)

        if t_star <= 0.0:
            # Closest point is A = (-2, 0), dist to O = 2.0
            dist = 2.0
        elif t_star >= 1.0:
            # Closest point is P itself
            dist = np.sqrt(x * x + y * y)
        else:
            # Perpendicular distance from O to the line through A and P
            # = |cross(P-A, A-O)| / |P-A|
            # cross = dx * (0 - 0) - dy * (0 - 2) = 2*dy = 2*y
            dist = abs(2.0 * y) / np.sqrt(seg_len_sq)

        return dist - self.obs_radius

    def reset(self) -> np.ndarray:
        """Reset to a random state in the domain."""
        x = self.rng.uniform(-self.pos_bound, self.pos_bound)
        y = self.rng.uniform(-self.pos_bound, self.pos_bound)
        vx = self.rng.uniform(-self.vel_bound, self.vel_bound)
        vy = self.rng.uniform(-self.vel_bound, self.vel_bound)
        self.state = np.array([x, y, vx, vy], dtype=np.float32)
        self.step_count = 0
        return self.state.copy()

    def step(self, action: np.ndarray):
        """
        Euler integration of double integrator dynamics.
        Returns: (next_state, safety_margin, done, info)
        """
        action = np.clip(action, -self.a_max, self.a_max).astype(np.float32)

        x, y, vx, vy = self.state
        ax, ay = action

        # Double integrator dynamics
        x_new = x + vx * self.dt
        y_new = y + vy * self.dt
        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt

        # Clip to domain
        x_new = np.clip(x_new, -self.pos_bound, self.pos_bound)
        y_new = np.clip(y_new, -self.pos_bound, self.pos_bound)
        vx_new = np.clip(vx_new, -self.vel_bound, self.vel_bound)
        vy_new = np.clip(vy_new, -self.vel_bound, self.vel_bound)

        self.state = np.array([x_new, y_new, vx_new, vy_new], dtype=np.float32)
        self.step_count += 1

        g = self.safety_margin(self.state)
        done = (self.step_count >= self.max_steps) or (g < 0)

        info = {"safety_margin": g}
        return self.state.copy(), g, done, info


class EnvPool:
    """Vectorized environment pool for parallel data collection."""

    def __init__(self, num_envs: int, dt: float = 0.05, seed: int = 0):
        self.envs = [DoubleIntegrator2DEnv(dt=dt, seed=seed + i)
                     for i in range(num_envs)]
        self.num_envs = num_envs
        self.state_dim = 4
        self.action_dim = 2

    def reset_all(self) -> np.ndarray:
        states = np.stack([env.reset() for env in self.envs])
        return states

    def step(self, actions: np.ndarray):
        next_states = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = []

        for i, (env, a) in enumerate(zip(self.envs, actions)):
            ns, g, done, info = env.step(a)
            if done:
                ns = env.reset()
            next_states[i] = ns
            dones[i] = done
            infos.append(info)

        return next_states, dones, infos


# ─── Replay Buffer ───────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.cap = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, action_dim), dtype=np.float32)
        self.g = np.zeros(capacity, dtype=np.float32)
        self.ns = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, g, ns, d):
        i = self.ptr % self.cap
        self.s[i] = s
        self.a[i] = a
        self.g[i] = g
        self.ns[i] = ns
        self.d[i] = d
        self.ptr += 1
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.s[idx], device=device),
            torch.tensor(self.a[idx], device=device),
            torch.tensor(self.g[idx], device=device),
            torch.tensor(self.ns[idx], device=device),
            torch.tensor(self.d[idx], device=device),
        )


# ─── Training ────────────────────────────────────────────────────────

def train(args):
    # ── DDP setup ──
    use_ddp = args.world_size > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        device = torch.device(args.device)

    save_dir = args.save_dir
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    # Each rank gets its own env pool and replay buffer
    pool = EnvPool(num_envs=args.num_envs, dt=args.dt, seed=42 + rank * 1000)

    S, A = pool.state_dim, pool.action_dim
    if rank == 0:
        total_envs = args.num_envs * args.world_size
        print(f"BRT Training | Double Integrator 2D | S={S} A={A}")
        print(f"  GPUs={args.world_size} | envs/GPU={args.num_envs} | total_envs={total_envs}")
        print(f"  batch/GPU={args.batch_size} | effective_batch={args.batch_size * args.world_size}")
        print(f"  Domain: [-2,2]^2 position, [-2,2]^2 velocity")
        print(f"  Obstacle: center=(0,0), radius=0.5")
        print(f"  dt={args.dt}, gamma={args.gamma}")

    # Networks
    actor = GaussianPolicy(S, A, hidden=args.hidden).to(device)
    critic = TwinQCritic(S, A, hidden=args.hidden).to(device)
    critic_tgt = TwinQCritic(S, A, hidden=args.hidden).to(device)
    critic_tgt.load_state_dict(critic.state_dict())
    for p in critic_tgt.parameters():
        p.requires_grad = False

    # Wrap in DDP for gradient synchronization
    if use_ddp:
        actor_ddp = DDP(actor, device_ids=[local_rank])
        critic_ddp = DDP(critic, device_ids=[local_rank])
    else:
        actor_ddp = actor
        critic_ddp = critic

    actor_opt = torch.optim.Adam(actor_ddp.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic_ddp.parameters(), lr=args.critic_lr)

    buf = ReplayBuffer(args.buffer_cap, S, A)
    states = pool.reset_all()

    for it in range(1, args.iterations + 1):
        t0 = time.time()

        # ── Collect experience (each rank collects independently) ──
        for _ in range(args.collect_steps):
            s_t = torch.tensor(states, dtype=torch.float32, device=device)
            with torch.no_grad():
                if use_ddp:
                    act_dist = actor.get_dist(s_t)
                else:
                    act_dist = actor_ddp.get_dist(s_t)
                actions = act_dist.sample()
            actions_np = actions.cpu().numpy()
            next_states, dones, infos = pool.step(actions_np)

            for i in range(args.num_envs):
                buf.add(states[i], actions_np[i],
                        infos[i]["safety_margin"],
                        next_states[i], float(dones[i]))
            states = next_states

        if buf.size < args.batch_size:
            continue

        # ── Update critic (HJ safety Bellman) ──
        critic_loss_avg = 0.0
        for _ in range(args.updates_per_iter):
            bs, ba, bg, bns, bd = buf.sample(args.batch_size, device)

            with torch.no_grad():
                # Next action from actor (for V(s') approximation)
                na = actor.get_dist(bns).mean
                nq = critic_tgt.min_q(bns, na)
                # HJ Bellman: target = (1 - γ(1-d))·g + γ(1-d)·min(g, V(s'))
                v_to_go = torch.minimum(bg, nq)
                tgt = (1.0 - args.gamma * (1.0 - bd)) * bg + \
                      args.gamma * (1.0 - bd) * v_to_go

            q1, q2 = critic_ddp(bs, ba)
            loss = F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt)

            critic_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_ddp.parameters(), 1.0)
            critic_opt.step()

            # Soft target update (on underlying module)
            with torch.no_grad():
                for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                    pt.data.mul_(1 - args.tau).add_(p.data * args.tau)

            critic_loss_avg += loss.item()

        # ── Update actor (maximize safety value) ──
        for _ in range(args.actor_updates):
            bs_a, _, _, _, _ = buf.sample(args.batch_size, device)
            mean, std = actor_ddp(bs_a)  # forward through DDP
            a_new = Normal(mean, std).rsample()
            q_val = critic.min_q(bs_a, a_new)
            actor_loss = -q_val.mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_ddp.parameters(), 0.5)
            actor_opt.step()

        elapsed = time.time() - t0
        critic_loss_avg /= max(args.updates_per_iter, 1)

        if rank == 0 and (it % 10 == 0 or it == 1):
            print(f"[{it:4d}/{args.iterations}] critic_loss={critic_loss_avg:.5f} "
                  f"buf={buf.size} | {elapsed:.2f}s", flush=True)

        if rank == 0 and (it % 50 == 0 or it == args.iterations):
            ckpt_path = os.path.join(save_dir, f"point_mass_brt_iter{it}.pth")
            torch.save({"critic": critic.state_dict(),
                        "actor": actor.state_dict()}, ckpt_path)

    # Save final (rank 0 only)
    if rank == 0:
        final_path = os.path.join(save_dir, "point_mass_brt_final.pth")
        torch.save({"critic": critic.state_dict(), "actor": actor.state_dict()}, final_path)
        print(f"Saved final model to {final_path}")

    if use_ddp:
        dist.destroy_process_group()

    return critic, actor


# ─── Visualization ────────────────────────────────────────────────────

def visualize_brt(critic, actor, device, save_path="point_mass_brt.png",
                  vel_slices=None):
    """
    Visualize the BRT as a 2D heatmap in position space (x, y)
    for fixed velocity slices.
    """
    if vel_slices is None:
        vel_slices = [
            (0.0, 0.0),    # stationary
            (1.0, 0.0),    # moving right
            (-1.0, 0.0),   # moving left
            (0.0, 1.0),    # moving up
        ]

    grid_res = 100
    xs = np.linspace(-2.0, 2.0, grid_res)
    ys = np.linspace(-2.0, 2.0, grid_res)
    X, Y = np.meshgrid(xs, ys)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (vx, vy) in enumerate(vel_slices):
        ax = axes[idx]

        # Build state grid for this velocity slice
        states = np.zeros((grid_res * grid_res, 4), dtype=np.float32)
        states[:, 0] = X.ravel()
        states[:, 1] = Y.ravel()
        states[:, 2] = vx
        states[:, 3] = vy

        s_t = torch.tensor(states, device=device)
        with torch.no_grad():
            # Get optimal action from actor for each state
            a_t = actor.get_dist(s_t).mean
            # Evaluate safety value
            V = critic.min_q(s_t, a_t).cpu().numpy()

        V_grid = V.reshape(grid_res, grid_res)

        # Plot heatmap
        im = ax.contourf(X, Y, V_grid, levels=50, cmap="RdYlGn")
        # BRT boundary (V=0 contour)
        ax.contour(X, Y, V_grid, levels=[0.0], colors="black", linewidths=2)
        # Obstacle
        circle = patches.Circle((0, 0), 0.5, fill=True, color="red", alpha=0.4,
                                label="Obstacle")
        ax.add_patch(circle)
        # Anchor point
        ax.plot(-2.0, 0.0, "bs", markersize=8, label="Anchor (-2,0)")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"BRT slice: vx={vx:.1f}, vy={vy:.1f}")
        plt.colorbar(im, ax=ax, label="V(s) (safety value)")

    plt.suptitle("Learned BRT for 2D Double Integrator (line-segment constraint)\n"
                 "Black contour = BRT boundary (V=0) | Segment from P to (-2,0) must clear obstacle\n"
                 "Red/negative = inside BRT (unsafe)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved BRT visualization to {save_path}")
    plt.close()


def visualize_from_checkpoint(ckpt_path: str, device: torch.device,
                              save_path: str = "point_mass_brt.png"):
    """Load a saved checkpoint and visualize."""
    S, A = 4, 2
    critic = TwinQCritic(S, A, hidden=256).to(device)
    actor = GaussianPolicy(S, A, hidden=256).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    critic.load_state_dict(ckpt["critic"])
    actor.load_state_dict(ckpt["actor"])
    critic.eval()
    actor.eval()

    visualize_brt(critic, actor, device, save_path=save_path)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Learn BRT for 2D point mass (double integrator)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_envs", type=int, default=128,
                   help="Envs per GPU rank")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--buffer_cap", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=1024,
                   help="Batch size per GPU rank")
    p.add_argument("--collect_steps", type=int, default=64)
    p.add_argument("--updates_per_iter", type=int, default=16)
    p.add_argument("--actor_updates", type=int, default=4)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--save_dir", default="/root/state_gated_ppo/checkpoints")
    p.add_argument("--world_size", type=int, default=None,
                   help="Number of GPUs (auto-detected from torchrun)")
    p.add_argument("--visualize", action="store_true",
                   help="Generate BRT visualization after training")
    p.add_argument("--viz_only", type=str, default=None,
                   help="Path to checkpoint to visualize (skip training)")
    args = p.parse_args()

    # Auto-detect world size from torchrun env
    if args.world_size is None:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.viz_only:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        visualize_from_checkpoint(args.viz_only, device)
        return

    critic, actor = train(args)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0 and args.visualize:
        device = next(critic.parameters()).device
        critic.eval()
        actor.eval()
        visualize_brt(critic, actor, device,
                      save_path=os.path.join(args.save_dir, "point_mass_brt.png"))


if __name__ == "__main__":
    main()
