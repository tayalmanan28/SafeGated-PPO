"""
Standalone HJ safety critic pretraining (SAC-style) for any robot.

Trains a twin-Q safety critic Q_c(s,a) using HJ Bellman:
  target = (1 - γ(1-d))·g + γ(1-d)·min(g, V(s'))

Usage:
  python train_hj.py --robot turtlebot --iterations 200
  python train_hj.py --robot drone     --iterations 200
  python train_hj.py --robot manipulator --iterations 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from base_env import EnvPool
from models import StatePolicy, SafetyCritic


def make_env_fn(robot, seed):
    if robot == "turtlebot":
        from turtlebot.env import TurtleBotNavEnv
        return lambda: TurtleBotNavEnv(seed=seed)
    elif robot == "drone":
        from drone.env import DroneNavEnv
        return lambda: DroneNavEnv(seed=seed)
    elif robot == "manipulator":
        from manipulator.env import ManipulatorReachEnv
        return lambda: ManipulatorReachEnv(seed=seed)
    else:
        raise ValueError(f"Unknown robot: {robot}")


class ReplayBuffer:
    def __init__(self, cap, s_dim, a_dim):
        self.cap = cap
        self.s = np.zeros((cap, s_dim), dtype=np.float32)
        self.a = np.zeros((cap, a_dim), dtype=np.float32)
        self.g = np.zeros(cap, dtype=np.float32)
        self.ns = np.zeros((cap, s_dim), dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
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

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (torch.tensor(self.s[idx], device=device),
                torch.tensor(self.a[idx], device=device),
                torch.tensor(self.g[idx], device=device),
                torch.tensor(self.ns[idx], device=device),
                torch.tensor(self.d[idx], device=device))


def main(args):
    device = torch.device(args.device)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    pool = EnvPool(lambda: make_env_fn(args.robot, seed=np.random.randint(1_000_000))(),
                   args.num_envs)
    S, A = pool.state_dim, pool.action_dim
    print(f"HJ Critic Training | Robot: {args.robot} | S={S} A={A} | envs={args.num_envs}")

    # Actor (random or pretrained) for exploration
    actor = StatePolicy(S, A, hidden=args.hidden).to(device)
    critic = SafetyCritic(S, A, hidden=args.hidden).to(device)
    critic_tgt = SafetyCritic(S, A, hidden=args.hidden).to(device)
    critic_tgt.load_state_dict(critic.state_dict())
    for p in critic_tgt.parameters():
        p.requires_grad = False

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    buf = ReplayBuffer(args.buffer_cap, S, A)
    states = pool.reset_all()

    for it in range(1, args.iterations + 1):
        t0 = time.time()

        # Collect data with exploration noise
        for _ in range(args.collect_steps):
            s_t = torch.tensor(states, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist_ = actor.get_dist(s_t)
                actions = dist_.sample()
            actions_np = actions.cpu().numpy()
            next_states, _, dones, infos = pool.step(actions_np)

            for i in range(args.num_envs):
                buf.add(states[i], actions_np[i],
                        infos[i]["safety_margin"],
                        next_states[i], float(dones[i]))
            states = next_states

        if buf.size < args.batch_size:
            continue

        # Train critic
        critic_loss_avg = 0.0
        for _ in range(args.updates_per_iter):
            bs, ba, bg, bns, bd = buf.sample(args.batch_size, device)

            with torch.no_grad():
                na = actor.get_dist(bns).mean  # max_a' Q(s',a') ≈ Q(s', ā(s'))
                nq = critic_tgt.min_q(bns, na)
                v_to_go = torch.minimum(bg, nq)
                tgt = (1.0 - args.gamma * (1.0 - bd)) * bg + \
                      args.gamma * (1.0 - bd) * v_to_go

            q1, q2 = critic(bs, ba)
            loss = F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt)

            critic_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            with torch.no_grad():
                for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                    pt.data.mul_(1 - args.tau).add_(p.data * args.tau)

            critic_loss_avg += loss.item()

        # Train actor to maximize Q_c (safety-aware exploration)
        for _ in range(args.actor_updates):
            bs, _, _, _, _ = buf.sample(args.batch_size, device)
            a_new = actor.get_dist(bs).rsample()
            q_val = critic.min_q(bs, a_new)
            actor_loss = -q_val.mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_opt.step()

        elapsed = time.time() - t0
        critic_loss_avg /= max(args.updates_per_iter, 1)
        print(f"[{it:4d}/{args.iterations}] critic_loss={critic_loss_avg:.4f} "
              f"buf={buf.size} | {elapsed:.1f}s", flush=True)

        if it % 50 == 0 or it == args.iterations:
            torch.save({"critic": critic.state_dict(),
                        "actor": actor.state_dict()},
                       os.path.join(save_dir, f"hj_{args.robot}_iter{it}.pth"))

    # Save final
    path = os.path.join(save_dir, f"hj_{args.robot}_final.pth")
    torch.save({"critic": critic.state_dict(), "actor": actor.state_dict()}, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=["turtlebot", "drone", "manipulator"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--buffer_cap", type=int, default=500_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--collect_steps", type=int, default=64)
    p.add_argument("--updates_per_iter", type=int, default=8)
    p.add_argument("--actor_updates", type=int, default=2)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--save_dir", default="/root/state_gated_ppo/checkpoints")
    main(p.parse_args())
