"""
Unified Feasibility-Gated PPO training for multiple robots (state-based).

Mirrors the gating logic from train_gated_qc_vision.py:
  Safe  (Q_c > δ):  standard PPO reward maximization
  Unsafe (Q_c ≤ δ): maximize Q_c (safety recovery) with adaptive coef

Supports: turtlebot, drone, manipulator.

Usage:
  python train.py --robot turtlebot --num_envs 64
  python train.py --robot drone     --num_envs 64
  python train.py --robot manipulator --num_envs 64
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from base_env import EnvPool
from models import StatePolicy, SafetyCritic, gated_ppo_update, nullspace_ppo_update


# ── Robot registry ──────────────────────────────────────────────────

def make_env_fn(robot: str, seed: int):
    if robot == "turtlebot":
        from turtlebot.env import TurtleBotNavEnv
        return lambda: TurtleBotNavEnv(seed=seed)
    elif robot == "drone":
        from drone.env import DroneNavEnv
        return lambda: DroneNavEnv(seed=seed)
    elif robot == "manipulator":
        from manipulator.env import ManipulatorReachEnv
        return lambda: ManipulatorReachEnv(seed=seed)
    elif robot == "fr3":
        from manipulator.fr3_env import FR3ReachAvoidEnv
        return lambda: FR3ReachAvoidEnv(seed=seed)
    else:
        raise ValueError(f"Unknown robot: {robot}")


# ── GAE computation ─────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N)
    for t in reversed(range(T)):
        nv = values[t + 1] if t < T - 1 else torch.zeros(N)
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * nv * mask - values[t]
        advantages[t] = last_gae = delta + gamma * lam * mask * last_gae
    returns = advantages + values
    return advantages, returns


# ── Main training loop ──────────────────────────────────────────────

def main(args):
    device = torch.device(args.device)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Create env pool — each env gets a different seed
    env_fn = make_env_fn(args.robot, seed=42)
    pool = EnvPool(lambda: make_env_fn(args.robot, seed=np.random.randint(1_000_000))(), args.num_envs)
    S, A = pool.state_dim, pool.action_dim
    print(f"Robot: {args.robot} | state_dim={S} | action_dim={A} | "
          f"num_envs={args.num_envs} | device={device}")

    # Policy + Safety critic
    policy = StatePolicy(S, A, hidden=args.hidden).to(device)
    safety_critic = SafetyCritic(S, A, hidden=args.hidden).to(device)
    safety_critic_tgt = SafetyCritic(S, A, hidden=args.hidden).to(device)
    safety_critic_tgt.load_state_dict(safety_critic.state_dict())
    for p in safety_critic_tgt.parameters():
        p.requires_grad = False

    # Load pretrained Q_c if available
    if args.qc_ckpt and os.path.exists(args.qc_ckpt):
        ck = torch.load(args.qc_ckpt, map_location="cpu", weights_only=False)
        safety_critic.load_state_dict(ck["critic"])
        safety_critic_tgt.load_state_dict(ck["critic"])
        print(f"Loaded Q_c from {args.qc_ckpt}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    qc_opt = torch.optim.Adam(safety_critic.parameters(), lr=args.qc_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)

    # Q_c replay buffer (simple lists)
    qc_buf = {"s": [], "a": [], "g": [], "ns": [], "d": []}
    QC_CAP = args.qc_buffer_cap

    # Adaptive safety coef (Lagrangian-style)
    safety_coef = args.safety_coef

    states = pool.reset_all()  # (N, S)
    best_reach = 0.0
    total_reaches = total_episodes = 0
    total_safe_episodes = 0
    env_violated = [False] * args.num_envs  # per-env violation flag for current episode

    for it in range(1, args.iterations + 1):
        t0 = time.time()

        # ── Rollout collection ──
        s_buf, a_buf, lp_buf, v_buf = [], [], [], []
        r_buf, d_buf, cost_buf, margin_buf = [], [], [], []
        iter_reaches = 0
        iter_episodes = 0
        iter_safe_episodes = 0

        for step in range(args.rollout_steps):
            s_t = torch.tensor(states, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist_ = policy.get_dist(s_t)
                actions = dist_.sample()
                log_probs = dist_.log_prob(actions).sum(-1)
                values = policy.value(s_t)

            actions_np = actions.cpu().numpy()
            next_states, rewards, dones, infos = pool.step(actions_np)

            costs = np.array([info["cost"] for info in infos], dtype=np.float32)
            margins = np.array([info["safety_margin"] for info in infos], dtype=np.float32)

            # Track episode-level metrics
            for i in range(args.num_envs):
                if costs[i] > 0:
                    env_violated[i] = True
                if dones[i]:
                    iter_episodes += 1
                    if infos[i]["reached"]:
                        iter_reaches += 1
                    if not env_violated[i]:
                        iter_safe_episodes += 1
                    env_violated[i] = False  # reset for next episode

            # Q_c replay (ring buffer via index)
            for i in range(args.num_envs):
                if len(qc_buf["s"]) < QC_CAP:
                    qc_buf["s"].append(states[i].copy())
                    qc_buf["a"].append(actions_np[i].copy())
                    qc_buf["g"].append(margins[i])
                    qc_buf["ns"].append(next_states[i].copy())
                    qc_buf["d"].append(float(dones[i]))
                else:
                    idx_w = np.random.randint(0, QC_CAP)
                    qc_buf["s"][idx_w] = states[i].copy()
                    qc_buf["a"][idx_w] = actions_np[i].copy()
                    qc_buf["g"][idx_w] = margins[i]
                    qc_buf["ns"][idx_w] = next_states[i].copy()
                    qc_buf["d"][idx_w] = float(dones[i])

            s_buf.append(s_t.cpu())
            a_buf.append(actions.cpu())
            lp_buf.append(log_probs.cpu())
            v_buf.append(values.cpu())
            r_buf.append(torch.tensor(rewards))
            d_buf.append(torch.tensor(dones))
            cost_buf.append(torch.tensor(costs))
            margin_buf.append(torch.tensor(margins))

            states = next_states

        # Stack: (T, N, ...)
        all_s = torch.stack(s_buf)
        all_a = torch.stack(a_buf)
        all_lp = torch.stack(lp_buf)
        all_v = torch.stack(v_buf)
        all_r = torch.stack(r_buf)
        all_d = torch.stack(d_buf)
        all_c = torch.stack(cost_buf)
        all_m = torch.stack(margin_buf)

        T, N = all_r.shape
        total_reaches += iter_reaches
        total_episodes += iter_episodes
        total_safe_episodes += iter_safe_episodes

        t_collect = time.time() - t0

        # ── Online Q_c training ──
        qc_loss_val = 0.0
        buf_len = len(qc_buf["s"])
        if buf_len >= args.qc_batch_size:
            for _ in range(args.qc_updates):
                idx = np.random.randint(0, buf_len, size=args.qc_batch_size)
                bs = torch.tensor(np.array([qc_buf["s"][i] for i in idx]),
                                  dtype=torch.float32, device=device)
                ba = torch.tensor(np.array([qc_buf["a"][i] for i in idx]),
                                  dtype=torch.float32, device=device)
                bg = torch.tensor(np.array([qc_buf["g"][i] for i in idx]),
                                  dtype=torch.float32, device=device)
                bns = torch.tensor(np.array([qc_buf["ns"][i] for i in idx]),
                                   dtype=torch.float32, device=device)
                bd = torch.tensor(np.array([qc_buf["d"][i] for i in idx]),
                                  dtype=torch.float32, device=device)

                with torch.no_grad():
                    # max_a' Q_c(s',a') ≈ Q_c(s', ā_θ(s'))  (policy mean)
                    na = policy.get_dist(bns).mean
                    nq = safety_critic_tgt.min_q(bns, na)
                    v_to_go = torch.minimum(bg, nq)
                    tgt = (1.0 - args.qc_gamma * (1.0 - bd)) * bg + \
                          args.qc_gamma * (1.0 - bd) * v_to_go

                q1, q2 = safety_critic(bs, ba)
                loss_qc = F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt)
                qc_opt.zero_grad()
                loss_qc.backward()
                torch.nn.utils.clip_grad_norm_(safety_critic.parameters(), 1.0)
                qc_opt.step()

                # Polyak update
                with torch.no_grad():
                    for p, pt in zip(safety_critic.parameters(), safety_critic_tgt.parameters()):
                        pt.data.mul_(1 - args.qc_tau).add_(p.data * args.qc_tau)

                qc_loss_val = loss_qc.item()

        # ── GAE ──
        advantages, returns = compute_gae(all_r, all_v, all_d)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Compute safety values for gating ──
        flat_s = all_s.reshape(-1, S).to(device)
        flat_a = all_a.reshape(-1, A).to(device)
        with torch.no_grad():
            safety_vals = safety_critic.min_q(flat_s, flat_a).cpu()

        # ── Gated PPO update ──
        # During warmup, disable gating (all states treated as safe)
        effective_delta = args.safety_delta if it > args.critic_warmup else -1e6

        update_fn = nullspace_ppo_update if args.method == 'nullspace' else gated_ppo_update
        stats = update_fn(
            policy, safety_critic, optimizer,
            flat_s,
            flat_a,
            all_lp.reshape(-1).to(device),
            advantages.reshape(-1).to(device),
            returns.reshape(-1).to(device),
            safety_vals.to(device),
            clip_eps=args.clip_eps,
            safety_delta=effective_delta,
            beta=safety_coef,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.batch_size,
        )
        scheduler.step()

        # ── Adaptive safety coef ──
        ep_cost = all_c.sum(dim=0).mean().item()
        if args.coef_lr > 0:
            safety_coef = max(args.coef_min,
                              min(args.coef_max,
                                  safety_coef + args.coef_lr * (ep_cost - args.cost_limit)))

        elapsed = time.time() - t0
        avg_r = all_r.mean().item()
        avg_c = all_c.mean().item()
        sps = T * N / elapsed

        reach_rate = iter_reaches / max(iter_episodes, 1)
        safety_rate = iter_safe_episodes / max(iter_episodes, 1)
        cum_reach = total_reaches / max(total_episodes, 1)
        cum_safety = total_safe_episodes / max(total_episodes, 1)

        if reach_rate > best_reach:
            best_reach = reach_rate
            torch.save({"policy": policy.state_dict(),
                        "critic": safety_critic.state_dict()},
                       os.path.join(save_dir, f"{args.robot}_best.pth"))

        print(f"[{it:4d}/{args.iterations}] loss={stats['policy_loss']:.4f} "
              f"rew={avg_r:.3f} cost={avg_c:.3f} qc={qc_loss_val:.4f} "
              f"scoef={safety_coef:.2f} sfrac={stats['safety_frac']:.2f} "
              f"reach={reach_rate:.0%}({iter_reaches}/{iter_episodes}) "
              f"safe={safety_rate:.0%}({iter_safe_episodes}/{iter_episodes}) | cum_r={cum_reach:.0%} cum_s={cum_safety:.0%} "
              f"| {sps:.0f} sps {elapsed:.1f}s",
              flush=True)

        if it % 50 == 0:
            torch.save({"policy": policy.state_dict(),
                        "critic": safety_critic.state_dict()},
                       os.path.join(save_dir, f"{args.robot}_iter{it}.pth"))

    cum_reach = total_reaches / max(total_episodes, 1)
    cum_safety = total_safe_episodes / max(total_episodes, 1)
    print(f"\n{'='*60}")
    print(f"FINAL — {args.robot}")
    print(f"  Success rate:  {cum_reach:.1%} (best iter: {best_reach:.1%})")
    print(f"  Safety rate:   {cum_safety:.1%} ({total_safe_episodes}/{total_episodes} episodes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=["turtlebot", "drone", "manipulator", "fr3"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rollout_steps", type=int, default=64)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--clip_eps", type=float, default=0.2)
    # Safety / gating
    p.add_argument("--method", choices=["gated", "nullspace"], default="gated",
                   help="'gated' = hard binary gate (ShieldVLA Eq.5), 'nullspace' = null-space projection")
    p.add_argument("--safety_delta", type=float, default=0.0,
                   help="Q_c gate threshold (gate = Q_c > delta). Higher = more conservative.")
    p.add_argument("--critic_warmup", type=int, default=0,
                   help="Disable gating for first N iterations (pure PPO warmup for Q_c)")
    p.add_argument("--safety_coef", type=float, default=5.0)
    p.add_argument("--coef_lr", type=float, default=0.05)
    p.add_argument("--coef_min", type=float, default=0.5)
    p.add_argument("--coef_max", type=float, default=15.0)
    p.add_argument("--cost_limit", type=float, default=3.0)
    # Q_c online learning
    p.add_argument("--qc_ckpt", default="", help="Pretrained Q_c checkpoint")
    p.add_argument("--qc_lr", type=float, default=3e-4)
    p.add_argument("--qc_gamma", type=float, default=0.995)
    p.add_argument("--qc_tau", type=float, default=0.01)
    p.add_argument("--qc_updates", type=int, default=4)
    p.add_argument("--qc_batch_size", type=int, default=256)
    p.add_argument("--qc_buffer_cap", type=int, default=100_000)
    # Save
    p.add_argument("--save_dir", default="/root/state_gated_ppo/checkpoints")
    main(p.parse_args())
