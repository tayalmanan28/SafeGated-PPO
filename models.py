"""
Shared state-based policy, critics, and gated PPO update logic.

- StatePolicy: MLP actor-critic (Gaussian policy)
- SafetyCritic: twin-Q for HJ safety value
- gated_ppo_update(): one PPO epoch with feasibility gating
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ── MLP helper ──────────────────────────────────────────────────────

def mlp(dims, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


# ── State-based Gaussian policy + reward critic ─────────────────────

class StatePolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.actor_net = mlp([state_dim, hidden, hidden, action_dim])
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_net = mlp([state_dim, hidden, hidden, 1])

    def forward(self, state):
        mean = torch.tanh(self.actor_net(state))
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, state):
        mean, std = self(state)
        return Normal(mean, std)

    def value(self, state):
        return self.critic_net(state).squeeze(-1)


# ── Twin-Q safety critic (HJ) ──────────────────────────────────────

class SafetyCritic(nn.Module):
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


# ── Gated PPO update ───────────────────────────────────────────────

def gated_ppo_update(
    policy: StatePolicy,
    safety_critic: SafetyCritic,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,      # (B, state_dim)
    actions: torch.Tensor,      # (B, act_dim)
    old_log_probs: torch.Tensor,  # (B,)
    advantages: torch.Tensor,   # (B,)  GAE advantages
    returns: torch.Tensor,      # (B,)  discounted returns
    safety_values: torch.Tensor,  # (B,)  UNUSED (kept for API compat)
    *,
    clip_eps: float = 0.2,
    safety_delta: float = 0.2,  # gate threshold: Q_c > δ → safe
    beta: float = 5.0,          # safety loss coefficient (Eq. 5)
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
):
    """ShieldVLA-style gated PPO (Eq. 5-6 of the paper).

    Gate is evaluated at the CURRENT policy mean ā_θ(s):
      ζ = 1[Q_c(s, ā_θ(s)) > δ]
    Safe:   L_reward (standard PPO clip)
    Unsafe: L_safety = -Q_c(s, ā_θ(s))  (deterministic PG through Q_c)
    """
    B = states.size(0)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "safety_frac": 0.0}
    n_updates = 0

    for _ in range(ppo_epochs):
        idx = torch.randperm(B, device=states.device)
        for start in range(0, B, minibatch_size):
            mb = idx[start : start + minibatch_size]
            s, a = states[mb], actions[mb]
            old_lp = old_log_probs[mb]
            adv, ret = advantages[mb], returns[mb]

            # Current policy distribution
            dist_ = policy.get_dist(s)
            new_lp = dist_.log_prob(a).sum(-1)
            ratio = (new_lp - old_lp).exp()

            # -- Per-sample feasibility gate at policy mean (Eq. 5) --
            mean_a = dist_.mean                        # ā_θ(s)
            # Stop grad on state features for Q_c; grad flows through action only
            with torch.no_grad():
                qc_at_mean = safety_critic.min_q(s, mean_a.detach())
            gate = (qc_at_mean > safety_delta).float()  # ζ_θ(s), stop-grad
            safety_frac = 1.0 - gate.mean().item()

            # L_reward: standard PPO clipped surrogate (Eq. 5, feasible branch)
            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
            reward_loss = -torch.min(surr1, surr2)     # per-sample

            # L_safety: deterministic PG through Q_c at policy mean (Eq. 6)
            # Gradient: -∇_a Q_c(s,a)|_{a=ā} · ∇_θ ā_θ(s)
            qc_for_grad = safety_critic.min_q(s, mean_a)  # grad through mean_a
            safety_loss = -qc_for_grad                  # per-sample

            # Per-sample gated loss (Eq. 5)
            per_sample_loss = gate * reward_loss + beta * (1.0 - gate) * safety_loss
            policy_loss = per_sample_loss.mean()

            # Value loss
            v = policy.value(s)
            value_loss = F.mse_loss(v, ret)

            # Entropy bonus
            entropy = dist_.entropy().sum(-1).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["safety_frac"] += safety_frac
            n_updates += 1

    for k in stats:
        stats[k] /= max(n_updates, 1)
    return stats

# ── Null-space projection PPO update ─────────────────────────────────

def nullspace_ppo_update(
    policy: StatePolicy,
    safety_critic: SafetyCritic,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,       # (B, state_dim)
    actions: torch.Tensor,      # (B, act_dim)
    old_log_probs: torch.Tensor,  # (B,)
    advantages: torch.Tensor,   # (B,)
    returns: torch.Tensor,      # (B,)
    safety_values: torch.Tensor,  # (B,) UNUSED
    *,
    clip_eps: float = 0.2,
    safety_delta: float = 0.0,
    beta: float = 5.0,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
):
    """Null-space projection PPO: project reward gradient onto the null
    space of the safety gradient so they never conflict.

    Always optimizes reward. When unsafe states exist, the safety gradient
    defines a constraint direction; the reward gradient is projected to
    remove any component opposing safety. The safety gradient is then
    added on top (scaled by beta).
    """
    B = states.size(0)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "safety_frac": 0.0}
    n_updates = 0

    for _ in range(ppo_epochs):
        idx = torch.randperm(B, device=states.device)
        for start in range(0, B, minibatch_size):
            mb = idx[start : start + minibatch_size]
            s, a = states[mb], actions[mb]
            old_lp = old_log_probs[mb]
            adv, ret = advantages[mb], returns[mb]

            # Current policy
            dist_ = policy.get_dist(s)
            new_lp = dist_.log_prob(a).sum(-1)
            ratio = (new_lp - old_lp).exp()

            mean_a = dist_.mean
            with torch.no_grad():
                qc_at_mean = safety_critic.min_q(s, mean_a.detach())
            unsafe_mask = (qc_at_mean <= safety_delta).float()
            safety_frac = unsafe_mask.mean().item()

            # L_reward: PPO clipped surrogate (always computed)
            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
            reward_loss = -torch.min(surr1, surr2).mean()

            # L_safety: deterministic PG through Q_c (only on unsafe states)
            qc_for_grad = safety_critic.min_q(s, mean_a)
            safety_loss = -(unsafe_mask * qc_for_grad).mean()

            # Value loss & entropy
            v = policy.value(s)
            value_loss = F.mse_loss(v, ret)
            entropy = dist_.entropy().sum(-1).mean()

            optimizer.zero_grad()

            if safety_frac > 0.01:
                # Get reward gradient
                reward_loss.backward(retain_graph=True)
                g_r = torch.cat([p.grad.flatten() if p.grad is not None
                                 else torch.zeros(p.numel(), device=p.device)
                                 for p in policy.parameters()])

                # Get safety gradient
                for p in policy.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                safety_loss.backward(retain_graph=True)
                g_s = torch.cat([p.grad.flatten() if p.grad is not None
                                 else torch.zeros(p.numel(), device=p.device)
                                 for p in policy.parameters()])

                # Project reward onto null space of safety
                g_s_norm_sq = torch.dot(g_s, g_s)
                if g_s_norm_sq > 1e-10:
                    dot = torch.dot(g_r, g_s)
                    # Only project out conflicting component (dot > 0 means
                    # reward grad opposes safety improvement)
                    if dot > 0:
                        g_r_proj = g_r - (dot / g_s_norm_sq) * g_s
                    else:
                        g_r_proj = g_r  # reward already helps safety, keep it
                else:
                    g_r_proj = g_r

                # Combined: projected reward + scaled safety
                g_combined = g_r_proj + beta * safety_frac * g_s

                # Write combined gradient back to params
                for p in policy.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                (vf_coef * value_loss - ent_coef * entropy).backward()
                offset = 0
                for p in policy.parameters():
                    numel = p.numel()
                    if p.grad is None:
                        continue
                    p.grad.add_(g_combined[offset:offset + numel].view_as(p.grad))
                    offset += numel
            else:
                # All safe — standard PPO
                total = reward_loss + vf_coef * value_loss - ent_coef * entropy
                total.backward()

            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            stats["policy_loss"] += reward_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["safety_frac"] += safety_frac
            n_updates += 1

    for k in stats:
        stats[k] /= max(n_updates, 1)
    return stats