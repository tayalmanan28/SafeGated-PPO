"""
Base environment interface for state-based gated PPO across multiple robots.

Every robot-specific env must subclass `BaseNavEnv` and implement:
  - reset()       → state (np.ndarray)
  - step(action)  → state, reward, done, info
  - state_dim     (property)
  - action_dim    (property)
  - cost(state)   → float  (positive = safe, negative = unsafe)

The vectorized `EnvPool` wraps N copies for parallel rollout.
"""
from __future__ import annotations

import abc
from typing import Dict, Tuple

import numpy as np


class BaseNavEnv(abc.ABC):
    """Single-instance navigation environment."""

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Returns (state, reward, done, info).
        info must include 'cost' (float, >0 safe, <0 unsafe)."""
        ...

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        ...


class EnvPool:
    """Vectorized wrapper around N BaseNavEnv instances."""

    def __init__(self, env_fn, n: int):
        self.envs = [env_fn() for _ in range(n)]
        self.n = n

    @property
    def state_dim(self) -> int:
        return self.envs[0].state_dim

    @property
    def action_dim(self) -> int:
        return self.envs[0].action_dim

    def reset_all(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray):
        """actions: (N, act_dim).  Returns states, rewards, dones, infos."""
        states, rewards, dones, infos = [], [], [], []
        for i, e in enumerate(self.envs):
            s, r, d, info = e.step(actions[i])
            if d:
                s = e.reset()
            states.append(s)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return (
            np.stack(states),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )
