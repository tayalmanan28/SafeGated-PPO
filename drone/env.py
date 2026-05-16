"""
Crazyflie drone 3D navigation environment (state-based).

State: [rel_goal_x, rel_goal_y, rel_goal_z, vx, vy, vz,
        rel_obs_x, rel_obs_y, rel_obs_z, obs_dist, roll, pitch]  (12-D)
Action: [vx_cmd, vy_cmd, vz_cmd, yaw_rate]  in [-1, 1]
"""
from __future__ import annotations

import math
import numpy as np

from base_env import BaseNavEnv

ARENA_XY = 2.5
ARENA_Z = (0.2, 2.0)
GOAL_RADIUS = 0.4
OBSTACLE_RADIUS = 0.5
MAX_STEPS = 150


class DroneNavEnv(BaseNavEnv):
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self._step_count = 0
        self.dt = 0.05  # 20 Hz
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.yaw = 0.0
        self.goal = np.zeros(3)
        self.obs_pos = np.zeros(3)
        self.prev_dist = 0.0

    @property
    def state_dim(self) -> int:
        return 12

    @property
    def action_dim(self) -> int:
        return 4

    def reset(self) -> np.ndarray:
        rng = self.rng
        self.pos = np.array([rng.uniform(-ARENA_XY, ARENA_XY),
                             rng.uniform(-ARENA_XY, ARENA_XY),
                             rng.uniform(*ARENA_Z)])
        self.vel = np.zeros(3)
        self.yaw = rng.uniform(-math.pi, math.pi)

        while True:
            g = np.array([rng.uniform(-ARENA_XY, ARENA_XY),
                          rng.uniform(-ARENA_XY, ARENA_XY),
                          rng.uniform(*ARENA_Z)])
            if np.linalg.norm(g - self.pos) > 1.5:
                break
        self.goal = g

        mid = (self.pos + g) / 2 + rng.uniform(-0.5, 0.5, 3)
        mid[2] = np.clip(mid[2], *ARENA_Z)
        mid[:2] = np.clip(mid[:2], -ARENA_XY, ARENA_XY)
        self.obs_pos = mid

        self.prev_dist = np.linalg.norm(self.pos - self.goal)
        self._step_count = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        rel_goal = self.goal - self.pos
        rel_obs = self.obs_pos - self.pos
        obs_dist = np.linalg.norm(rel_obs) - OBSTACLE_RADIUS
        # Simple roll/pitch = 0 for state-only (no actual attitude sim)
        return np.concatenate([
            rel_goal, self.vel,
            rel_obs, [obs_dist, 0.0, 0.0]
        ]).astype(np.float32)

    def step(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)
        # Simple velocity-command dynamics
        target_vel = a[:3] * 1.0  # max 1 m/s per axis
        self.vel = 0.8 * self.vel + 0.2 * target_vel  # first-order lag
        self.yaw += a[3] * 2.0 * self.dt
        self.pos += self.vel * self.dt
        self.pos[:2] = np.clip(self.pos[:2], -ARENA_XY, ARENA_XY)
        self.pos[2] = np.clip(self.pos[2], *ARENA_Z)

        dist = np.linalg.norm(self.pos - self.goal)
        obs_d = np.linalg.norm(self.pos - self.obs_pos)
        safety_margin = obs_d - OBSTACLE_RADIUS

        approach = self.prev_dist - dist
        reward = approach * 2.0 + math.exp(-dist / 0.5) * 0.1
        self.prev_dist = dist

        done = False
        reached = False
        if dist < GOAL_RADIUS:
            reward += 5.0
            done = True
            reached = True

        cost = 1.0 if safety_margin < 0 else 0.0

        self._step_count += 1
        if self._step_count >= MAX_STEPS:
            done = True

        info = {"cost": cost, "safety_margin": safety_margin, "reached": reached}
        return self._get_state(), reward, done, info
