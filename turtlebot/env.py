"""
TurtleBot 2D navigation environment (state-based).

State: [rel_goal_x, rel_goal_y, cos_heading, sin_heading, obs_rel_x, obs_rel_y, obs_dist]
Action: [v, omega]  (linear velocity, angular velocity)
"""
from __future__ import annotations

import math
import numpy as np

from base_env import BaseNavEnv

ARENA = 2.5          # half-size of arena
GOAL_RADIUS = 0.5
OBSTACLE_RADIUS = 0.6
MAX_STEPS = 100


class TurtleBotNavEnv(BaseNavEnv):
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self._step_count = 0
        self.dt = 0.1  # 10 Hz
        self.x = self.y = self.yaw = 0.0
        self.goal = np.zeros(2)
        self.obs_pos = np.zeros(2)
        self.prev_dist = 0.0

    @property
    def state_dim(self) -> int:
        return 7

    @property
    def action_dim(self) -> int:
        return 2

    def reset(self) -> np.ndarray:
        rng = self.rng
        # Random start
        self.x, self.y = rng.uniform(-ARENA, ARENA, 2)
        self.yaw = rng.uniform(-math.pi, math.pi)

        # Random goal far from start
        while True:
            g = rng.uniform(-ARENA, ARENA, 2)
            if np.linalg.norm(g - [self.x, self.y]) > 1.5:
                break
        self.goal = g

        # Obstacle between start and goal
        mid = (np.array([self.x, self.y]) + g) / 2 + rng.uniform(-0.5, 0.5, 2)
        self.obs_pos = np.clip(mid, -ARENA, ARENA)

        self.prev_dist = np.linalg.norm([self.x - g[0], self.y - g[1]])
        self._step_count = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        rel_gx = dx * c - dy * s
        rel_gy = dx * s + dy * c

        odx = self.obs_pos[0] - self.x
        ody = self.obs_pos[1] - self.y
        rel_ox = odx * c - ody * s
        rel_oy = odx * s + ody * c
        obs_dist = math.sqrt(odx**2 + ody**2) - OBSTACLE_RADIUS

        return np.array([rel_gx, rel_gy, math.cos(self.yaw), math.sin(self.yaw),
                         rel_ox, rel_oy, obs_dist], dtype=np.float32)

    def step(self, action: np.ndarray):
        v = float(np.clip(action[0], 0.0, 1.0))
        w = float(np.clip(action[1], -2.0, 2.0))

        # Simple unicycle integration
        self.yaw += w * self.dt
        self.x += v * math.cos(self.yaw) * self.dt
        self.y += v * math.sin(self.yaw) * self.dt
        self.x = np.clip(self.x, -ARENA, ARENA)
        self.y = np.clip(self.y, -ARENA, ARENA)

        dist = math.sqrt((self.x - self.goal[0])**2 + (self.y - self.goal[1])**2)
        obs_d = math.sqrt((self.x - self.obs_pos[0])**2 + (self.y - self.obs_pos[1])**2)
        safety_margin = obs_d - OBSTACLE_RADIUS

        # Reward
        approach = self.prev_dist - dist
        reward = approach * 2.0
        angle_to_goal = math.atan2(self.goal[1] - self.y, self.goal[0] - self.x)
        heading_err = abs(math.atan2(math.sin(angle_to_goal - self.yaw),
                                      math.cos(angle_to_goal - self.yaw)))
        reward += (1.0 - heading_err / math.pi) * 0.05
        reward += math.exp(-dist / 0.5) * 0.1
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
