"""
Manipulator reach-avoid environment (state-based).

A simplified 3-DOF planar arm must reach a target end-effector position
while avoiding a spherical obstacle in workspace.

Actions are joint-position deltas (like π₀.5 model outputs), not torques.

State: [joint_angles(3), joint_vels(3), rel_ee_goal(2), rel_ee_obs(2), obs_dist]  (11-D)
Action: [Δq1, Δq2, Δq3]  in [-1, 1]  (scaled to ±0.1 rad per step)
"""
from __future__ import annotations

import math
import numpy as np

from base_env import BaseNavEnv

LINK_LENGTHS = np.array([0.4, 0.3, 0.2])
GOAL_RADIUS = 0.15
OBSTACLE_RADIUS = 0.12
MAX_STEPS = 200
WORKSPACE_R = sum(LINK_LENGTHS)  # max reach


def forward_kinematics(q):
    """Return end-effector (x, y) for 3-DOF planar arm."""
    x = y = 0.0
    angle = 0.0
    for i in range(3):
        angle += q[i]
        x += LINK_LENGTHS[i] * math.cos(angle)
        y += LINK_LENGTHS[i] * math.sin(angle)
    return np.array([x, y], dtype=np.float32)


class ManipulatorReachEnv(BaseNavEnv):
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self._step_count = 0
        self.dt = 0.02
        self.q = np.zeros(3)     # joint angles
        self.qd = np.zeros(3)    # joint velocities
        self.goal_xy = np.zeros(2)
        self.obs_xy = np.zeros(2)
        self.prev_dist = 0.0

    @property
    def state_dim(self) -> int:
        return 11

    @property
    def action_dim(self) -> int:
        return 3

    def _random_reachable_point(self):
        """Sample a random point within workspace."""
        while True:
            p = self.rng.uniform(-WORKSPACE_R, WORKSPACE_R, 2)
            if np.linalg.norm(p) < WORKSPACE_R * 0.9:
                return p.astype(np.float32)

    def reset(self) -> np.ndarray:
        rng = self.rng
        self.q = rng.uniform(-math.pi / 2, math.pi / 2, 3)
        self.qd = np.zeros(3)
        ee = forward_kinematics(self.q)

        self.goal_xy = self._random_reachable_point()
        while np.linalg.norm(self.goal_xy - ee) < 0.2:
            self.goal_xy = self._random_reachable_point()

        # Place obstacle near path
        mid = (ee + self.goal_xy) / 2 + rng.uniform(-0.1, 0.1, 2)
        self.obs_xy = mid.astype(np.float32)

        self.prev_dist = np.linalg.norm(ee - self.goal_xy)
        self._step_count = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        ee = forward_kinematics(self.q)
        rel_goal = self.goal_xy - ee
        rel_obs = self.obs_xy - ee
        obs_dist = np.linalg.norm(rel_obs) - OBSTACLE_RADIUS
        return np.concatenate([
            self.q, self.qd, rel_goal, rel_obs, [obs_dist]
        ]).astype(np.float32)

    def step(self, action: np.ndarray):
        # Joint position delta control (π₀.5 style)
        dq = np.clip(action, -1.0, 1.0) * 0.15  # ±0.15 rad max per step
        q_target = self.q + dq
        # PD tracking to target
        kp, kd = 20.0, 2.0
        qdd = kp * (q_target - self.q) - kd * self.qd
        self.qd += qdd * self.dt
        self.qd = np.clip(self.qd, -5.0, 5.0)
        self.q += self.qd * self.dt

        ee = forward_kinematics(self.q)
        dist = np.linalg.norm(ee - self.goal_xy)
        obs_d = np.linalg.norm(ee - self.obs_xy)
        safety_margin = obs_d - OBSTACLE_RADIUS

        approach = self.prev_dist - dist
        # Dense reward: approach + proximity + direction bonus
        reward = approach * 10.0
        reward += math.exp(-dist / 0.3) * 0.3   # proximity shaping
        reward += max(0, approach) * 5.0          # extra for positive progress
        self.prev_dist = dist

        done = False
        reached = False
        if dist < GOAL_RADIUS:
            reward += 20.0
            done = True
            reached = True

        cost = 1.0 if safety_margin < 0 else 0.0

        self._step_count += 1
        if self._step_count >= MAX_STEPS:
            done = True

        info = {"cost": cost, "safety_margin": safety_margin, "reached": reached}
        return self._get_state(), reward, done, info
