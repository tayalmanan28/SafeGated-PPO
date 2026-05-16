"""
Franka FR3 7-DOF reach-avoid environment (state-based, MuJoCo).

The robot must move its end-effector to a randomized goal position while
avoiding a long cylindrical obstacle placed in the workspace.

Actions are joint-position deltas (π₀.5 style):
  action ∈ [-1, 1]^7, scaled to ±0.05 rad, added to current qpos → sent
  as position targets to FR3's built-in PD actuators.

State (21-D):
  joint_angles(7), joint_vels(7), rel_ee_goal(3), rel_ee_obs(3), obs_dist(1)

Safety:
  cost=1 if EE is within OBSTACLE_RADIUS of the cylinder axis, else 0.
  safety_margin = distance_to_cylinder_axis − OBSTACLE_RADIUS
"""
from __future__ import annotations

import math
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_env import BaseNavEnv

_SCENE_XML = os.path.join(os.path.dirname(__file__),
                          "mujoco_menagerie", "franka_fr3", "fr3_reach_avoid.xml")

# FR3 joint names (7-DOF)
_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]

# Workspace bounds for randomizing goals (meters, in robot base frame)
_GOAL_BOUNDS_LO = np.array([0.25, -0.45, 0.15], dtype=np.float32)
_GOAL_BOUNDS_HI = np.array([0.65,  0.45, 0.65], dtype=np.float32)

# Obstacle cylinder: radius and half-height from the XML geom
OBSTACLE_RADIUS = 0.06    # cylinder radius
OBSTACLE_HALFH  = 0.25    # cylinder half-height (z extent)

GOAL_RADIUS  = 0.05       # success threshold (m)
MAX_STEPS    = 200
DQ_SCALE     = 0.05       # action → ±0.05 rad per step
CTRL_DT      = 0.02       # 50 Hz control (10 physics steps @ 0.002 dt)
PHYSICS_SUBSTEPS = 10


def _dist_point_to_cylinder_axis(point, cyl_pos, cyl_halfh):
    """Horizontal distance from point to vertical cylinder axis,
    clamped to the cylinder's z-extent."""
    dx = point[0] - cyl_pos[0]
    dy = point[1] - cyl_pos[1]
    horiz = math.sqrt(dx * dx + dy * dy)
    # Vertical: clamp to cylinder extent
    dz = point[2] - cyl_pos[2]
    dz_clamp = max(-cyl_halfh, min(cyl_halfh, dz))
    vert = dz - dz_clamp
    return math.sqrt(horiz * horiz + vert * vert)


class FR3ReachAvoidEnv(BaseNavEnv):
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self._step_count = 0

        # Load model
        self.model = mujoco.MjModel.from_xml_path(_SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        # Cache IDs
        self._joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                           for n in _JOINT_NAMES]
        self._qpos_adr = [self.model.jnt_qposadr[j] for j in self._joint_ids]
        self._qvel_adr = [self.model.jnt_dofadr[j]   for j in self._joint_ids]
        self._ee_site  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                            "attachment_site")
        self._obs_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            "obstacle")
        self._goal_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                             "goal_target")

        # Joint limits from model
        self._jnt_lo = np.array([self.model.jnt_range[j, 0] for j in self._joint_ids])
        self._jnt_hi = np.array([self.model.jnt_range[j, 1] for j in self._joint_ids])

        # Home position (roughly centred)
        self._q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        self.goal_pos = np.zeros(3)
        self.prev_dist = 0.0

    @property
    def state_dim(self) -> int:
        return 21  # 7 + 7 + 3 + 3 + 1

    @property
    def action_dim(self) -> int:
        return 7

    def _get_qpos(self):
        return np.array([self.data.qpos[a] for a in self._qpos_adr])

    def _get_qvel(self):
        return np.array([self.data.qvel[a] for a in self._qvel_adr])

    def _ee_pos(self):
        return self.data.site_xpos[self._ee_site].copy()

    def _obs_pos(self):
        return self.data.xpos[self._obs_body].copy()

    def _get_state(self) -> np.ndarray:
        qpos = self._get_qpos()
        qvel = self._get_qvel()
        ee = self._ee_pos()
        obs_p = self._obs_pos()
        rel_goal = self.goal_pos - ee
        rel_obs  = obs_p - ee
        obs_dist = _dist_point_to_cylinder_axis(ee, obs_p, OBSTACLE_HALFH) - OBSTACLE_RADIUS
        return np.concatenate([qpos, qvel, rel_goal, rel_obs, [obs_dist]]).astype(np.float32)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)

        # 1. Place obstacle first (in front of robot)
        obs_base = np.array([0.45, 0.0, 0.35])
        obs_offset = self.rng.uniform([-0.08, -0.12, -0.05], [0.08, 0.12, 0.05])
        obs_p = obs_base + obs_offset
        self.model.body_pos[self._obs_body] = obs_p

        # 2. Pick a goal on one side of the obstacle (positive-y side)
        goal_side = self.rng.choice([-1, 1])  # left or right of obstacle
        for _ in range(100):
            g = self.rng.uniform(_GOAL_BOUNDS_LO, _GOAL_BOUNDS_HI)
            d_obs = _dist_point_to_cylinder_axis(g, obs_p, OBSTACLE_HALFH)
            # Goal must be on the chosen side and clear of obstacle
            if d_obs > OBSTACLE_RADIUS + 0.08 and (g[1] - obs_p[1]) * goal_side > 0.06:
                break
        self.goal_pos = g.astype(np.float32)
        self.model.body_pos[self._goal_body] = self.goal_pos

        # 3. Init arm so EE ends up on the OPPOSITE side of obstacle from goal
        for _ in range(50):
            q0 = self._q_home + self.rng.uniform(-0.4, 0.4, 7)
            q0 = np.clip(q0, self._jnt_lo + 0.05, self._jnt_hi - 0.05)
            for i, a in enumerate(self._qpos_adr):
                self.data.qpos[a] = q0[i]
            self.data.ctrl[:7] = q0
            mujoco.mj_forward(self.model, self.data)
            ee = self._ee_pos()
            d_ee_obs = _dist_point_to_cylinder_axis(ee, obs_p, OBSTACLE_HALFH)
            # EE must be on opposite side of obstacle from goal, and clear of it
            if d_ee_obs > OBSTACLE_RADIUS + 0.05 and (ee[1] - obs_p[1]) * goal_side < -0.04:
                break

        mujoco.mj_forward(self.model, self.data)

        self.prev_dist = np.linalg.norm(self._ee_pos() - self.goal_pos)
        self._step_count = 0
        return self._get_state()

    def step(self, action: np.ndarray):
        # Joint-position delta (π₀.5 style)
        dq = np.clip(action, -1.0, 1.0) * DQ_SCALE
        q_current = self._get_qpos()
        q_target = np.clip(q_current + dq, self._jnt_lo, self._jnt_hi)

        # Set position actuator targets
        self.data.ctrl[:7] = q_target

        # Step physics
        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        ee = self._ee_pos()
        obs_p = self._obs_pos()
        dist = np.linalg.norm(ee - self.goal_pos)
        cyl_dist = _dist_point_to_cylinder_axis(ee, obs_p, OBSTACLE_HALFH)
        safety_margin = cyl_dist - OBSTACLE_RADIUS

        # Reward
        approach = self.prev_dist - dist
        reward = approach * 10.0
        reward += math.exp(-dist / 0.1) * 0.2  # proximity
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
