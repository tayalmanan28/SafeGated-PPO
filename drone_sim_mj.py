"""
MuJoCo Crazyflie 2.1 simulator + cascaded geometric controller.

Exposes a TurtleBot-style interface so the OmniVLA-style policy layer
sees a familiar (v, omega) -> 4D body-velocity command:

    sim.step(vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)

These are world-frame velocity commands.  Internally we run:

    1. PD velocity tracker -> desired body acceleration -> thrust + tilt
    2. Geometric attitude controller (roll, pitch, yaw) -> body moments
    3. Direct write to MuJoCo motor channels (body_thrust, x_moment,
       y_moment, z_moment) -- same actuator layout as Menagerie cf2.xml.

This is the standard cascaded geometric controller used by Lee et al.
("Geometric tracking control of a quadrotor UAV on SE(3)", 2010); we
linearize roll/pitch around hover to make it numerically tame.

The wrapper hides all this from the VLA, exactly as the omnivla
TurtleBotSim wrapper hides the diff-drive kinematics.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np


# Bitcraze Crazyflie 2.1 physical constants
MASS = 0.027            # kg
GRAVITY = 9.81          # m/s^2
HOVER_THRUST = MASS * GRAVITY        # 0.265 N  (ctrl ~0.265 / 0.35 ~= 0.757)
MAX_THRUST = 0.35       # N (matches ctrlrange of body_thrust)

# Gains (hand-tuned, conservative)
@dataclass
class ControllerGains:
    # Outer-loop: linear velocity -> body accel (m/s -> m/s^2).
    kp_vxy: float = 4.0
    kp_vz:  float = 8.0
    # Inner-loop: attitude PD -> body moment (normalized ctrl in [-1, 1]).
    # With gear ~ 1e-3 N*m and J ~ 2.4e-5 kg*m^2, ctrl=1 gives ~42 rad/s^2,
    # giving an attitude bandwidth of ~10 Hz with these gains.
    kp_att: float = 12.0
    kd_att: float = 1.2
    # Yaw rate (rad/s) -> z_moment ctrl
    kp_yaw_rate: float = 0.5


class DroneSim:
    """MuJoCo Crazyflie 2.1 simulator with a cascaded geometric controller.

    The public interface is intentionally analogous to TurtleBotSim from the
    omnivla branch: a single step(.) that takes a high-level command and
    advances physics for N substeps.
    """

    def __init__(self,
                 scene_xml: str,
                 width: int = 640,
                 height: int = 480,
                 gains: Optional[ControllerGains] = None):
        os.environ.setdefault("MUJOCO_GL", "egl")
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.width, self.height = width, height
        self.gains = gains or ControllerGains()

        # Cache body / actuator / sensor ids
        self.cf_bid       = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,     "cf2")
        self.cf_joint_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cf2_freejoint")]
        self.cf_joint_vadr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cf2_freejoint")]
        self.act_thrust = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "body_thrust")
        self.act_xm     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "x_moment")
        self.act_ym     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "y_moment")
        self.act_zm     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "z_moment")

        # Goal body ids -- the room has goal_red / goal_green / goal_blue
        self.goal_ids = {
            color: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"goal_{color}")
            for color in ("red", "green", "blue")
        }
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ state
    def reset(self, x: float = 0.0, y: float = 0.0, z: float = 0.4):
        mujoco.mj_resetData(self.model, self.data)
        q = self.cf_joint_qadr
        self.data.qpos[q + 0] = x
        self.data.qpos[q + 1] = y
        self.data.qpos[q + 2] = z
        self.data.qpos[q + 3] = 1.0           # quat w
        self.data.qpos[q + 4:q + 7] = 0.0     # quat x,y,z
        v = self.cf_joint_vadr
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_pose(self):
        """Return (x, y, z, yaw_radians)."""
        x, y, z = (float(self.data.xpos[self.cf_bid][i]) for i in range(3))
        # body_quat = (w, x, y, z) in MuJoCo
        q = self.data.xquat[self.cf_bid]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                         1 - 2 * (q[2] * q[2] + q[3] * q[3]))
        return x, y, z, yaw

    def get_lin_vel(self):
        v = self.cf_joint_vadr
        return float(self.data.qvel[v + 0]), float(self.data.qvel[v + 1]), float(self.data.qvel[v + 2])

    def get_ang_vel_body(self):
        v = self.cf_joint_vadr
        return float(self.data.qvel[v + 3]), float(self.data.qvel[v + 4]), float(self.data.qvel[v + 5])

    def get_roll_pitch(self):
        q = self.data.xquat[self.cf_bid]
        # roll  = atan2(2(wx + yz), 1 - 2(x^2 + y^2))
        # pitch = asin (2(wy - zx))
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        roll  = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return roll, pitch

    # ----------------------------------------------------------- controller
    def step(self,
             vx_cmd: float,
             vy_cmd: float,
             vz_cmd: float,
             yaw_rate_cmd: float,
             n_substeps: int = 5):
        """Track a world-frame velocity command for `n_substeps` physics steps.

        Args:
            vx_cmd, vy_cmd, vz_cmd: desired world-frame linear velocity (m/s).
            yaw_rate_cmd: desired body yaw rate (rad/s).
        """
        g = self.gains
        for _ in range(n_substeps):
            vx, vy, vz = self.get_lin_vel()
            roll, pitch = self.get_roll_pitch()
            wx, wy, wz  = self.get_ang_vel_body()
            _, _, _, yaw = self.get_pose()

            # --- Outer loop: velocity -> desired body accel (world frame).
            ax_world = g.kp_vxy * (vx_cmd - vx)
            ay_world = g.kp_vxy * (vy_cmd - vy)
            az_des   = g.kp_vz  * (vz_cmd - vz)

            # Rotate (ax, ay) into body frame so the tilt commands hold
            # under arbitrary yaw.
            c, s = math.cos(yaw), math.sin(yaw)
            ax_body =  c * ax_world + s * ay_world
            ay_body = -s * ax_world + c * ay_world

            # Desired thrust along world +z (small-angle approx).
            thrust = MASS * (GRAVITY + az_des)
            thrust = float(np.clip(thrust, 0.0, MAX_THRUST))

            # Desired roll/pitch in body frame (small-angle).
            #   ax_body ~= +g * pitch ;  ay_body ~= -g * roll
            pitch_des =  ax_body / GRAVITY
            roll_des  = -ay_body / GRAVITY
            pitch_des = float(np.clip(pitch_des, -0.35, 0.35))
            roll_des  = float(np.clip(roll_des,  -0.35, 0.35))

            # --- Inner loop: attitude PD -> body moments (normalized ctrl).
            # The motor "gear" in the XML is NEGATIVE (-1e-5), so ctrl=+1
            # produces a NEGATIVE body moment.  We compute the PD output in
            # "moment-space" and then negate when writing to ctrl.
            xm_mom = g.kp_att * (roll_des  - roll)  - g.kd_att * wx
            ym_mom = g.kp_att * (pitch_des - pitch) - g.kd_att * wy
            zm_mom = g.kp_yaw_rate * (yaw_rate_cmd - wz)

            self.data.ctrl[self.act_thrust] = thrust
            self.data.ctrl[self.act_xm]     = float(np.clip(-xm_mom, -1.0, 1.0))
            self.data.ctrl[self.act_ym]     = float(np.clip(-ym_mom, -1.0, 1.0))
            self.data.ctrl[self.act_zm]     = float(np.clip(-zm_mom, -1.0, 1.0))

            mujoco.mj_step(self.model, self.data)

    # ----------------------------------------------------------- rendering
    def get_ego_image(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="ego_cam")
        return self.renderer.render().copy()

    def get_overview_image(self, camera: str = "isometric") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
