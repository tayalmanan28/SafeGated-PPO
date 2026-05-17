"""
Render drone GIF using MuJoCo drone_scene.xml with the trained state-based policy.
Produces isometric, top-down, and egocentric views.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import mujoco
from PIL import Image
from models import StatePolicy
from drone.env import DroneNavEnv, OBSTACLE_RADIUS, GOAL_RADIUS

# ── Config ──────────────────────────────────────────────────────────
SCENE_XML = os.path.join(os.path.dirname(__file__), "drone_scene.xml")
CKPT_PATH = "/data/t-manantayal/SafeGatedVLA/state_gated_ppo_v5/drone/drone_best.pth"
OUT_DIR = "/data/t-manantayal/SafeGatedVLA/state_gated_ppo_v5/drone"
RENDER_SIZE = 640
MAX_STEPS = 150

# ── Load policy ─────────────────────────────────────────────────────
ckpt = torch.load(CKPT_PATH, map_location="cpu")
policy = StatePolicy(12, 4, hidden=256)
policy.load_state_dict(ckpt["policy"])
policy.eval()

# ── Find a good episode ─────────────────────────────────────────────
best = None
for seed in range(2000):
    env = DroneNavEnv(seed=seed)
    state = env.reset()
    start = env.pos.copy()
    goal = env.goal.copy()
    obs = env.obs_pos.copy()
    
    # Everything within ±1.8 (visible, away from walls)
    if np.any(np.abs(start[:2]) > 1.8) or np.any(np.abs(goal[:2]) > 1.8) or np.any(np.abs(obs[:2]) > 1.8):
        continue
    
    # Obstacle in the straight-line path
    sg = goal - start
    sg_len = np.linalg.norm(sg)
    sg_hat = sg / (sg_len + 1e-8)
    so = obs - start
    proj = np.dot(so, sg_hat)
    perp = np.linalg.norm(so - proj * sg_hat)
    if perp > 0.3 or proj < 0.3 * sg_len or proj > 0.7 * sg_len:
        continue
    
    # Run policy
    traj = [env.pos.copy()]
    done = False
    cost = 0
    reached = False
    min_d = 999
    while not done:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, _ = policy(s_t)
            action = mean.squeeze(0).numpy()
        state, _, done, info = env.step(action)
        traj.append(env.pos.copy())
        cost += info["cost"]
        reached = info.get("reached", False)
        min_d = min(min_d, np.linalg.norm(env.pos - obs) - OBSTACLE_RADIUS)
    
    if not reached or cost > 0 or min_d < 0.05:
        continue
    
    # Measure avoidance deviation
    traj = np.array(traj)
    devs = [np.linalg.norm((p - start) - np.dot(p - start, sg_hat) * sg_hat) for p in traj]
    max_dev = max(devs)
    
    if max_dev > 0.4:
        final_dist = np.linalg.norm(traj[-1] - goal)
        if final_dist < GOAL_RADIUS:
            print(f"Seed {seed}: steps={len(traj)}, clearance={min_d:.3f}, dev={max_dev:.2f}, final_dist={final_dist:.3f}")
            print(f"  Start={start}")
            print(f"  Goal={goal}")  
            print(f"  Obstacle={obs}")
            best = {"traj": traj, "start": start, "goal": goal, "obs": obs}
            break

if best is None:
    print("ERROR: no good episode found")
    sys.exit(1)

trajectory = best["traj"]
goal_pos = best["goal"]
obs_pos = best["obs"]
start_pos = best["start"]

# ── Setup MuJoCo ────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=RENDER_SIZE, width=RENDER_SIZE)

cf_jnt_qadr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cf2_freejoint")]
obs_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
goal_green_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_green")

def set_drone_pos(pos, look_at=None):
    """Set drone position and orientation (facing look_at if provided)."""
    q = cf_jnt_qadr
    data.qpos[q:q+3] = pos
    if look_at is not None:
        d = look_at - pos
        yaw = np.arctan2(d[1], d[0])
        data.qpos[q+3:q+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    else:
        data.qpos[q+3:q+7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

# Place obstacle and goal in MuJoCo scene
model.body_pos[obs_bid] = obs_pos
model.body_pos[goal_green_bid] = goal_pos
mujoco.mj_forward(model, data)

# Hide the other goal markers
for color in ("red", "blue"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"goal_{color}")
    model.body_pos[bid] = [0, 0, -5]

# ── 1. Isometric view ──────────────────────────────────────────────
print("Rendering isometric view...")
frames = []
mid = (start_pos + goal_pos) / 2
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = mid
cam.distance = 5.5
cam.azimuth = -50
cam.elevation = -25

for i, pos in enumerate(trajectory):
    next_pos = trajectory[min(i+1, len(trajectory)-1)]
    set_drone_pos(pos, look_at=next_pos)
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())

for _ in range(15):
    frames.append(frames[-1].copy())

path = os.path.join(OUT_DIR, "drone_mujoco_isometric.gif")
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=80, loop=0)
print(f"  Saved: {path} ({os.path.getsize(path)/1024:.0f} KB)")

# ── 2. Egocentric (onboard camera) view ───────────────────────────
print("Rendering egocentric view...")
frames = []
for i, pos in enumerate(trajectory):
    # Orient drone toward goal so ego_cam sees obstacle & goal ahead
    look_target = goal_pos  # always face the goal
    set_drone_pos(pos, look_at=look_target)
    renderer.update_scene(data, camera="ego_cam")
    frames.append(renderer.render().copy())

for _ in range(15):
    frames.append(frames[-1].copy())

path = os.path.join(OUT_DIR, "drone_mujoco_egocentric.gif")
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=80, loop=0)
print(f"  Saved: {path} ({os.path.getsize(path)/1024:.0f} KB)")

print("\nDone!")
