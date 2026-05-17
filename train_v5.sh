#!/bin/bash
# v5: Fixes 1+2+3 — More iterations (1000), higher β (start=10, max=50), lower LR (1e-4)
# Goal: near 100% safety without sacrificing success

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl

SAVE=/data/t-manantayal/SafeGatedVLA/state_gated_ppo_v5
mkdir -p $SAVE

COMMON="--iterations 1000 --lr 1e-4 --safety_coef 10.0 --coef_lr 0.1 --coef_min 2.0 --coef_max 50.0 --cost_limit 1.0 --critic_warmup 50 --ppo_epochs 8 --num_envs 64"

echo "=== Training Turtlebot v5 ==="
python train.py --robot turtlebot $COMMON --save_dir $SAVE/turtlebot 2>&1 | tee $SAVE/turtlebot.log

echo "=== Training Drone v5 ==="
python train.py --robot drone $COMMON --save_dir $SAVE/drone 2>&1 | tee $SAVE/drone.log

echo "=== Training FR3 v5 ==="
python train.py --robot fr3 $COMMON --save_dir $SAVE/fr3 2>&1 | tee $SAVE/fr3.log

echo "=== All done ==="
