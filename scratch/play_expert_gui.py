import gymnasium as gym
import panda_gym
import numpy as np
import h5py
import time

env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="human", render_width=640, render_height=480, max_episode_steps=150)
obs, info = env.reset(seed=42)

with h5py.File('data/panda_pick_and_place/episode_1.hdf5', 'r') as root:
    actions = root['action'][()]
    qpos = root['observations/qpos'][()]

# Reset objects and robot to step 0
robot = env.unwrapped.robot
sim = env.unwrapped.sim
qpos_init = qpos[0]
robot.set_joint_angles(qpos_init[0:7])
for finger_idx in robot.fingers_indices:
    sim.set_joint_angle(robot.body_name, finger_idx, qpos_init[7] / 2.0)
sim.set_base_pose("object", position=qpos_init[8:11], orientation=np.array([1.0, 0.0, 0.0, 0.0]))
sim.set_base_pose("target", position=qpos_init[11:14], orientation=np.array([1.0, 0.0, 0.0, 0.0]))
env.unwrapped.task.goal = qpos_init[11:14].copy()
sim.step()

print("Playing expert trajectory in GUI...")
for t, action in enumerate(actions):
    env.step(action[:8])
    time.sleep(0.05) # Slow down to real-time (20 Hz)
    
    # Optional print to check success
    cube_pos = sim.get_base_position("object")
    target_pos = sim.get_base_position("target")
    dist = np.linalg.norm(cube_pos - target_pos)
    if dist < 0.05:
        print(f"Success achieved at step {t}!")
        time.sleep(1.0) # Pause to show success
        break
env.close()
