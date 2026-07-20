import sys
sys.argv = ['compare_rollouts.py', '--task', 'panda_pick_and_place']
sys.path.append('shaka_act')

import os
import pickle
import torch
import numpy as np
import h5py
import gymnasium as gym
import panda_gym
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import make_policy, get_image

device = os.environ.get('DEVICE', 'cuda')
task = 'panda_pick_and_place'
checkpoint_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], task)

# We need the stats for normalisation
stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

# Load true qpos from Episode 1 HDF5 to get starting state
with h5py.File('data/panda_pick_and_place/episode_1.hdf5', 'r') as root:
    true_qpos = root['/observations/qpos'][()]

pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

ckpt_path = os.path.join(checkpoint_dir, 'policy_best.ckpt')
policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
policy.load_state_dict(torch.load(ckpt_path, map_location=device))
policy.to(device)
policy.eval()

# Let's run a short 5-step rollout using BOTH styles and print values

def run_style_a():
    print("\n--- Style A (overfit_one_episode.py style) ---")
    env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="rgb_array", render_width=640, render_height=480, max_episode_steps=150)
    obs, info = env.reset(seed=42)
    robot = env.unwrapped.robot
    sim = env.unwrapped.sim
    
    qpos_init = true_qpos[0]
    arm_angles = qpos_init[0:7]
    gripper_width = qpos_init[7]
    cube_pos = qpos_init[8:11]
    target_pos = qpos_init[11:14]
    
    robot.set_joint_angles(arm_angles)
    for finger_idx in robot.fingers_indices:
        sim.set_joint_angle(robot.body_name, finger_idx, gripper_width / 2.0)
    sim.set_base_pose("object", position=cube_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    sim.set_base_pose("target", position=target_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    env.unwrapped.task.goal = target_pos.copy()
    sim.step()
    
    obs = env.unwrapped._get_obs()
    obs['achieved_goal'] = cube_pos.copy()
    obs['desired_goal'] = target_pos.copy()
    
    for t in range(5):
        img = env.render()
        current_ee_pos = robot.get_ee_position()
        current_fingers_width = robot.get_fingers_width()
        current_arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
        current_cube_pos = obs['achieved_goal']
        current_target_pos = obs['desired_goal']
        
        qpos_numpy = np.concatenate([current_arm_joint_angles, [current_fingers_width], current_cube_pos, current_target_pos])
        qpos = pre_process(qpos_numpy)
        qpos_torch = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
        
        images_dict = {'front': img}
        curr_image = get_image(images_dict, POLICY_CONFIG['camera_names'], device)
        curr_image = torch.nn.functional.interpolate(curr_image.squeeze(0), size=(240, 320), mode='bilinear', align_corners=False).unsqueeze(0)
        
        with torch.inference_mode():
            all_actions = policy(qpos_torch, curr_image)
            raw_action = all_actions[0, 0]
            
        action = raw_action.cpu().numpy()
        action = action * stats['action_std'] + stats['action_mean']
        action = action[:8]
        action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {t} | qpos0: {qpos_numpy[0]:.4f} | act0: {action[0]:.4f}")
    env.close()

def run_style_b():
    print("\n--- Style B (render_simulations.py style) ---")
    env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="rgb_array", render_width=640, render_height=480, max_episode_steps=250)
    obs, info = env.reset(seed=42)
    robot = env.unwrapped.robot
    sim = env.unwrapped.sim
    
    qpos_init = true_qpos[0]
    arm_angles = qpos_init[0:7]
    gripper_width = qpos_init[7]
    cube_pos = qpos_init[8:11]
    target_pos = qpos_init[11:14]
    
    robot.set_joint_angles(arm_angles)
    for finger_idx in robot.fingers_indices:
        sim.set_joint_angle(robot.body_name, finger_idx, gripper_width / 2.0)
    sim.set_base_pose("object", position=cube_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    sim.set_base_pose("target", position=target_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    env.unwrapped.task.goal = target_pos.copy()
    sim.step()
    
    obs = env.unwrapped._get_obs()
    obs['achieved_goal'] = cube_pos.copy()
    obs['desired_goal'] = target_pos.copy()
    
    for t in range(5):
        img = env.render()
        arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
        current_fingers_width = robot.get_fingers_width()
        qpos_numpy = np.concatenate([arm_joint_angles, [current_fingers_width], obs['achieved_goal'], obs['desired_goal']])
        qpos = pre_process(qpos_numpy)
        qpos_torch = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
        
        images_dict = {'front': img}
        curr_image = get_image(images_dict, POLICY_CONFIG['camera_names'], device)
        curr_image = torch.nn.functional.interpolate(curr_image.squeeze(0), size=(240, 320), mode='bilinear', align_corners=False).unsqueeze(0)
        
        with torch.inference_mode():
            all_actions = policy(qpos_torch, curr_image)
            raw_action = all_actions[0, 0]
            
        action = raw_action.cpu().numpy()
        action = action * stats['action_std'] + stats['action_mean']
        action = action[:8]
        action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {t} | qpos0: {qpos_numpy[0]:.4f} | act0: {action[0]:.4f}")
    env.close()

if __name__ == "__main__":
    run_style_a()
    run_style_b()
