import sys
sys.argv = ['test_rollout_with_ckpt.py', '--task', 'panda_pick_and_place']
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

# Try loading different checkpoints and running rollout
checkpoints = [
    'policy_last_klw10.ckpt',
    'policy_last.ckpt',
    'policy_epoch_0_seed_42.ckpt'
]

# We need the stats for normalisation
stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

# Load true qpos from Episode 1 HDF5
with h5py.File('data/panda_pick_and_place/episode_1.hdf5', 'r') as root:
    true_qpos = root['/observations/qpos'][()]

pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

for ckpt_name in checkpoints:
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_name} not found.")
        continue
    
    print(f"\n--- Testing {ckpt_name} ---")
    try:
        policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
        policy.load_state_dict(torch.load(ckpt_path, map_location=device))
        policy.to(device)
        policy.eval()
        print(f"Loaded {ckpt_name} successfully.")
    except Exception as e:
        print(f"Failed to load {ckpt_name}: {e}")
        continue

    # Init env
    env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="rgb_array", max_episode_steps=150)
    obs, info = env.reset(seed=42)
    robot = env.unwrapped.robot
    sim = env.unwrapped.sim
    
    # Reset robot and objects to the exact step 0 state from Episode 1
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
    
    success = False
    for t in range(150):
        arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
        current_fingers_width = robot.get_fingers_width()
        qpos_numpy = np.concatenate([arm_joint_angles, [current_fingers_width], obs['achieved_goal'], obs['desired_goal']])
        qpos = pre_process(qpos_numpy)
        qpos_torch = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
        
        img = env.render()
        images_dict = {'front': img}
        curr_image = get_image(images_dict, POLICY_CONFIG['camera_names'], device)
        
        with torch.inference_mode():
            all_actions = policy(qpos_torch, curr_image)
            # Take index 0 action (no temporal aggregation for simplicity of test)
            action = all_actions[0, 0].cpu().numpy()
            
        # Postprocess action
        action_post = action * stats['action_std'] + stats['action_mean']
        action_post = action_post[:8]
        action_post = np.clip(action_post, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = env.step(action_post)
        
        # Check success criteria
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        is_success = dist < 0.05
        if is_success:
            success = True
            print(f"Success achieved at step {t}!")
            break
            
    final_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
    print(f"Final placement error: {final_dist*100:.2f} cm")
    print(f"Success: {success}")
    env.close()
