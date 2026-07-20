import sys
sys.argv = ['render_simulations.py', '--task', 'panda_pick_and_place']
sys.path.append('shaka_act')

import os
import pickle
import torch
import numpy as np
import h5py
import gymnasium as gym
import panda_gym
import cv2
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import make_policy, get_image

device = os.environ.get('DEVICE', 'cuda')
task = 'panda_pick_and_place'
checkpoint_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], task)
artifacts_dir = '/home/sanjay/.gemini/antigravity/brain/afe0fc21-6fac-4890-9e52-4e3fbf7eec4c'

def save_video_cv2(path, images, fps=20):
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for img in images:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()

def render_expert():
    print("\n--- Rendering Expert Trajectory ---")
    dataset_path = 'data/panda_pick_and_place/episode_1.hdf5'
    with h5py.File(dataset_path, 'r') as root:
        # Load front camera images (shape [150, 480, 640, 3] or similar)
        images = root['/observations/images/front'][()]
    
    # Save as video
    video_path = os.path.join(artifacts_dir, 'expert_episode_1.mp4')
    save_video_cv2(video_path, images, fps=20)
    print(f"Saved expert video to: {video_path}")

def render_policy():
    print("\n--- Rendering Policy Rollout ---")
    ckpt_path = os.path.join(checkpoint_dir, 'policy_best.ckpt')
    if not os.path.exists(ckpt_path):
        print(f"policy_best.ckpt not found at {ckpt_path}!")
        return

    # We need the stats for normalisation
    stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # Load true qpos from Episode 1 HDF5 to get starting state
    with h5py.File('data/panda_pick_and_place/episode_1.hdf5', 'r') as root:
        true_qpos = root['/observations/qpos'][()]

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

    policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.to(device)
    policy.eval()
    print("Loaded policy_best.ckpt successfully.")

    env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="rgb_array", render_width=640, render_height=480, max_episode_steps=250)
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
    
    rollout_frames = []
    success = False
    
    # RHC temporal aggregation parameters
    temporal_agg = POLICY_CONFIG['temporal_agg']
    query_frequency = POLICY_CONFIG['num_queries']
    all_time_actions = torch.zeros([250, 250 + query_frequency, POLICY_CONFIG['action_dim']]).to(device)
    
    for t in range(250):
        # Render frame
        img = env.render()
        rollout_frames.append(img)
        
        arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
        current_fingers_width = robot.get_fingers_width()
        qpos_numpy = np.concatenate([arm_joint_angles, [current_fingers_width], obs['achieved_goal'], obs['desired_goal']])
        qpos = pre_process(qpos_numpy)
        qpos_torch = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
        
        images_dict = {'front': img}
        curr_image = get_image(images_dict, POLICY_CONFIG['camera_names'], device)
        curr_image = torch.nn.functional.interpolate(curr_image.squeeze(0), size=(240, 320), mode='bilinear', align_corners=False).unsqueeze(0)
        
        with torch.inference_mode():
            if temporal_agg:
                all_actions = policy(qpos_torch, curr_image)
                all_time_actions[[t], t:t+query_frequency] = all_actions
                actions_for_curr_step = all_time_actions[max(0, t - query_frequency + 1) : t + 1, t]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                if t % query_frequency == 0:
                    all_actions = policy(qpos_torch, curr_image)
                raw_action = all_actions[0, t % query_frequency]
                
        # Postprocess action
        raw_action_numpy = raw_action.cpu().numpy()
        if temporal_agg:
            raw_action_numpy = raw_action_numpy[0]
        action = raw_action_numpy * stats['action_std'] + stats['action_mean']
        action = action[:8]
        action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        is_success = dist < 0.05
        if is_success:
            success = True
            print(f"Success achieved at step {t}!")
            # Keep rendering some steps after success to show placement stability
            for _ in range(10):
                img_suc = env.render()
                rollout_frames.append(img_suc)
            break
            
    final_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
    print(f"Final placement error: {final_dist*100:.2f} cm")
    print(f"Success: {success}")
    
    # Save rollout video
    video_path = os.path.join(artifacts_dir, 'rollout_episode_1.mp4')
    save_video_cv2(video_path, rollout_frames, fps=20)
    print(f"Saved rollout video to: {video_path}")
    env.close()

if __name__ == "__main__":
    render_expert()
    render_policy()
