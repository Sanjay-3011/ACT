import sys
sys.argv = ['play_policy_gui.py', '--task', 'panda_pick_and_place']
sys.path.append('shaka_act')

import os
import pickle
import torch
import numpy as np
import h5py
import gymnasium as gym
import panda_gym
import time
import pybullet as p
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import make_policy, get_image

device = os.environ.get('DEVICE', 'cuda')
task = 'panda_pick_and_place'
checkpoint_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], task)

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

# Create env in GUI mode
env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="human", render_width=640, render_height=480, max_episode_steps=150)
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

temporal_agg = POLICY_CONFIG['temporal_agg']
query_frequency = POLICY_CONFIG['num_queries']
all_time_actions = torch.zeros([150, 150 + query_frequency, POLICY_CONFIG['action_dim']]).to(device)

print("Playing trained policy rollout in GUI...")
for t in range(150):
    # Direct PyBullet query to bypass "direct mode only" check inside sim.render
    target_pos_cam = env.unwrapped.render_target_position
    view_matrix = sim.physics_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target_pos_cam,
        distance=env.unwrapped.render_distance,
        yaw=env.unwrapped.render_yaw,
        pitch=env.unwrapped.render_pitch,
        roll=env.unwrapped.render_roll,
        upAxisIndex=2,
    )
    proj_matrix = sim.physics_client.computeProjectionMatrixFOV(
        fov=60, aspect=float(640) / 480, nearVal=0.1, farVal=100.0
    )
    (_, _, rgba, _, _) = sim.physics_client.getCameraImage(
        width=640,
        height=480,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        shadow=True,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgba = np.array(rgba, dtype=np.uint8).reshape((480, 640, 4))
    img = rgba[..., :3]
    
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
    time.sleep(0.02) # Control playback speed (50 Hz)
    
    dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
    is_success = dist < 0.05
    if is_success:
        print(f"Success achieved at step {t}!")
        time.sleep(1.0) # Pause to show success
        break
        
final_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
print(f"Final placement error: {final_dist*100:.2f} cm")
print(f"Success: {is_success}")
env.close()
