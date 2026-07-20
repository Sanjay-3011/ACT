import sys
sys.argv = ['test_determinism.py', '--task', 'panda_pick_and_place']
sys.path.append('shaka_act')

import os
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import panda_gym
import h5py
import pickle
import random
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import make_policy, get_image

def test_run(seed=42):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task = 'panda_pick_and_place'
    dataset_dir = os.path.join(TASK_CONFIG['dataset_dir'], task)
    checkpoint_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], task)

    with open(os.path.join(checkpoint_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)

    policy_config = POLICY_CONFIG.copy()
    policy_config['device'] = device
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)
    policy.eval()

    best_ckpt_path = os.path.join(checkpoint_dir, 'policy_best.ckpt')
    policy.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    # Load episode 1 data to set initial state
    with h5py.File(os.path.join(dataset_dir, "episode_1.hdf5"), 'r') as root:
        true_qpos = root['observations/qpos'][()]

    env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="rgb_array", render_width=640, render_height=480, max_episode_steps=150)
    
    # Reset with seed!
    obs, info = env.reset(seed=seed)
    robot = env.unwrapped.robot
    sim = env.unwrapped.sim

    # Reset robot and objects to the exact step 0 state
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

    # Override stale observation values
    obs['achieved_goal'] = cube_pos.copy()
    obs['desired_goal'] = target_pos.copy()

    # Pre-process
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    
    # Step 0 prediction
    img = env.render()
    current_ee_pos = robot.get_ee_position()
    current_fingers_width = robot.get_fingers_width()
    current_arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
    current_cube_pos = obs['achieved_goal']
    current_target_pos = obs['desired_goal']
    
    qpos_numpy = np.concatenate([current_arm_joint_angles, [current_fingers_width], current_cube_pos, current_target_pos])
    qpos = pre_process(qpos_numpy)
    qpos_torch = torch.from_numpy(qpos).float().to(device).unsqueeze(0)

    with torch.inference_mode():
        images_dict = {'front': img}
        curr_image = get_image(images_dict, POLICY_CONFIG['camera_names'], device)
        # Resize image
        curr_image = F.interpolate(curr_image.squeeze(0), size=(240, 320), mode='bilinear', align_corners=False).unsqueeze(0)
        
        all_actions = policy(qpos_torch, curr_image)
    
    env.close()
    return all_actions[0, 0].cpu().numpy(), img, qpos_numpy

# Run 1 with seed 42
act1, img1, qp1 = test_run(42)
# Run 2 with seed 42
act2, img2, qp2 = test_run(42)
# Run 3 with seed 100
act3, img3, qp3 = test_run(100)

print("Difference between Run 1 and Run 2 (same seed):", np.linalg.norm(act1 - act2))
print("Difference between Run 1 and Run 3 (diff seed):", np.linalg.norm(act1 - act3))
print("Image 1 vs Image 2 sum difference:", np.sum(np.abs(img1 - img2)))
print("Image 1 vs Image 3 sum difference:", np.sum(np.abs(img1 - img3)))
print("qpos 1 vs qpos 2 norm difference:", np.linalg.norm(qp1 - qp2))
print("qpos 1 vs qpos 3 norm difference:", np.linalg.norm(qp1 - qp3))
print("qpos 1:", qp1)
print("qpos 3:", qp3)
