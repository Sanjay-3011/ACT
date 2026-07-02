import os
import sys
sys.path.append('shaka_act')
import pickle
import numpy as np
import gymnasium as gym
import panda_gym
import torch
from copy import deepcopy

# Import configuration
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import make_policy, get_image

def debug_policy():
    device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    task = 'panda_pick_and_place'
    
    # Paths
    checkpoint_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], task)
    ckpt_path = os.path.join(checkpoint_dir, 'policy_last.ckpt')
    stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
    
    # Load stats
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    # Load policy
    policy_config = deepcopy(POLICY_CONFIG)
    policy_config['device'] = device
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    policy.to(device)
    policy.eval()
    print(f"Loaded policy from {ckpt_path}")
    
    env = gym.make(
        "PandaPickAndPlace-v3", 
        control_type="joints", 
        render_mode="rgb_array", 
        render_width=640, 
        render_height=480,
        max_episode_steps=100
    )
    
    query_frequency = POLICY_CONFIG['num_queries']
    
    for ep in range(2):
        obs, info = env.reset()
        robot = env.unwrapped.robot
        print(f"\n--- Episode {ep} ---")
        print("Cube initial pos:", obs['achieved_goal'])
        print("Target pos:", obs['desired_goal'])
        
        all_actions = None
        success = False
        
        for t in range(50):
            img = env.render()
            
            arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
            current_fingers_width = robot.get_fingers_width()
            qpos_numpy = np.concatenate([arm_joint_angles, [current_fingers_width]])
            
            qpos = pre_process(qpos_numpy)
            qpos_torch = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
            
            images_dict = {'front': img}
            curr_image = get_image(images_dict, POLICY_CONFIG['camera_names'], device)
            
            with torch.inference_mode():
                if t % query_frequency == 0:
                    all_actions = policy(qpos_torch, curr_image)
                raw_action = all_actions[0, t % query_frequency]
                
            action = post_process(raw_action.cpu().numpy())
            action = np.clip(action, -1.0, 1.0)
            
            if t < 10:
                print(f"Step {t}:")
                print(f"  qpos: {qpos_numpy}")
                print(f"  predicted action: {action}")
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info.get('is_success', False) or terminated:
                success = True
                print(f"SUCCESS at step {t}!")
                break
                
            if terminated or truncated:
                print(f"Failed at step {t}")
                break
                
    env.close()

if __name__ == "__main__":
    debug_policy()
