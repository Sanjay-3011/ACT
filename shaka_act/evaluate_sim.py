import sys
import argparse

# Parse our custom arguments first and remove them from sys.argv
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--n_episodes', type=int, default=50)
parser.add_argument('--max_steps', type=int, default=120)
args, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args

import os
import pickle
import numpy as np
import gymnasium as gym
import panda_gym
import torch
import mediapy as media
from copy import deepcopy
from time import time
from tqdm import tqdm

# Import configuration
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import make_policy, get_image

def evaluate_policy(n_episodes=50, max_steps=120):
    device = os.environ['DEVICE']
    task = 'panda_pick_and_place'
    
    # Paths
    checkpoint_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], task)
    ckpt_path = os.path.join(checkpoint_dir, 'policy_last.ckpt')
    stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Ensure training completes first.")
        
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
    
    # Init environment
    env = gym.make(
        "PandaPickAndPlace-v3", 
        control_type="joints", 
        render_mode="rgb_array", 
        render_width=640, 
        render_height=480,
        max_episode_steps=max_steps
    )
    
    successes = 0
    episode_lengths = []
    recorded_count = 0
    videos_dir = "evaluation_videos"
    os.makedirs(videos_dir, exist_ok=True)
    
    query_frequency = POLICY_CONFIG['num_queries']
    
    print("\nStarting evaluation of 50 episodes...")
    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        # Helper function to run a single rollout with a specific seed and rendering mode
        def run_rollout(seed, record_video):
            obs, info = env.reset(seed=seed)
            robot = env.unwrapped.robot
            
            frames = []
            success = False
            ep_len = 0
            all_actions = None
            
            temporal_agg = POLICY_CONFIG['temporal_agg']
            if temporal_agg:
                all_time_actions = torch.zeros([max_steps, max_steps + query_frequency, POLICY_CONFIG['state_dim']]).to(device)
            
            for t in range(max_steps):
                ep_len += 1
                
                # Get current camera image (only render if we need to record a video or query the policy)
                if record_video or temporal_agg or (t % query_frequency == 0):
                    img = env.render()
                    if record_video:
                        frames.append(img)
                else:
                    img = None
                
                # Query joint states and task goals (Goal Conditioning)
                arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
                current_fingers_width = robot.get_fingers_width()
                cube_pos = obs['achieved_goal']
                target_pos = obs['desired_goal']
                qpos_numpy = np.concatenate([arm_joint_angles, [current_fingers_width], cube_pos, target_pos])
                
                # Preprocess inputs
                qpos = pre_process(qpos_numpy)
                qpos_torch = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                
                # Policy query
                with torch.inference_mode():
                    images_dict = {'front': img}
                    curr_image = get_image(images_dict, POLICY_CONFIG['camera_names'], device)
                    
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
                action = post_process(raw_action_numpy)
                action = action[:8]  # Slice action to keep only original joints + gripper
                action = np.clip(action, -1.0, 1.0)
                
                # Environment step
                obs, reward, terminated, truncated, info = env.step(action)
                
                if info.get('is_success', False) or terminated:
                    success = True
                    if record_video:
                        # Record final success frame
                        frames.append(env.render())
                    break
                    
                if terminated or truncated:
                    break
            
            return success, ep_len, frames

        # Run in fast mode first
        success, ep_len, _ = run_rollout(seed=ep, record_video=False)
        
        episode_lengths.append(ep_len)
        if success:
            successes += 1
            # If we need a video, re-run in slow mode to capture the frames
            if recorded_count < 5:
                _, _, frames = run_rollout(seed=ep, record_video=True)
                video_path = os.path.join(videos_dir, f'success_episode_{ep}.mp4')
                media.write_video(video_path, frames, fps=30)
                print(f"\nSaved successful rollout video to {video_path}")
                recorded_count += 1
                
    env.close()
    
    # Calculate metrics
    success_rate = (successes / n_episodes) * 100.0
    avg_len = np.mean(episode_lengths)
    std_len = np.std(episode_lengths)
    
    print("\n" + "="*40)
    print("Evaluation Metrics:")
    print(f"Success Rate: {success_rate:.2f}% ({successes}/{n_episodes})")
    print(f"Average Episode Length: {avg_len:.2f} steps")
    print(f"Standard Deviation of Episode Length: {std_len:.2f} steps")
    print("="*40 + "\n")
    
    # Save metrics to file
    metrics_path = "evaluation_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("ACT Evaluation Metrics\n")
        f.write("======================\n")
        f.write(f"Success Rate: {success_rate:.2f}% ({successes}/{n_episodes})\n")
        f.write(f"Average Episode Length: {avg_len:.2f} steps\n")
        f.write(f"Standard Deviation: {std_len:.2f} steps\n")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate_policy(n_episodes=args.n_episodes, max_steps=args.max_steps)
