import sys
sys.argv = ['overfit_one_episode.py', '--task', 'panda_pick_and_place']
sys.path.append('shaka_act')

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import gymnasium as gym
import panda_gym
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import make_policy, get_image, get_norm_stats
from torch.utils.data import DataLoader
import torch.nn.functional as F
from copy import deepcopy

class PreloadedEpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super().__init__()
        self.episode_ids = episode_ids
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        
        # Load all episodes in RAM
        self.data_list = []
        for episode_id in episode_ids:
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                is_sim = root.attrs['sim']
                original_action_shape = root['/action'].shape
                episode_len = original_action_shape[0]
                
                action_full = root['/action'][()]
                active_len = len(action_full)
                for idx in range(len(action_full) - 1, -1, -1):
                    if np.any(action_full[idx] != 0):
                        active_len = idx + 1
                        break
                
                images_all = {}
                for cam_name in camera_names:
                    images_all[cam_name] = root[f'/observations/images/{cam_name}'][()]
                
                qpos_all = root['/observations/qpos'][()]
                action_all = root['/action'][()]
                
            self.data_list.append({
                'is_sim': is_sim,
                'original_action_shape': original_action_shape,
                'episode_len': episode_len,
                'active_len': active_len,
                'images_all': images_all,
                'qpos_all': qpos_all,
                'action_all': action_all
            })

    def __len__(self):
        return len(self.episode_ids) * 150

    def __getitem__(self, index):
        episode_idx = index % len(self.episode_ids)
        ep = self.data_list[episode_idx]
        
        active_len = ep['active_len']
        if active_len > 1:
            start_ts = np.random.choice(active_len - 1)
        else:
            start_ts = 0
            
        qpos = ep['qpos_all'][start_ts]
        image_dict = {}
        for cam_name in self.camera_names:
            image_dict[cam_name] = ep['images_all'][cam_name][start_ts]
            
        if ep['is_sim']:
            action = ep['action_all'][start_ts:active_len]
            action_len = active_len - start_ts
        else:
            action = ep['action_all'][max(0, start_ts - 1):active_len]
            action_len = active_len - max(0, start_ts - 1)
            
        padded_action = np.zeros(ep['original_action_shape'], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(ep['episode_len'])
        is_pad[action_len:] = 1
        
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0
        # Resize to 240x320 for 4x training speedup
        image_data = F.interpolate(image_data, size=(240, 320), mode='bilinear', align_corners=False)
        
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        return image_data, qpos_data, action_data, is_pad

def train_and_eval_one_episode():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task = 'panda_pick_and_place'
    dataset_dir = "data/panda_pick_and_place"
    
    # 1. Modify configs for 1 episode overfitting
    POLICY_CONFIG['kl_weight'] = 10
    POLICY_CONFIG['lr'] = 1e-4
    POLICY_CONFIG['lr_backbone'] = 1e-5
    POLICY_CONFIG['temporal_agg'] = True
    
    # We load stats only on episode 1
    def get_norm_stats_for_episode(dataset_dir, episode_idx):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            
            # Find actual active length
            active_len = len(action)
            for idx in range(len(action) - 1, -1, -1):
                if np.any(action[idx] != 0):
                    active_len = idx + 1
                    break
                    
            qpos_data = torch.from_numpy(qpos[:active_len])
            action_data = torch.from_numpy(action[:active_len])
            
            action_mean = action_data.mean(dim=0)
            action_std = action_data.std(dim=0)
            action_std = torch.clip(action_std, 1e-2, np.inf)
            
            qpos_mean = qpos_data.mean(dim=0)
            qpos_std = qpos_data.std(dim=0)
            qpos_std = torch.clip(qpos_std, 1e-2, np.inf)
            
            stats = {
                "action_mean": action_mean.numpy(),
                "action_std": action_std.numpy(),
                "qpos_mean": qpos_mean.numpy(),
                "qpos_std": qpos_std.numpy(),
                "example_qpos": qpos
            }
            return stats
            
    stats = get_norm_stats_for_episode(dataset_dir, episode_idx=1)
    
    # Save stats
    checkpoint_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], task)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'dataset_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
        
    # Create dataset with train_indices = [1]
    train_dataset = PreloadedEpisodicDataset([1], dataset_dir, TASK_CONFIG['camera_names'], stats)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
    
    # Make policy
    policy_config = POLICY_CONFIG.copy()
    policy_config['device'] = device
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)
    
    optimizer = policy.configure_optimizers()
    
    # 2. Train for 300 epochs
    print("\n=== STARTING OVERFITTING TRAINING ON EPISODE 1 ===", flush=True)
    policy.train()
    for epoch in range(300):
        epoch_loss = 0
        epoch_l1 = 0
        epoch_l1_pos0 = 0
        epoch_l1_pos1 = 0
        epoch_kl = 0
        batch_count = 0
        for data in train_dataloader:
            image_data, qpos_data, action_data, is_pad = data
            image_data = image_data.to(device)
            qpos_data = qpos_data.to(device)
            action_data = action_data.to(device)
            is_pad = is_pad.to(device)
            
            optimizer.zero_grad()
            forward_dict = policy(qpos_data, image_data, action_data, is_pad)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_l1 += forward_dict['l1'].item()
            epoch_kl += forward_dict['kl'].item()
            
            # Position 0 L1
            a_hat = forward_dict['a_hat']
            l1_pos0 = torch.abs(action_data[:, 0] - a_hat[:, 0])
            mask0 = ~is_pad[:, 0].unsqueeze(-1)
            l1_pos0 = (l1_pos0 * mask0).sum() / (mask0.sum() + 1e-6)
            
            # Position 1 L1
            l1_pos1 = torch.abs(action_data[:, 1] - a_hat[:, 1])
            mask1 = ~is_pad[:, 1].unsqueeze(-1)
            l1_pos1 = (l1_pos1 * mask1).sum() / (mask1.sum() + 1e-6)
            
            epoch_l1_pos0 += l1_pos0.item()
            epoch_l1_pos1 += l1_pos1.item()
            batch_count += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/batch_count:.6f} | L1_avg: {epoch_l1/batch_count:.6f} | L1_pos0: {epoch_l1_pos0/batch_count:.6f} | L1_pos1: {epoch_l1_pos1/batch_count:.6f} | KL: {epoch_kl/batch_count:.6f}", flush=True)
            
    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, 'policy_last_1000epochs.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    
    # 3. Rollout evaluation on the exact starting state of episode 0
    print("\n=== STARTING ROLLOUT EVALUATION ===")
    policy.eval()
    
    # Load episode 1 data to set initial state and track targets
    with h5py.File(os.path.join(dataset_dir, "episode_1.hdf5"), 'r') as root:
        true_actions = root['action'][()]
        true_qpos = root['observations/qpos'][()]
        
    env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="rgb_array", render_width=640, render_height=480)
    obs, info = env.reset()
    robot = env.unwrapped.robot
    sim = env.unwrapped.sim
    
    # Reset robot and objects to the exact step 0 state from episode 0
    qpos_init = true_qpos[0]
    arm_angles = qpos_init[0:7]
    gripper_width = qpos_init[7]
    cube_pos = qpos_init[8:11]
    target_pos = qpos_init[11:14]
    
    # Programmatic reset
    robot.set_joint_angles(arm_angles)
    for finger_idx in robot.fingers_indices:
        sim.set_joint_angle(robot.body_name, finger_idx, gripper_width / 2.0)
    sim.set_base_pose("object", position=cube_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    sim.set_base_pose("target", position=target_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    env.unwrapped.task.goal = target_pos.copy()
    
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    temporal_agg = POLICY_CONFIG['temporal_agg']
    query_frequency = POLICY_CONFIG['num_queries']
    max_steps = 150
    state_dim = POLICY_CONFIG['state_dim']
    
    if temporal_agg:
        all_time_actions = torch.zeros([max_steps, max_steps + query_frequency, state_dim]).to(device)
        
    success = False
    for t in range(max_steps):
        # Render image
        img = env.render()
        
        # Get current state from environment
        current_ee_pos = robot.get_ee_position()
        current_fingers_width = robot.get_fingers_width()
        current_arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
        current_cube_pos = obs['achieved_goal']
        current_target_pos = obs['desired_goal']
        
        qpos_numpy = np.concatenate([current_arm_joint_angles, [current_fingers_width], current_cube_pos, current_target_pos])
        
        # Preprocess
        qpos = pre_process(qpos_numpy)
        qpos_torch = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
        
        # Policy query
        with torch.inference_mode():
            images_dict = {'front': img}
            curr_image = get_image(images_dict, POLICY_CONFIG['camera_names'], device)
            curr_image = F.interpolate(curr_image.squeeze(0), size=(240, 320), mode='bilinear', align_corners=False).unsqueeze(0)
            
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
        action = action[:8]
        action = np.clip(action, -1.0, 1.0)
        
        # Step env
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print comparison for debugging
        if t <= 15 or t % 5 == 0:
            true_act = true_actions[t] if t < len(true_actions) else np.zeros(8)
            print(f"Step {t:2d} | True gripper act: {true_act[7]:.4f} | Pred gripper act: {action[7]:.4f} | True joint 0: {true_act[0]:.4f} | Pred joint 0: {action[0]:.4f}", flush=True)
            
        if info.get('is_success', False):
            success = True
            print(f"\nSUCCESS ACHIEVED AT STEP {t}!", flush=True)
            
        if terminated or truncated:
            print(f"Environment ended at step {t}. Terminated={terminated}, Truncated={truncated}", flush=True)
            break
            
    final_cube_pos = obs['achieved_goal']
    final_dist = np.linalg.norm(final_cube_pos - target_pos)
    print("\n=== FINAL PLACEMENT RESULTS ===")
    print(f"Final Cube Position (Actual):  {final_cube_pos}")
    print(f"Desired Target Position:       {target_pos}")
    print(f"Placement Error (Distance):    {final_dist:.6f} meters ({(final_dist * 100):.2f} cm)")
    print("===============================\n")
    
    print(f"\nEpisode Rollout Complete. Success: {success}", flush=True)
    env.close()

if __name__ == "__main__":
    train_and_eval_one_episode()
