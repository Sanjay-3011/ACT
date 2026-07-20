import sys
sys.path.append('shaka_act')
import os
import torch
import numpy as np
import h5py
import gymnasium as gym
import panda_gym
from config.config import POLICY_CONFIG
from training.policy import ACTPolicy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure policy config
    policy_config = {
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'kl_weight': 10,
        'backbone': 'resnet18',
        'camera_names': ['front'],
        'num_queries': 100,
        'state_dim': 14,
        'action_dim': 8,
        'hidden_dim': 512,
        'nheads': 8,
        'dim_feedforward': 3200,
        'enc_layers': 4,
        'dec_layers': 7,
        'position_embedding': 'sine',
    }
    
    policy = ACTPolicy(policy_config)
    policy.to(device)
    
    ckpt_path = "checkpoints/panda_pick_and_place/policy_last_1000epochs.ckpt"
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()
    
    # Load dataset stats
    dataset_dir = "data/panda_pick_and_place"
    
    # Load episode 1 data
    with h5py.File(os.path.join(dataset_dir, "episode_1.hdf5"), 'r') as root:
        actions = root['action'][()]
        qpos = root['observations/qpos'][()]
        images = root['/observations/images/front'][()]
        
    # Get normalized stats
    # To normalize properly, let's compute norm stats as in the dataset class
    action_mean = np.mean(actions, axis=0)
    action_std = np.std(actions, axis=0) + 1e-5
    
    qpos_mean = np.mean(qpos, axis=0)
    qpos_std = np.std(qpos, axis=0) + 1e-5
    
    # Normalize inputs
    qpos_norm = (qpos - qpos_mean) / qpos_std
    actions_norm = (actions - action_mean) / action_std
    
    # Process images (convert to PyTorch tensors and normalize)
    images_torch = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
    images_torch = torch.nn.functional.interpolate(images_torch, size=(240, 320), mode='bilinear', align_corners=False)
    
    qpos_torch = torch.from_numpy(qpos_norm).float()
    actions_norm_torch = torch.from_numpy(actions_norm).float()
    
    # We want to run forward pass on Episode 1 batches.
    # The dataloader fetches random chunks, but here we can just pass the first frame (t=0)
    # or average over all steps to see the training loss at those positions on this episode!
    # Let's average the L1 error for each position across all valid chunk predictions.
    
    total_l1_pos = {30: 0.0, 60: 0.0, 90: 0.0, 0: 0.0, 1: 0.0, 2: 0.0}
    count = 0
    
    # We can query policy from step t=0 to t=50 (where 100-step predictions are valid within the 150-step trajectory)
    with torch.no_grad():
        for t in range(50):
            # Input observations
            img = images_torch[t].unsqueeze(0).unsqueeze(1).to(device)
            qp = qpos_torch[t].unsqueeze(0).to(device)
            
            # Predict actions (forward pass at inference time, or training format to get the specific model output)
            # In inference mode, policy(qp, img) returns a_hat
            a_hat = policy(qp, img) # shape [1, 100, 8]
            
            # True actions in the future (from t to t+100)
            true_a = actions_norm_torch[t:t+100].to(device) # shape [100, 8]
            
            if true_a.shape[0] < 100:
                continue
                
            for pos in total_l1_pos.keys():
                err = torch.abs(true_a[pos] - a_hat[0, pos]) # shape [8]
                total_l1_pos[pos] += err.mean().item()
            count += 1
            
    print(f"Computed over {count} steps:")
    for pos, val in total_l1_pos.items():
        avg_val = val / count
        print(f"Position {pos:2d} | Normalized L1: {avg_val:.6f}")

if __name__ == "__main__":
    main()
