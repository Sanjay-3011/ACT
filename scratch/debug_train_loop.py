import sys
sys.argv = ['debug_train_loop.py', '--task', 'panda_pick_and_place']
sys.path.append('shaka_act')

import os
import torch
import numpy as np
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import load_data, make_policy, make_optimizer

device = os.environ.get('DEVICE', 'cuda')
task = 'panda_pick_and_place'
checkpoint_dir = os.path.join(TRAIN_CONFIG['checkpoint_dir'], task)
data_dir = os.path.join(TASK_CONFIG['dataset_dir'], task)
num_episodes = 5  # use 5 episodes for quick debug

print("Loading data...")
train_dataloader, val_dataloader, stats, _ = load_data(
    data_dir, num_episodes, TASK_CONFIG['camera_names'],
    TRAIN_CONFIG['batch_size_train'], TRAIN_CONFIG['batch_size_val']
)

print("Making policy...")
policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
policy.to(device)

print("Making optimizer...")
optimizer = make_optimizer(POLICY_CONFIG['policy_class'], policy)

print("Entering validation loop...")
policy.eval()
with torch.inference_mode():
    for batch_idx, data in enumerate(val_dataloader):
        print(f"Val batch {batch_idx}")
        image_data, qpos_data, action_data, is_pad = data
        image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
        out = policy(qpos_data, image_data, action_data, is_pad)
        print("Val batch done")
        break

print("Entering training loop...")
policy.train()
optimizer.zero_grad()
for batch_idx, data in enumerate(train_dataloader):
    print(f"Train batch {batch_idx} loaded")
    image_data, qpos_data, action_data, is_pad = data
    print("Moving data to device...")
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    
    print("Running forward pass...")
    out = policy(qpos_data, image_data, action_data, is_pad)
    
    print("Running backward pass...")
    loss = out['loss']
    loss.backward()
    
    print("Running optimizer step...")
    optimizer.step()
    optimizer.zero_grad()
    print("Train batch done successfully!")
    break

print("DEBUG SUCCESSFUL!")
