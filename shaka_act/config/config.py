import os
import sys

# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = 'data/'

# checkpoint directory
CHECKPOINT_DIR = 'checkpoints/'

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
os.environ['DEVICE'] = device

# Detect task name from arguments
task_name = 'task1'
for arg in sys.argv:
    if arg.startswith('--task='):
        task_name = arg.split('=')[1]
    elif arg == '--task' and len(sys.argv) > sys.argv.index(arg) + 1:
        task_name = sys.argv[sys.argv.index(arg) + 1]

# Set task-specific configurations
if task_name == 'panda_pick_and_place':
    state_dim = 14
    action_dim = 14
    episode_len = 120
else:
    state_dim = 5
    action_dim = 5
    episode_len = 300

# robot port names (only used for real robot)
ROBOT_PORTS = {
    'leader': '/dev/tty.usbmodem57380045221',
    'follower': '/dev/tty.usbmodem57380046991'
}

# task config
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': episode_len,
    'state_dim': state_dim,
    'action_dim': action_dim,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
    'camera_port': 0
}

# policy config
POLICY_CONFIG = {
    'lr': 2e-4,
    'device': device,
    'num_queries': 100,
    'kl_weight': 0,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 2e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    'temporal_agg': True,
    'state_dim': state_dim,
    'action_dim': action_dim
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 100,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}