import os
import h5py
import torch
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from training.policy import ACTPolicy, CNNMLPPolicy


import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, is_train=False):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_train = is_train
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomCrop(size=(200, 200)),
                transforms.Resize(size=(224, 224))
            ])
        else:
            self.transform = transforms.Resize(size=(224, 224))
        
        # Preload all episodes in RAM to eliminate disk I/O bottlenecks
        print(f"Preloading {len(episode_ids)} episodes into RAM...", flush=True)
        self.episodes_data = {}
        for ep_id in episode_ids:
            dataset_path = os.path.join(self.dataset_dir, f'episode_{ep_id}.hdf5')
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
                
            self.episodes_data[ep_id] = {
                'is_sim': is_sim,
                'original_action_shape': original_action_shape,
                'episode_len': episode_len,
                'active_len': active_len,
                'images_all': images_all,
                'qpos_all': qpos_all,
                'action_all': action_all
            }
        self.is_sim = is_sim

    def __len__(self):
        return len(self.episode_ids) * 150

    def __getitem__(self, index):
        episode_id = self.episode_ids[index % len(self.episode_ids)]
        ep = self.episodes_data[episode_id]
        
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
        
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        if self.transform is not None:
            augmented_images = []
            for cam_img in image_data:
                augmented_images.append(self.transform(cam_img))
            image_data = torch.stack(augmented_images, dim=0)
            
        image_data = image_data.float() / 255.0
            
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
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
                    
            all_qpos_data.append(torch.from_numpy(qpos[:active_len]))
            all_action_data.append(torch.from_numpy(action[:active_len]))
            
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0)
    action_std = all_action_data.std(dim=0)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0)
    qpos_std = all_qpos_data.std(dim=0)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, is_train=True)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=0)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return optimizer

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def get_image(images, camera_names, device='cpu'):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(images[cam_name], 'h w c -> c h w')
        curr_image = torch.from_numpy(curr_image / 255.0).float()
        curr_images.append(curr_image)
    curr_image = torch.stack(curr_images, dim=0).to(device).unsqueeze(0)
    return curr_image

def compute_dict_mean(epoch_dicts):
    result = {}
    num_items = len(epoch_dicts)
    for k in epoch_dicts[0]:
        # Only compute mean and include in result if the key is a scalar to avoid batch-size shape mismatch and item() printing errors
        is_scalar = isinstance(epoch_dicts[0][k], (int, float)) or (isinstance(epoch_dicts[0][k], torch.Tensor) and epoch_dicts[0][k].ndim == 0)
        if is_scalar:
            value_sum = 0
            for epoch_dict in epoch_dicts:
                value_sum += epoch_dict[k]
            result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pos2pwm(pos:np.ndarray) -> np.ndarray:
    """
    :param pos: numpy array of joint positions in range [-pi, pi]
    :return: numpy array of pwm values in range [0, 4096]
    """ 
    return (pos / 3.14 + 1.) * 2048
    
def pwm2pos(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm values in range [0, 4096]
    :return: numpy array of joint positions in range [-pi, pi]
    """
    return (pwm / 2048 - 1) * 3.14

def pwm2vel(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm/s joint velocities
    :return: numpy array of rad/s joint velocities 
    """
    return pwm * 3.14 / 2048

def vel2pwm(vel:np.ndarray) -> np.ndarray:
    """
    :param vel: numpy array of rad/s joint velocities
    :return: numpy array of pwm/s joint velocities
    """
    return vel * 2048 / 3.14
    
def pwm2norm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of pwm values in range [0, 4096]
    :return: numpy array of values in range [0, 1]
    """
    return x / 4096
    
def norm2pwm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of values in range [0, 1]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return x * 4096