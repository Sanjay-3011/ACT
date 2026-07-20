import os
import h5py
import numpy as np

dataset_dir = "data/panda_pick_and_place"

for ep_id in range(1, 6):
    file_path = os.path.join(dataset_dir, f"episode_{ep_id}.hdf5")
    if not os.path.exists(file_path):
        print(f"Episode {ep_id} not found!")
        continue
        
    with h5py.File(file_path, 'r') as root:
        actions = root['action'][()]
        
    # Detect first step where gripper closes (action[t, 7] < -0.1)
    t_grasp = None
    for t in range(len(actions)):
        if actions[t, 7] < -0.1:
            t_grasp = t
            break
            
    if t_grasp is None:
        print(f"Episode {ep_id}: Grasp step not detected!")
        continue
        
    # Compute chunk weights at t_start = 0 (chunk_size = 100)
    chunk_size = 100
    early_count = t_grasp
    tail_count = chunk_size - t_grasp
    
    early_weight_sum = early_count * 5.0
    tail_weight_sum = tail_count * 0.2
    ratio = early_weight_sum / (tail_weight_sum + 1e-6)
    
    # Compute normalized weights at pos 0 and pos 30 to see scale
    raw_sum = early_weight_sum + tail_weight_sum
    norm_early = 5.0 / raw_sum * chunk_size
    norm_tail = 0.2 / raw_sum * chunk_size
    
    print(f"Episode {ep_id}: Detected T_grasp = {t_grasp} | Early Sum = {early_weight_sum:.1f} | Tail Sum = {tail_weight_sum:.1f} | Ratio = {ratio:.3f}x | Norm Early = {norm_early:.4f} | Norm Tail = {norm_tail:.4f}")
