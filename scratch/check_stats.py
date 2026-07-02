import pickle

stats_path = "checkpoints/panda_pick_and_place/dataset_stats.pkl"
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

print("Action Mean:", stats['action_mean'])
print("Action Std:", stats['action_std'])
print("Qpos Mean:", stats['qpos_mean'])
print("Qpos Std:", stats['qpos_std'])
