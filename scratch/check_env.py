import gymnasium as gym
import panda_gym

env = gym.make("PandaPickAndPlace-v3", control_type="joints")
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
obs, info = env.reset()
print("Observation keys:", obs.keys())
print("Observation shapes:")
for k, v in obs.items():
    print(f"  {k}: {v.shape}")
    print(f"    value: {v}")
