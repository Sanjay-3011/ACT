import gymnasium as gym
import panda_gym
import numpy as np

env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="human")
obs, info = env.reset()
try:
    img = env.unwrapped.sim.render(width=640, height=480)
    print("sim.render success! Image shape:", img.shape)
except Exception as e:
    print("sim.render failed:", e)
env.close()
