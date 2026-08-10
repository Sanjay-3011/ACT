import gymnasium as gym
import panda_gym
import numpy as np
import cv2
import pybullet as p
import os

env = gym.make("PandaPickAndPlace-v3", control_type="joints", render_mode="human", render_width=640, render_height=480)
obs, info = env.reset(seed=42)
sim = env.unwrapped.sim

target_pos_cam = env.unwrapped.render_target_position
view_matrix = sim.physics_client.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=target_pos_cam,
    distance=env.unwrapped.render_distance,
    yaw=env.unwrapped.render_yaw,
    pitch=env.unwrapped.render_pitch,
    roll=env.unwrapped.render_roll,
    upAxisIndex=2,
)
proj_matrix = sim.physics_client.computeProjectionMatrixFOV(
    fov=60, aspect=float(640) / 480, nearVal=0.1, farVal=100.0
)
(_, _, rgba, _, _) = sim.physics_client.getCameraImage(
    width=640,
    height=480,
    viewMatrix=view_matrix,
    projectionMatrix=proj_matrix,
    shadow=True,
    renderer=p.ER_BULLET_HARDWARE_OPENGL,
)
rgba = np.array(rgba, dtype=np.uint8).reshape((480, 640, 4))
img = rgba[..., :3]

os.makedirs("data/rendered_video", exist_ok=True)
cv2.imwrite("data/rendered_video/test_gui_frame.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Saved image to data/rendered_video/test_gui_frame.png. Shape:", img.shape)
env.close()
