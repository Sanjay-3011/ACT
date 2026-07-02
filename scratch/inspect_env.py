import gymnasium as gym
import panda_gym

env = gym.make("PandaPickAndPlace-v3")
obs, info = env.reset()

print("Env dir:", dir(env.unwrapped))
robot = env.unwrapped.robot
print("Robot dir:", dir(robot))

# Let's inspect some robot methods
try:
    print("ee position:", robot.get_ee_position())
except Exception as e:
    print("Error get_ee_position:", e)

try:
    print("fingers width:", robot.get_fingers_width())
except Exception as e:
    print("Error get_fingers_width:", e)

try:
    print("joint indices:", robot.joint_indices)
    joint_angles = [robot.get_joint_angle(i) for i in robot.joint_indices]
    print("joint angles:", joint_angles)
except Exception as e:
    print("Error joint angles:", e)

