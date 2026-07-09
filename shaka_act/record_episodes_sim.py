import os
import h5py
import numpy as np
import gymnasium as gym
import panda_gym
from tqdm import tqdm

def is_gripping(env):
    sim = env.unwrapped.sim
    panda_id = sim._bodies_idx['panda']
    object_id = sim._bodies_idx['object']
    contacts = sim.physics_client.getContactPoints(bodyA=panda_id, bodyB=object_id)
    for c in contacts:
        link_a = c[3]  # Link index on bodyA (panda)
        force = c[9]   # Normal force
        if link_a in [9, 10] and force > 0.01:
            return True
    return False

def collect_episodes(num_episodes=55, max_steps=150):
    env = gym.make(
        "PandaPickAndPlace-v3", 
        control_type="joints", 
        render_mode="rgb_array", 
        render_width=640, 
        render_height=480,
        max_episode_steps=max_steps
    )
    # Force the target to always be on the table
    env.unwrapped.task.goal_range_low[2] = 0.0
    env.unwrapped.task.goal_range_high[2] = 0.0
    
    # Output directory
    out_dir = "data/panda_pick_and_place"
    os.makedirs(out_dir, exist_ok=True)
    
    success_count = 0
    pbar = tqdm(total=num_episodes, desc="Collecting episodes")
    
    while success_count < num_episodes:
        obs, info = env.reset()
        
        # State machine states
        # 0: MOVE_ABOVE_CUBE
        # 1: DESCEND_TO_CUBE
        # 2: CLOSE_GRIPPER (hold position, close fingers)
        # 3: LIFT_CUBE
        # 4: MOVE_ABOVE_TARGET
        # 5: DESCEND_TO_TARGET
        # 6: OPEN_GRIPPER (hold position, open fingers)
        # 7: RETRACT
        # 8: SUCCESS_DONE
        state = 0
        state_timer = 0
        
        qpos_list = []
        qvel_list = []
        images_list = []
        actions_list = []
        
        # Access underlying robot
        robot = env.unwrapped.robot
        
        # Define key state thresholds and target offsets
        gripper_open_width = 0.08
        gripper_closed_width = 0.0
        
        # Record initial position of the cube (fixed reference frame)
        cube_initial_pos = obs['achieved_goal'].copy()
        
        # Run loop
        success = False
        for step in range(max_steps):
            # 1. Get current robot states
            current_ee_pos = robot.get_ee_position()
            current_fingers_width = robot.get_fingers_width()
            
            arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
            arm_joint_vel = np.array([robot.get_joint_velocity(joint=i) for i in range(7)])
            
            # Gripper velocity (average of finger joints 9 and 10)
            gripper_vel = robot.get_joint_velocity(9) + robot.get_joint_velocity(10)
            
            # Get positions of cube and target
            cube_pos = obs['achieved_goal'].copy()
            target_pos = obs['desired_goal'].copy()
            
            qpos = np.concatenate([arm_joint_angles, [current_fingers_width], cube_pos, target_pos])
            qvel = np.concatenate([arm_joint_vel, [gripper_vel], np.zeros(6)])
            
            # Render camera image
            img = env.render()
            
            qpos_list.append(qpos)
            qvel_list.append(qvel)
            images_list.append(img)
            
            # State Machine Controller
            target_ee_pos = current_ee_pos.copy()
            target_fingers_width = current_fingers_width
            
            if state == 0:  # MOVE_ABOVE_CUBE
                target_ee_pos = cube_initial_pos + np.array([0.0, 0.0, 0.12])
                target_fingers_width = gripper_open_width
                if np.linalg.norm(target_ee_pos - current_ee_pos) < 0.015:
                    state = 1
            
            elif state == 1:  # DESCEND_TO_CUBE
                target_ee_pos = cube_initial_pos + np.array([0.0, 0.0, 0.005])
                target_fingers_width = gripper_open_width
                if np.linalg.norm(target_ee_pos - current_ee_pos) < 0.015:
                    state = 2
                    state_timer = 0
            
            elif state == 2:  # CLOSE_GRIPPER
                target_ee_pos = cube_initial_pos + np.array([0.0, 0.0, 0.005])
                target_fingers_width = gripper_closed_width
                state_timer += 1
                if state_timer > 15:
                    state = 3
            
            elif state == 3:  # LIFT_CUBE
                target_ee_pos = cube_initial_pos + np.array([0.0, 0.0, 0.15])
                target_fingers_width = gripper_closed_width
                if np.linalg.norm(target_ee_pos - current_ee_pos) < 0.015:
                    state = 4
            
            elif state == 4:  # MOVE_ABOVE_TARGET
                target_ee_pos = target_pos + np.array([0.0, 0.0, 0.15])
                target_fingers_width = gripper_closed_width
                if np.linalg.norm(target_ee_pos - current_ee_pos) < 0.02:
                    state = 5
            
            elif state == 5:  # DESCEND_TO_TARGET
                target_ee_pos = target_pos + np.array([0.0, 0.0, 0.015])
                target_fingers_width = gripper_closed_width
                if np.linalg.norm(target_ee_pos - current_ee_pos) < 0.005:
                    state = 6
                    state_timer = 0
            
            elif state == 6:  # OPEN_GRIPPER
                target_ee_pos = target_pos + np.array([0.0, 0.0, 0.015])
                target_fingers_width = gripper_open_width
                state_timer += 1
                if state_timer > 15:
                    state = 7
            
            elif state == 7:  # RETRACT
                target_ee_pos = target_pos + np.array([0.0, 0.0, 0.15])
                target_fingers_width = gripper_open_width
                if np.linalg.norm(target_ee_pos - current_ee_pos) < 0.015:
                    state = 8
            
            elif state == 8:  # SUCCESS_DONE
                target_ee_pos = target_pos + np.array([0.0, 0.0, 0.15])
                target_fingers_width = gripper_open_width
            
            # Compute actions via inverse kinematics
            target_arm_angles = robot.inverse_kinematics(
                link=robot.ee_link, 
                position=target_ee_pos, 
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )[:7]
            
            action_joints = (target_arm_angles - arm_joint_angles) / 0.05
            action_joints = np.clip(action_joints, -1.0, 1.0)
            
            action_gripper = (target_fingers_width - current_fingers_width) / 0.2
            action_gripper = np.clip(action_gripper, -1.0, 1.0)
            
            action = np.concatenate([action_joints, [action_gripper], np.zeros(6)])
            actions_list.append(action)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action[:8])
            
            # Check success
            cube_pos = obs['achieved_goal']
            target_pos = obs['desired_goal']
            dist = np.linalg.norm(cube_pos - target_pos)
            gripper_width = robot.get_fingers_width()
            currently_gripping = is_gripping(env)
            
            is_success = (dist < 0.05) and (gripper_width > 0.07) and (not currently_gripping)
            if is_success:
                success = True
                
            if state == 8:
                break
                
            if truncated:
                break
                
        if success:
            # Pad the lists to max_steps
            last_qpos = qpos_list[-1]
            last_qvel = np.zeros_like(qvel_list[-1])  # zero velocity
            last_img = images_list[-1]
            last_action = np.zeros_like(actions_list[-1])  # zero action (hold position)
            
            while len(qpos_list) < max_steps:
                qpos_list.append(last_qpos)
                qvel_list.append(last_qvel)
                images_list.append(last_img)
                actions_list.append(last_action)
                
            # Convert to numpy arrays
            qpos_arr = np.array(qpos_list, dtype=np.float32)
            qvel_arr = np.array(qvel_list, dtype=np.float32)
            actions_arr = np.array(actions_list, dtype=np.float32)
            images_arr = np.array(images_list, dtype=np.uint8)
            
            dataset_path = os.path.join(out_dir, f'episode_{success_count}.hdf5')
            with h5py.File(dataset_path, 'w') as root:
                root.attrs['sim'] = True
                root.create_dataset('action', data=actions_arr)
                obs_group = root.create_group('observations')
                obs_group.create_dataset('qpos', data=qpos_arr)
                obs_group.create_dataset('qvel', data=qvel_arr)
                
                image_group = obs_group.create_group('images')
                image_group.create_dataset('front', data=images_arr, dtype='uint8', chunks=(1, 480, 640, 3))
                
            success_count += 1
            pbar.update(1)
            
    env.close()
    print(f"Successfully collected {num_episodes} demonstration episodes.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=55, help='Number of episodes to collect')
    args = parser.parse_args()
    collect_episodes(num_episodes=args.num_episodes)
