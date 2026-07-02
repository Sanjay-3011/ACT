import gymnasium as gym
import panda_gym
import numpy as np

def test_single_episode():
    # Allow 200 steps
    env = gym.make(
        "PandaPickAndPlace-v3", 
        control_type="joints", 
        render_mode="rgb_array", 
        render_width=640, 
        render_height=480,
        max_episode_steps=200
    )
    
    obs, info = env.reset()
    robot = env.unwrapped.robot
    gripper_open_width = 0.08
    gripper_closed_width = 0.0
    
    state = 0
    state_timer = 0
    success = False
    
    cube_initial_pos = obs['achieved_goal'].copy()
    print("Initial cube pos:", cube_initial_pos)
    print("Initial target pos:", obs['desired_goal'])
    
    for step in range(200):
        current_ee_pos = robot.get_ee_position()
        current_fingers_width = robot.get_fingers_width()
        
        arm_joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
        cube_pos = obs['achieved_goal'].copy()
        target_pos = obs['desired_goal'].copy()
        
        target_ee_pos = current_ee_pos.copy()
        target_fingers_width = current_fingers_width
        
        if state == 0:  # MOVE_ABOVE_CUBE
            target_ee_pos = cube_initial_pos + np.array([0.0, 0.0, 0.12])
            target_fingers_width = gripper_open_width
            dist = np.linalg.norm(target_ee_pos - current_ee_pos)
            if dist < 0.015:
                state = 1
                print(f"Step {step}: Transition to DESCEND_TO_CUBE. dist={dist:.4f}")
        
        elif state == 1:  # DESCEND_TO_CUBE
            target_ee_pos = cube_initial_pos + np.array([0.0, 0.0, 0.005])
            target_fingers_width = gripper_open_width
            dist = np.linalg.norm(target_ee_pos - current_ee_pos)
            if dist < 0.015:
                state = 2
                state_timer = 0
                print(f"Step {step}: Transition to CLOSE_GRIPPER. dist={dist:.4f}")
        
        elif state == 2:  # CLOSE_GRIPPER
            target_ee_pos = cube_initial_pos + np.array([0.0, 0.0, 0.005])
            target_fingers_width = gripper_closed_width
            state_timer += 1
            if state_timer > 15:
                state = 3
                print(f"Step {step}: Transition to LIFT_CUBE. gripper_width={current_fingers_width:.4f}")
        
        elif state == 3:  # LIFT_CUBE
            target_ee_pos = cube_initial_pos + np.array([0.0, 0.0, 0.15])
            target_fingers_width = gripper_closed_width
            dist = np.linalg.norm(target_ee_pos - current_ee_pos)
            if dist < 0.015:
                state = 4
                print(f"Step {step}: Transition to MOVE_ABOVE_TARGET. dist={dist:.4f}")
        
        elif state == 4:  # MOVE_ABOVE_TARGET
            target_ee_pos = target_pos + np.array([0.0, 0.0, 0.15])
            target_fingers_width = gripper_closed_width
            dist = np.linalg.norm(target_ee_pos - current_ee_pos)
            if dist < 0.02:
                state = 5
                print(f"Step {step}: Transition to DESCEND_TO_TARGET. dist={dist:.4f}")
        
        elif state == 5:  # DESCEND_TO_TARGET
            target_ee_pos = target_pos + np.array([0.0, 0.0, 0.015])
            target_fingers_width = gripper_closed_width
            dist = np.linalg.norm(target_ee_pos - current_ee_pos)
            if dist < 0.015:
                state = 6
                state_timer = 0
                print(f"Step {step}: Transition to OPEN_GRIPPER. dist={dist:.4f}")
        
        elif state == 6:  # OPEN_GRIPPER
            target_ee_pos = target_pos + np.array([0.0, 0.0, 0.015])
            target_fingers_width = gripper_open_width
            state_timer += 1
            if state_timer > 15:
                state = 7
                print(f"Step {step}: Transition to RETRACT. gripper_width={current_fingers_width:.4f}")
        
        elif state == 7:  # RETRACT
            target_ee_pos = target_pos + np.array([0.0, 0.0, 0.15])
            target_fingers_width = gripper_open_width
            dist = np.linalg.norm(target_ee_pos - current_ee_pos)
            if dist < 0.015:
                state = 8
                print(f"Step {step}: Retracted to start height. Done.")
        
        elif state == 8:
            pass
            
        # Solve IK
        target_arm_angles = robot.inverse_kinematics(
            link=robot.ee_link, 
            position=target_ee_pos, 
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )[:7]
        
        action_joints = (target_arm_angles - arm_joint_angles) / 0.05
        action_joints = np.clip(action_joints, -1.0, 1.0)
        
        action_gripper = (target_fingers_width - current_fingers_width) / 0.2
        action_gripper = np.clip(action_gripper, -1.0, 1.0)
        
        action = np.concatenate([action_joints, [action_gripper]])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check success
        cube_target_dist = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        if cube_target_dist < 0.05 and state >= 6:
            success = True
            print(f"Step {step}: SUCCESS! cube_target_dist={cube_target_dist:.4f}")
            break
            
        if terminated or truncated:
            print(f"Step {step}: Terminated/Truncated. Terminated={terminated}, Truncated={truncated}, reward={reward}, info={info}")
            break
            
    print(f"Episode finished. Success: {success}")
    env.close()

if __name__ == "__main__":
    test_single_episode()
