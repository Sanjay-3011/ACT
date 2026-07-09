import gymnasium as gym
import panda_gym
import numpy as np
import time

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

def test_single_episode_gui():
    # Set render_mode to "human" to open the interactive 3D PyBullet GUI window
    env = gym.make(
        "PandaPickAndPlace-v3", 
        control_type="joints", 
        render_mode="human", 
        max_episode_steps=200
    )
    # Force the target to always be on the table
    env.unwrapped.task.goal_range_low[2] = 0.0
    env.unwrapped.task.goal_range_high[2] = 0.0
    
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
            target_ee_pos = obs['desired_goal'].copy() + np.array([0.0, 0.0, 0.15])
            target_fingers_width = gripper_closed_width
            dist = np.linalg.norm(target_ee_pos - current_ee_pos)
            if dist < 0.02:
                state = 5
                print(f"Step {step}: Transition to DESCEND_TO_TARGET. dist={dist:.4f}")
        
        elif state == 5:  # DESCEND_TO_TARGET
            target_ee_pos = obs['desired_goal'].copy() + np.array([0.0, 0.0, 0.015])
            target_fingers_width = gripper_closed_width
            dist = np.linalg.norm(target_ee_pos - current_ee_pos)
            if dist < 0.005:
                state = 6
                state_timer = 0
                print(f"Step {step}: Transition to OPEN_GRIPPER. dist={dist:.4f}")
        
        elif state == 6:  # OPEN_GRIPPER
            target_ee_pos = obs['desired_goal'].copy() + np.array([0.0, 0.0, 0.015])
            target_fingers_width = gripper_open_width
            state_timer += 1
            if state_timer > 15:
                state = 7
                print(f"Step {step}: Transition to RETRACT. gripper_width={current_fingers_width:.4f}")
        
        elif state == 7:  # RETRACT
            target_ee_pos = obs['desired_goal'].copy() + np.array([0.0, 0.0, 0.15])
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
        
        # Add a tiny delay to make the simulation readable/smooth to human eyes
        time.sleep(0.02)
        
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
            print(f"Step {step}: Expert completed the entire pick, place, release, and retract sequence.")
            break
            
        if truncated:
            print(f"Step {step}: Episode truncated.")
            break
            
    final_cube_pos = obs['achieved_goal']
    target_pos = obs['desired_goal']
    final_dist = np.linalg.norm(final_cube_pos - target_pos)
    print("\n=== FINAL PLACEMENT RESULTS ===")
    print(f"Final Cube Position (Actual):  {final_cube_pos}")
    print(f"Desired Target Position:       {target_pos}")
    print(f"Placement Error (Distance):    {final_dist:.6f} meters ({(final_dist * 100):.2f} cm)")
    print("===============================\n")
    
    print(f"Episode finished. Success: {success}")
    
    # Keep the window open and responsive after execution completes
    print("Keeping simulation window open. Close the PyBullet window or press Ctrl+C in terminal to exit.")
    try:
        sim = env.unwrapped.sim
        while True:
            # Check if visualizer is still connected
            if not sim.physics_client.isConnected():
                break
            # Step the simulation slightly to keep interface active
            sim.physics_client.stepSimulation()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting simulation.")
    finally:
        env.close()

if __name__ == "__main__":
    test_single_episode_gui()
