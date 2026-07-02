#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class KukaTrajectoryClient(Node):

    def __init__(self):
        super().__init__('kuka_trajectory_client')
        self._action_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

    def send_goal(self):
        self.get_logger().info('Waiting for arm_controller action server...')
        self._action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        # Official joint names
        goal_msg.trajectory.joint_names = [
            'joint_a1', 'joint_a2', 'joint_a3', 'joint_a4', 'joint_a5', 'joint_a6'
        ]

        # Define trajectory points
        # Point 1: Home position (all 0s)
        p1 = JointTrajectoryPoint()
        p1.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p1.time_from_start = Duration(sec=2, nanosec=0)

        # Point 2: Wave left (joint_a1 rotated)
        p2 = JointTrajectoryPoint()
        p2.positions = [1.0, -0.5, 0.5, 0.0, 0.5, 0.0]
        p2.time_from_start = Duration(sec=5, nanosec=0)

        # Point 3: Wave right (joint_a1 rotated opposite)
        p3 = JointTrajectoryPoint()
        p3.positions = [-1.0, -0.5, 0.5, 0.0, -0.5, 0.0]
        p3.time_from_start = Duration(sec=8, nanosec=0)

        # Point 4: Return to home
        p4 = JointTrajectoryPoint()
        p4.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p4.time_from_start = Duration(sec=11, nanosec=0)

        goal_msg.trajectory.points = [p1, p2, p3, p4]

        self.get_logger().info('Sending trajectory goal to arm_controller...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by arm_controller :(')
            return

        self.get_logger().info('Goal accepted! Executing trajectory...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Trajectory execution complete!')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    client = KukaTrajectoryClient()
    client.send_goal()
    rclpy.spin(client)

if __name__ == '__main__':
    main()
