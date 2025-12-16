#!/usr/bin/env python3
"""
First Motion Node

This node implements the first basic motion capabilities for the robot,
including simple movement commands and safety checks.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
import math


class MotionNode(Node):
    def __init__(self):
        super().__init__('motion_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/motion_status', 10)

        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, '/motion_cmd', self.motion_cmd_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop', self.emergency_stop_callback, 10)

        # Timer for safety checks
        self.safety_timer = self.create_timer(0.1, self.safety_check)

        self.get_logger().info('Motion Node Started')

        # Robot state
        self.current_twist = Twist()
        self.emergency_stop_active = False
        self.obstacle_detected = False
        self.obstacle_distance = float('inf')
        self.safety_threshold = 0.5  # meters

    def motion_cmd_callback(self, msg):
        """Handle incoming motion commands"""
        if not self.emergency_stop_active and not self.obstacle_detected:
            self.current_twist = msg
            self.cmd_vel_pub.publish(msg)
            self.get_logger().debug(f'Motion command: linear={msg.linear}, angular={msg.angular}')
        else:
            # Stop robot if safety conditions are violated
            stop_twist = Twist()
            self.cmd_vel_pub.publish(stop_twist)
            self.current_twist = stop_twist

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        # Check for obstacles in front of the robot (within 30 degrees)
        front_ranges = msg.ranges[:15] + msg.ranges[-15:]  # Front 30 degrees

        valid_ranges = [r for r in front_ranges if r > 0 and not math.isinf(r)]
        if valid_ranges:
            self.obstacle_distance = min(valid_ranges)
            self.obstacle_detected = self.obstacle_distance < self.safety_threshold
        else:
            self.obstacle_distance = float('inf')
            self.obstacle_detected = False

    def emergency_stop_callback(self, msg):
        """Handle emergency stop commands"""
        self.emergency_stop_active = msg.data
        if msg.data:
            # Immediately stop the robot
            stop_twist = Twist()
            self.cmd_vel_pub.publish(stop_twist)
            self.current_twist = stop_twist
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')

    def safety_check(self):
        """Perform periodic safety checks"""
        status_msg = String()

        if self.emergency_stop_active:
            status_msg.data = "EMERGENCY_STOP"
        elif self.obstacle_detected:
            status_msg.data = f"OBSTACLE_DETECTED_{self.obstacle_distance:.2f}m"
        else:
            status_msg.data = "SAFE"

        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)

    node = MotionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Motion Node')
    finally:
        # Ensure robot stops when shutting down
        stop_twist = Twist()
        node.cmd_vel_pub.publish(stop_twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()