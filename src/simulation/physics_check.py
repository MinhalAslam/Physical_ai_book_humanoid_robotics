#!/usr/bin/env python3
"""
Physics Simulation Checker Node

This node verifies the physics simulation environment and checks
that all necessary components are properly configured.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image, CameraInfo
import time


class PhysicsSimulationChecker(Node):
    def __init__(self):
        super().__init__('physics_simulation_checker')

        # Create publishers for robot control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create subscribers for sensor data
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Timer to periodically check simulation status
        self.timer = self.create_timer(1.0, self.check_simulation_status)

        self.get_logger().info('Physics Simulation Checker Node Started')

        # Status flags
        self.laser_received = False
        self.camera_received = False
        self.simulation_ready = False

    def scan_callback(self, msg):
        """Callback for laser scan data"""
        self.laser_received = True
        self.get_logger().debug('Received laser scan data')

    def image_callback(self, msg):
        """Callback for camera image data"""
        self.camera_received = True
        self.get_logger().debug('Received camera image data')

    def check_simulation_status(self):
        """Check if simulation is properly configured"""
        if self.laser_received and self.camera_received:
            if not self.simulation_ready:
                self.simulation_ready = True
                self.get_logger().info('✅ Simulation environment verified - All sensors connected')
        else:
            if self.simulation_ready:
                self.simulation_ready = False
                self.get_logger().info('⚠️ Simulation environment check - Waiting for sensors')

    def move_robot(self, linear_x=0.0, angular_z=0.0):
        """Send movement command to robot"""
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)

    node = PhysicsSimulationChecker()

    try:
        # Test basic movement after 2 seconds
        def test_movement():
            node.get_logger().info('Testing robot movement...')
            node.move_robot(linear_x=0.5)  # Move forward
            time.sleep(2)
            node.move_robot(linear_x=0.0)  # Stop

        # Schedule test after 3 seconds
        timer = node.create_timer(3.0, test_movement)

        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Physics Simulation Checker')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()