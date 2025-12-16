#!/usr/bin/env python3
"""
Human-Robot Interaction Scene Manager

This node manages human-robot interaction scenarios in simulation,
including scene setup, interaction tracking, and safety monitoring.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import LaserScan
from builtin_interfaces.msg import Time
import time
import math


class HRISceneManager(Node):
    def __init__(self):
        super().__init__('hri_scene_manager')

        # Publishers
        self.scene_status_pub = self.create_publisher(String, '/hri_scene_status', 10)
        self.interaction_alert_pub = self.create_publisher(String, '/hri_interaction_alert', 10)
        self.proximity_warning_pub = self.create_publisher(Bool, '/hri_proximity_warning', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.human_pose_sub = self.create_subscription(
            Pose, '/human_pose', self.human_pose_callback, 10)

        # Timer for scene monitoring
        self.monitor_timer = self.create_timer(0.2, self.monitor_scene)

        self.get_logger().info('HRI Scene Manager Node Started')

        # Scene state
        self.human_detected = False
        self.human_distance = float('inf')
        self.human_pose = None
        self.interaction_zone = 2.0  # meters
        self.safety_zone = 0.5  # meters
        self.scene_status = "IDLE"

    def laser_callback(self, msg):
        """Process laser scan to detect humans in the environment"""
        # Simple human detection based on laser scan patterns
        # In a real system, this would use more sophisticated detection
        valid_ranges = [r for r in msg.ranges if r > 0 and not math.isinf(r) and not math.isnan(r)]

        if valid_ranges:
            min_distance = min(valid_ranges)
            self.human_distance = min_distance

            # If something is within interaction range, assume it's a human
            if min_distance <= self.interaction_zone:
                self.human_detected = True
                self.scene_status = "HUMAN_DETECTED"
            else:
                self.human_detected = False
                self.scene_status = "SEARCHING"
        else:
            self.human_detected = False
            self.human_distance = float('inf')
            self.scene_status = "IDLE"

    def human_pose_callback(self, msg):
        """Receive precise human pose if available from tracking system"""
        self.human_pose = msg
        self.human_detected = True
        self.scene_status = "HUMAN_TRACKED"

    def monitor_scene(self):
        """Monitor the HRI scene and publish status updates"""
        # Create scene status message
        status_msg = String()
        status_msg.data = f"{self.scene_status}: DISTANCE_{self.human_distance:.2f}m"
        self.scene_status_pub.publish(status_msg)

        # Check for safety violations
        if self.human_detected and self.human_distance <= self.safety_zone:
            warning_msg = Bool()
            warning_msg.data = True
            self.proximity_warning_pub.publish(warning_msg)

            alert_msg = String()
            alert_msg.data = f"SAFETY_VIOLATION: Human too close - {self.human_distance:.2f}m"
            self.interaction_alert_pub.publish(alert_msg)

            self.get_logger().warn(f'Safety zone violation: Human at {self.human_distance:.2f}m')
        else:
            # Publish safety clear
            warning_msg = Bool()
            warning_msg.data = False
            self.proximity_warning_pub.publish(warning_msg)

        # Log scene status periodically
        if self.human_detected:
            self.get_logger().info(
                f'HRI Scene: {self.scene_status}, Human distance: {self.human_distance:.2f}m'
            )

    def setup_interaction_scenario(self, scenario_name):
        """Setup a specific HRI scenario"""
        self.get_logger().info(f'Setting up HRI scenario: {scenario_name}')

        # In a real implementation, this would configure the simulation
        # environment for specific interaction scenarios
        scenarios = {
            'greeting': {'distance': 1.5, 'orientation': 'frontal'},
            'following': {'distance': 1.0, 'orientation': 'lateral'},
            'handover': {'distance': 0.8, 'orientation': 'frontal'},
            'navigation': {'distance': 2.0, 'orientation': 'any'}
        }

        if scenario_name in scenarios:
            config = scenarios[scenario_name]
            self.interaction_zone = config['distance'] + 0.5  # Add buffer
            self.get_logger().info(f'Scenario {scenario_name} configured with distance {config["distance"]}m')
            return True
        else:
            self.get_logger().warn(f'Unknown scenario: {scenario_name}')
            return False


def main(args=None):
    rclpy.init(args=args)

    node = HRISceneManager()

    try:
        # Setup default greeting scenario
        node.setup_interaction_scenario('greeting')

        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down HRI Scene Manager')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()