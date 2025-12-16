#!/usr/bin/env python3
"""
Emergency Stop and Safety System

This module implements the core safety systems and emergency stop functionality.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
import time


class SafetySystem(Node):
    def __init__(self):
        super().__init__('safety_system')

        # Publishers
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.safety_status_pub = self.create_publisher(String, '/safety_status', 10)
        self.stop_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.emergency_stop_sub = self.create_subscription(
            Bool, '/emergency_stop_request', self.emergency_stop_callback, 10)

        # Initialize safety state
        self.emergency_stop_active = False
        self.safety_enabled = True

        # Timer for safety monitoring
        self.safety_timer = self.create_timer(0.1, self.safety_monitor)

        self.get_logger().info('Safety System Node Started')

    def emergency_stop_callback(self, msg):
        """Handle emergency stop requests"""
        if msg.data:
            self.activate_emergency_stop()
        else:
            self.deactivate_emergency_stop()

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True

            # Publish emergency stop signal
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_pub.publish(stop_msg)

            # Send stop command to robot
            stop_twist = Twist()
            self.stop_cmd_pub.publish(stop_twist)

            self.get_logger().error('EMERGENCY STOP ACTIVATED')

            # Publish safety status
            status_msg = String()
            status_msg.data = "EMERGENCY_STOP_ACTIVE"
            self.safety_status_pub.publish(status_msg)

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        if self.emergency_stop_active:
            self.emergency_stop_active = False

            # Publish emergency stop clear signal
            clear_msg = Bool()
            clear_msg.data = False
            self.emergency_stop_pub.publish(clear_msg)

            self.get_logger().info('EMERGENCY STOP CLEARED')

            # Publish safety status
            status_msg = String()
            status_msg.data = "EMERGENCY_STOP_CLEARED"
            self.safety_status_pub.publish(status_msg)

    def safety_monitor(self):
        """Monitor safety status"""
        if self.safety_enabled:
            status_msg = String()
            status_msg.data = "SAFE" if not self.emergency_stop_active else "EMERGENCY_STOP"
            self.safety_status_pub.publish(status_msg)

    def request_emergency_stop(self):
        """Request emergency stop from within the system"""
        self.activate_emergency_stop()

    def is_safe_to_operate(self):
        """Check if it's safe to operate"""
        return self.safety_enabled and not self.emergency_stop_active


def main(args=None):
    rclpy.init(args=args)

    node = SafetySystem()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Safety System')
    finally:
        # Ensure emergency stop is cleared on shutdown
        if node.emergency_stop_active:
            node.deactivate_emergency_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()