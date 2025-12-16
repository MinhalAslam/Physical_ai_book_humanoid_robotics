#!/usr/bin/env python3
"""
Robot State Machine

This module implements the robot's state machine with various safety states
and state transition logic.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from enum import Enum
import time


class RobotState(Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    OPERATIONAL = "OPERATIONAL"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SAFETY_OVERRIDE = "SAFETY_OVERRIDE"
    CHARGING = "CHARGING"
    ERROR = "ERROR"


class RobotStateMachine(Node):
    def __init__(self):
        super().__init__('robot_state_machine')

        # Publisher for state updates
        self.state_pub = self.create_publisher(String, '/robot_state', 10)

        # Initialize state
        self.current_state = RobotState.INITIALIZING
        self.previous_state = None

        # Timer for state monitoring
        self.state_timer = self.create_timer(0.5, self.state_monitor)

        self.get_logger().info('Robot State Machine Node Started')

        # Initialize the state machine
        self.initialize_robot()

    def initialize_robot(self):
        """Initialize the robot and transition to operational state"""
        self.get_logger().info('Initializing robot systems...')

        # Simulate initialization process
        time.sleep(1)  # Simulate hardware initialization

        self.transition_to_state(RobotState.OPERATIONAL)
        self.get_logger().info('Robot initialization complete')

    def transition_to_state(self, new_state):
        """Safely transition to a new state"""
        self.previous_state = self.current_state
        self.current_state = new_state

        # Publish state change
        state_msg = String()
        state_msg.data = f"STATE_CHANGE: {self.previous_state.value} -> {self.current_state.value}"
        self.state_pub.publish(state_msg)

        self.get_logger().info(f'State transition: {self.previous_state.value} -> {self.current_state.value}')

    def state_monitor(self):
        """Monitor robot state and handle transitions"""
        # Publish current state
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)

        # Example state logic - in a real system this would be more complex
        if self.current_state == RobotState.ERROR:
            # Attempt recovery from error state
            self.get_logger().warn('Attempting error recovery...')
            self.transition_to_state(RobotState.IDLE)

    def trigger_emergency_stop(self):
        """Trigger emergency stop state"""
        if self.current_state != RobotState.EMERGENCY_STOP:
            self.transition_to_state(RobotState.EMERGENCY_STOP)

    def clear_emergency_stop(self):
        """Clear emergency stop and return to safe state"""
        if self.current_state == RobotState.EMERGENCY_STOP:
            self.transition_to_state(RobotState.IDLE)

    def request_operational_state(self):
        """Request transition to operational state"""
        if self.current_state in [RobotState.IDLE, RobotState.CHARGING]:
            self.transition_to_state(RobotState.OPERATIONAL)


def main(args=None):
    rclpy.init(args=args)

    node = RobotStateMachine()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Robot State Machine')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()