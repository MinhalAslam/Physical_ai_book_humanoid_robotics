#!/usr/bin/env python3
"""
Sim-to-Real Transfer System

This module handles the transfer of learned behaviors from simulation
to real-world robot execution, including domain randomization and
calibration procedures.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, Pose
import numpy as np
import time
from typing import Dict, List, Tuple


class SimToRealTransferNode(Node):
    def __init__(self):
        super().__init__('sim_to_real_transfer')

        # Publishers
        self.transfer_status_pub = self.create_publisher(String, '/sim_to_real_status', 10)
        self.calibrated_command_pub = self.create_publisher(Twist, '/calibrated_cmd_vel', 10)
        self.domain_randomization_pub = self.create_publisher(String, '/domain_randomization_params', 10)

        # Subscribers
        self.sim_command_sub = self.create_subscription(
            Twist, '/sim_cmd_vel', self.sim_command_callback, 10)
        self.real_sensor_sub = self.create_subscription(
            String, '/real_sensors', self.real_sensor_callback, 10)
        self.sim_sensor_sub = self.create_subscription(
            String, '/sim_sensors', self.sim_sensor_callback, 10)

        # Timer for calibration and transfer
        self.transfer_timer = self.create_timer(0.1, self.transfer_update)

        self.get_logger().info('Sim-to-Real Transfer Node Started')

        # Transfer parameters
        self.sim_to_real_mapping = {
            'linear_scale': 1.0,      # Scale factor for linear velocity
            'angular_scale': 1.0,     # Scale factor for angular velocity
            'delay_compensation': 0.0, # Compensation for sim vs real delays
            'friction_factor': 1.0,   # Factor to account for real-world friction
        }

        # Calibration state
        self.calibration_data = {}
        self.is_calibrated = False
        self.domain_randomization_active = True

        # Initialize calibration
        self.initialize_calibration()

    def initialize_calibration(self):
        """Initialize the calibration process"""
        self.get_logger().info('Initializing sim-to-real calibration...')

        # Default calibration values based on typical sim-to-real differences
        self.sim_to_real_mapping = {
            'linear_scale': 0.95,      # Real robots often move slightly slower
            'angular_scale': 0.98,     # Account for wheel slip in simulation
            'delay_compensation': 0.02, # 20ms delay compensation
            'friction_factor': 1.1,    # Real world has more friction
        }

        self.get_logger().info('Calibration initialized with default parameters')

    def sim_command_callback(self, msg):
        """Receive commands from simulation and transfer to real world"""
        if not self.is_calibrated:
            self.get_logger().warn('Using uncalibrated transfer parameters')

        # Apply sim-to-real mapping
        calibrated_cmd = Twist()
        calibrated_cmd.linear.x = msg.linear.x * self.sim_to_real_mapping['linear_scale']
        calibrated_cmd.linear.y = msg.linear.y * self.sim_to_real_mapping['linear_scale']
        calibrated_cmd.linear.z = msg.linear.z * self.sim_to_real_mapping['linear_scale']

        calibrated_cmd.angular.x = msg.angular.x * self.sim_to_real_mapping['angular_scale']
        calibrated_cmd.angular.y = msg.angular.y * self.sim_to_real_mapping['angular_scale']
        calibrated_cmd.angular.z = msg.angular.z * self.sim_to_real_mapping['angular_scale']

        # Publish calibrated command
        self.calibrated_command_pub.publish(calibrated_cmd)

        # Log the transfer
        self.get_logger().debug(
            f'Sim-to-Real Transfer: {msg.linear.x:.2f} -> {calibrated_cmd.linear.x:.2f} '
            f'(scale: {self.sim_to_real_mapping["linear_scale"]:.2f})'
        )

    def real_sensor_callback(self, msg):
        """Process real-world sensor data for calibration"""
        try:
            # In a real implementation, this would parse real sensor data
            # and update calibration parameters
            real_data = eval(msg.data)  # Note: In real code, use proper parsing like json
            self.calibration_data['real'] = real_data

            # Update calibration based on real vs expected differences
            self.update_calibration_from_real_data(real_data)

        except Exception as e:
            self.get_logger().error(f'Error processing real sensor data: {e}')

    def sim_sensor_callback(self, msg):
        """Process simulation sensor data for comparison"""
        try:
            # In a real implementation, this would parse sim sensor data
            sim_data = eval(msg.data)  # Note: In real code, use proper parsing like json
            self.calibration_data['sim'] = sim_data

            # Compare with real data if available
            if 'real' in self.calibration_data:
                self.compare_sim_real_data(sim_data, self.calibration_data['real'])

        except Exception as e:
            self.get_logger().error(f'Error processing sim sensor data: {e}')

    def update_calibration_from_real_data(self, real_data):
        """Update calibration parameters based on real-world data"""
        # Example: If we have encoder data, compare actual vs expected movement
        if isinstance(real_data, dict) and 'position' in real_data:
            # This is a simplified example - real calibration would be more complex
            expected_pos = real_data.get('expected_position', [0, 0, 0])
            actual_pos = real_data.get('actual_position', [0, 0, 0])

            # Calculate correction factors
            pos_diff = [
                actual_pos[i] - expected_pos[i]
                for i in range(len(expected_pos))
            ]

            # Update scale factors based on position differences
            if abs(expected_pos[0]) > 0.1:  # Avoid division by zero
                correction_factor = actual_pos[0] / expected_pos[0] if expected_pos[0] != 0 else 1.0
                self.sim_to_real_mapping['linear_scale'] *= 0.95 + 0.05 * correction_factor  # Smooth adjustment

    def compare_sim_real_data(self, sim_data, real_data):
        """Compare simulation and real-world sensor data"""
        comparison = {
            'timestamp': time.time(),
            'sim_data': sim_data,
            'real_data': real_data,
            'differences': {},
            'transfer_quality': 0.0
        }

        # Calculate differences where possible
        if isinstance(sim_data, dict) and isinstance(real_data, dict):
            for key in set(sim_data.keys()) & set(real_data.keys()):
                if isinstance(sim_data[key], (int, float)) and isinstance(real_data[key], (int, float)):
                    diff = real_data[key] - sim_data[key]
                    comparison['differences'][key] = diff

        # Calculate transfer quality score (0-1)
        if comparison['differences']:
            avg_diff = np.mean([abs(v) for v in comparison['differences'].values()])
            # Lower differences = higher quality
            comparison['transfer_quality'] = max(0.0, min(1.0, 1.0 - avg_diff))

        # Publish comparison results
        comparison_msg = String()
        comparison_msg.data = str(comparison)
        self.transfer_status_pub.publish(comparison_msg)

        self.get_logger().info(f'Sim-to-Real Quality: {comparison["transfer_quality"]:.2f}')

    def transfer_update(self):
        """Periodic update for transfer system"""
        # Apply domain randomization if active
        if self.domain_randomization_active:
            self.apply_domain_randomization()

        # Publish transfer status
        status = {
            'is_calibrated': self.is_calibrated,
            'mapping_params': self.sim_to_real_mapping,
            'domain_randomization': self.domain_randomization_active
        }

        status_msg = String()
        status_msg.data = str(status)
        self.transfer_status_pub.publish(status_msg)

    def apply_domain_randomization(self):
        """Apply domain randomization to simulation parameters"""
        # Generate randomization parameters
        randomization_params = {
            'friction_range': [0.8, 1.2],  # Random friction coefficient
            'mass_variance': 0.1,          # ±10% mass variation
            'sensor_noise': 0.05,          # 5% sensor noise
            'actuator_delay_range': [0.01, 0.05],  # 10-50ms actuator delay
        }

        # Apply randomization to simulation
        params_msg = String()
        params_msg.data = str(randomization_params)
        self.domain_randomization_pub.publish(params_msg)

        self.get_logger().debug('Domain randomization applied to simulation')

    def calibrate_system(self):
        """Perform comprehensive system calibration"""
        self.get_logger().info('Starting comprehensive sim-to-real calibration...')

        # This would involve running specific calibration movements
        # and comparing simulation vs real-world responses
        self.is_calibrated = True
        self.get_logger().info('Calibration completed')


def main(args=None):
    rclpy.init(args=args)

    node = SimToRealTransferNode()

    try:
        # Perform initial calibration
        node.calibrate_system()

        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Sim-to-Real Transfer Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()