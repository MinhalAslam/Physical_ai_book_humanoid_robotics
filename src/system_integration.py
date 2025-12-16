#!/usr/bin/env python3
"""
System Architecture Integration

This module integrates all the individual components into a unified system
that demonstrates the complete Physical AI capabilities.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
import time
import threading
from typing import Dict, List


class SystemIntegrationNode(Node):
    def __init__(self):
        super().__init__('system_integration')

        # Publishers for integrated system
        self.system_status_pub = self.create_publisher(String, '/system_status', 10)
        self.integrated_command_pub = self.create_publisher(String, '/integrated_command', 10)

        # Subscribers from various subsystems
        self.vision_sub = self.create_subscription(
            String, '/vision_analysis', self.vision_callback, 10)
        self.speech_sub = self.create_subscription(
            String, '/speech_text', self.speech_callback, 10)
        self.llm_sub = self.create_subscription(
            String, '/llm_thought', self.llm_callback, 10)
        self.vla_sub = self.create_subscription(
            String, '/vla_output', self.vla_callback, 10)

        # Timer for system monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_system)

        self.get_logger().info('System Integration Node Started')

        # System state tracking
        self.system_components = {
            'vision': {'active': False, 'last_update': 0},
            'speech': {'active': False, 'last_update': 0},
            'llm': {'active': False, 'last_update': 0},
            'vla': {'active': False, 'last_update': 0},
            'motion': {'active': False, 'last_update': 0}
        }

        self.integration_active = True

    def vision_callback(self, msg):
        """Handle vision system updates"""
        self.system_components['vision']['active'] = True
        self.system_components['vision']['last_update'] = time.time()
        self.get_logger().debug('Vision system update received')

    def speech_callback(self, msg):
        """Handle speech system updates"""
        self.system_components['speech']['active'] = True
        self.system_components['speech']['last_update'] = time.time()
        self.get_logger().debug('Speech system update received')

    def llm_callback(self, msg):
        """Handle LLM system updates"""
        self.system_components['llm']['active'] = True
        self.system_components['llm']['last_update'] = time.time()
        self.get_logger().debug('LLM system update received')

    def vla_callback(self, msg):
        """Handle VLA system updates"""
        self.system_components['vla']['active'] = True
        self.system_components['vla']['last_update'] = time.time()
        self.get_logger().debug('VLA system update received')

    def monitor_system(self):
        """Monitor the integrated system status"""
        active_components = sum(1 for comp in self.system_components.values() if comp['active'])
        total_components = len(self.system_components)

        status_msg = String()
        status_msg.data = f"SYSTEM_INTEGRATION: {active_components}/{total_components} components active"

        # Check for system health
        current_time = time.time()
        inactive_components = []
        for name, comp in self.system_components.items():
            if not comp['active'] or (current_time - comp['last_update']) > 5.0:  # 5 seconds timeout
                inactive_components.append(name)

        if inactive_components:
            status_msg.data += f" | INACTIVE: {', '.join(inactive_components)}"
            self.get_logger().warn(f"Inactive components: {', '.join(inactive_components)}")
        else:
            status_msg.data += " | ALL_SYSTEMS_NORMAL"

        self.system_status_pub.publish(status_msg)

        # If all systems are active, demonstrate integration
        if active_components == total_components and self.integration_active:
            self.demonstrate_integration()

    def demonstrate_integration(self):
        """Demonstrate the integration of all systems"""
        self.get_logger().info('Demonstrating full system integration...')

        # Example integration scenario: Process a command through all systems
        integration_demo = {
            'timestamp': time.time(),
            'scenario': 'Integrated Command Processing',
            'components_involved': list(self.system_components.keys()),
            'status': 'SUCCESS',
            'description': 'All systems working together in harmony'
        }

        demo_msg = String()
        demo_msg.data = str(integration_demo)
        self.integrated_command_pub.publish(demo_msg)

    def shutdown_integration(self):
        """Properly shutdown the integration system"""
        self.integration_active = False
        self.get_logger().info('System integration shutting down...')


def main(args=None):
    rclpy.init(args=args)

    node = SystemIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down System Integration Node')
        node.shutdown_integration()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()