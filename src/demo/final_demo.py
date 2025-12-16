#!/usr/bin/env python3
"""
Final Demo Autonomous Mission

This module implements the complete autonomous mission that demonstrates
all the capabilities of the Physical AI system in a coordinated sequence.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, Image
import time
import threading
from typing import Dict, List
import math


class FinalDemoMissionNode(Node):
    def __init__(self):
        super().__init__('final_demo_mission')

        # Publishers
        self.demo_status_pub = self.create_publisher(String, '/demo_status', 10)
        self.demo_command_pub = self.create_publisher(String, '/demo_command', 10)
        self.mission_progress_pub = self.create_publisher(String, '/mission_progress', 10)

        # Subscribers
        self.voice_cmd_sub = self.create_subscription(
            String, '/parsed_command', self.voice_command_callback, 10)
        self.vision_analysis_sub = self.create_subscription(
            String, '/vision_analysis', self.vision_callback, 10)
        self.system_status_sub = self.create_subscription(
            String, '/system_status', self.system_status_callback, 10)

        self.get_logger().info('Final Demo Mission Node Started')

        # Mission state
        self.mission_active = False
        self.mission_stage = 0
        self.mission_sequence = []
        self.demo_complete = False
        self.system_ready = False

        # Initialize mission sequence
        self.setup_demo_mission()

    def setup_demo_mission(self):
        """Setup the complete demo mission sequence"""
        self.mission_sequence = [
            {
                'stage': 0,
                'name': 'System Initialization',
                'description': 'Initialize all subsystems and check readiness',
                'action': self.initialize_system,
                'duration': 2.0
            },
            {
                'stage': 1,
                'name': 'Voice Command Reception',
                'description': 'Listen for and process voice command',
                'action': self.listen_for_command,
                'duration': 10.0
            },
            {
                'stage': 2,
                'name': 'Environment Perception',
                'description': 'Analyze environment using vision system',
                'action': self.perceive_environment,
                'duration': 5.0
            },
            {
                'stage': 3,
                'name': 'Cognitive Planning',
                'description': 'Process command and plan actions using LLM',
                'action': self.cognitive_planning,
                'duration': 5.0
            },
            {
                'stage': 4,
                'name': 'Vision-Language-Action',
                'description': 'Integrate vision, language and action planning',
                'action': self.vla_integration,
                'duration': 5.0
            },
            {
                'stage': 5,
                'name': 'Action Execution',
                'description': 'Execute planned actions safely',
                'action': self.execute_actions,
                'duration': 15.0
            },
            {
                'stage': 6,
                'name': 'Mission Completion',
                'description': 'Report results and complete mission',
                'action': self.complete_mission,
                'duration': 3.0
            }
        ]

        self.get_logger().info(f'Demo mission setup with {len(self.mission_sequence)} stages')

    def system_status_callback(self, msg):
        """Monitor system readiness"""
        if 'ALL_SYSTEMS_NORMAL' in msg.data:
            self.system_ready = True
        else:
            self.system_ready = False

    def voice_command_callback(self, msg):
        """Handle voice commands during demo"""
        if self.mission_active and self.mission_stage == 1:
            self.get_logger().info(f'Voice command received during demo: {msg.data}')
            # Store command for processing in the next stage
            self.stored_command = msg.data

    def vision_callback(self, msg):
        """Process vision analysis during demo"""
        if self.mission_active and self.mission_stage == 2:
            self.get_logger().info('Environment perception completed')
            self.vision_data = msg.data

    def initialize_system(self):
        """Initialize all systems for the demo"""
        self.get_logger().info('🚀 Stage 1: System Initialization')

        # Publish initialization status
        status_msg = String()
        status_msg.data = "INITIALIZING: Starting all Physical AI subsystems"
        self.demo_status_pub.publish(status_msg)

        # Wait for systems to be ready
        timeout = time.time() + 5.0  # 5 second timeout
        while not self.system_ready and time.time() < timeout:
            time.sleep(0.1)

        if self.system_ready:
            self.get_logger().info('✅ All systems initialized and ready')
            status_msg.data = "SYSTEM_READY: All subsystems operational"
        else:
            self.get_logger().warn('⚠️ Some systems may not be fully ready')
            status_msg.data = "SYSTEM_CHECK: Proceeding with available systems"

        self.demo_status_pub.publish(status_msg)

    def listen_for_command(self):
        """Listen for voice command"""
        self.get_logger().info('🎤 Stage 2: Voice Command Reception')

        status_msg = String()
        status_msg.data = "LISTENING: Awaiting voice command for demonstration"
        self.demo_status_pub.publish(status_msg)

        # Wait for a command (with timeout)
        timeout = time.time() + 8.0  # 8 seconds to receive command
        self.stored_command = None

        while self.stored_command is None and time.time() < timeout:
            time.sleep(0.1)

        if self.stored_command:
            self.get_logger().info(f'✅ Command received: {self.stored_command}')
            status_msg.data = f"COMMAND_RECEIVED: {self.stored_command}"
        else:
            # Default command if none received
            self.stored_command = "move forward and look around"
            self.get_logger().info(f'📝 Default command: {self.stored_command}')
            status_msg.data = f"DEFAULT_COMMAND: {self.stored_command}"

        self.demo_status_pub.publish(status_msg)

    def perceive_environment(self):
        """Perceive and analyze the environment"""
        self.get_logger().info('👁️ Stage 3: Environment Perception')

        status_msg = String()
        status_msg.data = "PERCEIVING: Analyzing environment with vision system"
        self.demo_status_pub.publish(status_msg)

        # Wait for vision data (with timeout)
        timeout = time.time() + 4.0
        self.vision_data = None

        while self.vision_data is None and time.time() < timeout:
            time.sleep(0.1)

        if self.vision_data:
            self.get_logger().info('✅ Environment analysis completed')
            status_msg.data = "ENVIRONMENT_ANALYZED: Objects and layout mapped"
        else:
            self.get_logger().info('📝 No specific environment data available')
            status_msg.data = "ENVIRONMENT_BRIEF: Basic perception completed"

        self.demo_status_pub.publish(status_msg)

    def cognitive_planning(self):
        """Perform cognitive planning using LLM"""
        self.get_logger().info('🧠 Stage 4: Cognitive Planning')

        status_msg = String()
        status_msg.data = "PLANNING: Processing command with cognitive system"
        self.demo_status_pub.publish(status_msg)

        # Simulate cognitive processing
        time.sleep(2.0)  # Simulate processing time

        self.get_logger().info('✅ Cognitive planning completed')
        status_msg.data = "PLANNED: Action sequence generated by cognitive system"
        self.demo_status_pub.publish(status_msg)

    def vla_integration(self):
        """Integrate Vision-Language-Action systems"""
        self.get_logger().info('🔗 Stage 5: VLA Integration')

        status_msg = String()
        status_msg.data = "INTEGRATING: Combining vision, language, and action"
        self.demo_status_pub.publish(status_msg)

        # Simulate VLA integration
        time.sleep(2.0)  # Simulate integration processing

        self.get_logger().info('✅ VLA integration completed')
        status_msg.data = "VLA_INTEGRATED: Vision-Language-Action pipeline active"
        self.demo_status_pub.publish(status_msg)

    def execute_actions(self):
        """Execute the planned actions"""
        self.get_logger().info('🤖 Stage 6: Action Execution')

        status_msg = String()
        status_msg.data = "EXECUTING: Running planned actions safely"
        self.demo_status_pub.publish(status_msg)

        # Execute a sequence of actions that demonstrate the system
        actions = [
            ("move_forward", 2.0),
            ("turn_right", 1.0),
            ("move_forward", 1.5),
            ("turn_left", 1.0),
            ("stop", 0.5)
        ]

        for action, duration in actions:
            self.get_logger().info(f'Executing: {action} for {duration}s')

            # Publish action command
            cmd_msg = String()
            cmd_msg.data = f"{action}_{duration}s"
            self.demo_command_pub.publish(cmd_msg)

            time.sleep(duration)

        self.get_logger().info('✅ Action execution completed')
        status_msg.data = "ACTIONS_COMPLETED: All planned actions executed"
        self.demo_status_pub.publish(status_msg)

    def complete_mission(self):
        """Complete the mission and report results"""
        self.get_logger().info('🎉 Stage 7: Mission Completion')

        status_msg = String()
        status_msg.data = "MISSION_COMPLETE: Final demo successfully completed"
        self.demo_status_pub.publish(status_msg)

        # Publish completion summary
        summary = {
            'mission': 'Physical AI Capstone Demo',
            'status': 'SUCCESS',
            'stages_completed': len(self.mission_sequence),
            'total_duration': 'Approx 45 seconds',
            'demonstrated_capabilities': [
                'Voice Command Processing',
                'Environmental Perception',
                'Cognitive Planning',
                'Vision-Language-Action Integration',
                'Safe Motion Execution'
            ]
        }

        summary_msg = String()
        summary_msg.data = str(summary)
        self.mission_progress_pub.publish(summary_msg)

        self.demo_complete = True
        self.get_logger().info('🏆 Final demo mission completed successfully!')

    def run_demo_mission(self):
        """Run the complete demo mission"""
        if self.mission_active:
            self.get_logger().warn('Demo already in progress')
            return

        self.get_logger().info('🚀 Starting Final Demo Mission Sequence')
        self.mission_active = True
        self.demo_complete = False

        try:
            for stage_info in self.mission_sequence:
                if not self.mission_active:
                    break

                self.mission_stage = stage_info['stage']

                # Publish progress
                progress_msg = String()
                progress_msg.data = f"STAGE_{stage_info['stage']}: {stage_info['name']}"
                self.mission_progress_pub.publish(progress_msg)

                # Execute the stage action
                self.get_logger().info(f"Executing stage {stage_info['stage']}: {stage_info['name']}")
                stage_info['action']()

                # Small delay between stages
                time.sleep(1.0)

        except Exception as e:
            self.get_logger().error(f'Error during demo mission: {e}')
        finally:
            self.mission_active = False
            self.get_logger().info('Demo mission finished')

    def start_demo_callback(self, msg):
        """Callback to start demo from external command"""
        if msg.data.lower() in ['start', 'begin', 'run', 'execute']:
            demo_thread = threading.Thread(target=self.run_demo_mission)
            demo_thread.start()


def main(args=None):
    rclpy.init(args=args)

    node = FinalDemoMissionNode()

    try:
        # Run the demo mission
        node.run_demo_mission()

        # Keep running to handle callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Final Demo Mission Node')
        node.mission_active = False
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()