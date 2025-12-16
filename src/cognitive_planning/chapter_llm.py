#!/usr/bin/env python3
"""
Language Model Thinking System

This node implements cognitive planning using language models
to understand commands and generate action sequences.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Twist
import json
import time
import re
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TaskStep:
    """Represents a single step in a task sequence"""
    action: str
    parameters: Dict[str, any]
    priority: int = 1
    requires_confirmation: bool = False


class LLMThinkingSystem(Node):
    def __init__(self):
        super().__init__('llm_thinking_system')

        # Publishers
        self.thought_pub = self.create_publisher(String, '/llm_thought', 10)
        self.task_sequence_pub = self.create_publisher(String, '/task_sequence', 10)
        self.action_pub = self.create_publisher(String, '/planned_action', 10)
        self.confirmation_request_pub = self.create_publisher(String, '/confirmation_request', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/parsed_command', self.command_callback, 10)
        self.confirmation_sub = self.create_subscription(
            Bool, '/confirmation_response', self.confirmation_callback, 10)

        self.get_logger().info('LLM Thinking System Node Started')

        # Task planning state
        self.current_thought = ""
        self.task_queue = []
        self.waiting_for_confirmation = False
        self.confirmation_callback_func = None

    def command_callback(self, msg):
        """Process incoming commands through LLM reasoning"""
        command = msg.data
        self.get_logger().info(f'Received command for processing: {command}')

        # Generate thought process
        thought = self.generate_thought(command)
        self.current_thought = thought

        # Publish current thought
        thought_msg = String()
        thought_msg.data = thought
        self.thought_pub.publish(thought_msg)

        # Plan task sequence
        task_sequence = self.plan_task_sequence(command, thought)

        # Process the task sequence
        self.process_task_sequence(task_sequence)

    def generate_thought(self, command: str) -> str:
        """Generate a thought process for the given command"""
        # This is a simplified version - in reality, this would use an actual LLM
        thought_process = f"""
Thought Process for Command: "{command}"

1. Understanding: The user wants me to {self.interpret_command(command)}
2. Context: Current robot state is [IDLE], environment is [SAFE]
3. Constraints: Safety protocols active, battery level [HIGH]
4. Plan: {self.generate_plan(command)}
5. Risks: {self.assess_risks(command)}
6. Confirmation: {self.requires_confirmation(command)}

Reasoning: {self.detailed_reasoning(command)}
        """.strip()

        return thought_process

    def interpret_command(self, command: str) -> str:
        """Interpret what the command means"""
        interpretations = {
            'move_forward': 'move forward by a safe distance',
            'move_backward': 'move backward by a safe distance',
            'turn_left': 'turn left by 90 degrees',
            'turn_right': 'turn right by 90 degrees',
            'stop': 'stop all movement immediately',
            'follow_me': 'activate person following mode',
            'come_here': 'navigate to the user\'s location',
            'find_person': 'search for and locate a person',
            'take_picture': 'capture an image with the camera',
            'what_time': 'report the current time',
            'hello': 'greet the user appropriately',
            'goodbye': 'terminate interaction politely'
        }

        for cmd_type, interpretation in interpretations.items():
            if cmd_type in command:
                return interpretation

        return f'perform the requested action: {command}'

    def generate_plan(self, command: str) -> str:
        """Generate a high-level plan for the command"""
        plans = {
            'move_forward': 'Check surroundings → Move forward 1m → Stop → Report completion',
            'move_backward': 'Check surroundings → Move backward 1m → Stop → Report completion',
            'turn_left': 'Check surroundings → Turn left 90° → Stop → Report completion',
            'turn_right': 'Check surroundings → Turn right 90° → Stop → Report completion',
            'follow_me': 'Detect person → Calculate following distance → Start following → Monitor distance',
            'come_here': 'Detect user location → Plan navigation path → Execute navigation → Confirm arrival',
            'find_person': 'Scan environment → Detect humans → Report location → Wait for instruction'
        }

        for cmd_type, plan in plans.items():
            if cmd_type in command:
                return plan

        return 'Analyze command → Plan execution → Execute safely → Report result'

    def assess_risks(self, command: str) -> str:
        """Assess potential risks for the command"""
        if 'move' in command or 'turn' in command or 'go' in command:
            return 'Collision with obstacles, navigation errors'
        elif 'follow' in command:
            return 'Getting too close to person, losing track'
        elif 'find' in command:
            return 'Inefficient search pattern, missing target'
        else:
            return 'None identified'

    def requires_confirmation(self, command: str) -> str:
        """Determine if command requires user confirmation"""
        confirmation_commands = ['follow_me', 'come_here', 'find_person', 'take_picture']

        for cmd in confirmation_commands:
            if cmd in command:
                return 'YES - High-level action requiring confirmation'

        return 'NO - Simple movement command'

    def detailed_reasoning(self, command: str) -> str:
        """Provide detailed reasoning for the command"""
        reasoning = f"""
For the command '{command}', I need to consider:
- Safety: Ensure no obstacles in path
- Accuracy: Execute movement precisely
- Efficiency: Minimize time and energy
- Feedback: Report status to user
        """.strip()

        return reasoning

    def plan_task_sequence(self, command: str, thought: str) -> List[TaskStep]:
        """Convert command and thought into a sequence of executable tasks"""
        # Define task mappings
        task_mappings = {
            'move_forward': [
                TaskStep('check_sensors', {}, 1),
                TaskStep('move_robot', {'linear_x': 1.0, 'angular_z': 0.0}, 2),
                TaskStep('wait', {'duration': 2.0}, 3),
                TaskStep('stop_robot', {}, 4)
            ],
            'move_backward': [
                TaskStep('check_sensors', {}, 1),
                TaskStep('move_robot', {'linear_x': -1.0, 'angular_z': 0.0}, 2),
                TaskStep('wait', {'duration': 2.0}, 3),
                TaskStep('stop_robot', {}, 4)
            ],
            'turn_left': [
                TaskStep('check_sensors', {}, 1),
                TaskStep('move_robot', {'linear_x': 0.0, 'angular_z': 0.5}, 2),
                TaskStep('wait', {'duration': 3.0}, 3),
                TaskStep('stop_robot', {}, 4)
            ],
            'turn_right': [
                TaskStep('check_sensors', {}, 1),
                TaskStep('move_robot', {'linear_x': 0.0, 'angular_z': -0.5}, 2),
                TaskStep('wait', {'duration': 3.0}, 3),
                TaskStep('stop_robot', {}, 4)
            ],
            'stop': [
                TaskStep('stop_robot', {}, 1)
            ],
            'follow_me': [
                TaskStep('activate_tracking', {}, 1, True),
                TaskStep('start_following', {}, 2, True),
                TaskStep('monitor_distance', {}, 3)
            ],
            'come_here': [
                TaskStep('detect_user', {}, 1, True),
                TaskStep('plan_path_to_user', {}, 2, True),
                TaskStep('navigate_to_user', {}, 3, True),
                TaskStep('confirm_arrival', {}, 4)
            ],
            'find_person': [
                TaskStep('activate_detection', {}, 1, True),
                TaskStep('scan_environment', {}, 2),
                TaskStep('detect_persons', {}, 3),
                TaskStep('report_findings', {}, 4)
            ]
        }

        # Look for matching command
        for cmd_type, tasks in task_mappings.items():
            if cmd_type in command:
                return tasks

        # Default task for unrecognized commands
        return [TaskStep('unknown_command', {'command': command}, 1)]

    def process_task_sequence(self, task_sequence: List[TaskStep]):
        """Process the planned task sequence"""
        # Convert to JSON for publishing
        task_list = []
        for task in task_sequence:
            task_dict = {
                'action': task.action,
                'parameters': task.parameters,
                'priority': task.priority,
                'requires_confirmation': task.requires_confirmation
            }
            task_list.append(task_dict)

        # Publish task sequence
        sequence_msg = String()
        sequence_msg.data = json.dumps(task_list, indent=2)
        self.task_sequence_pub.publish(sequence_msg)

        # Execute tasks that don't require confirmation, queue others
        for task in task_sequence:
            if task.requires_confirmation:
                self.request_confirmation(task)
            else:
                self.execute_task(task)

    def request_confirmation(self, task: TaskStep):
        """Request user confirmation for high-level tasks"""
        self.waiting_for_confirmation = True

        confirmation_msg = String()
        confirmation_msg.data = f"Execute task: {task.action} with params {task.parameters}? Confirm with /confirm yes/no"
        self.confirmation_request_pub.publish(confirmation_msg)

        self.get_logger().info(f'Waiting for confirmation: {confirmation_msg.data}')

        # Store callback function for when confirmation is received
        self.confirmation_callback_func = lambda confirmed: self.handle_confirmation(task, confirmed)

    def handle_confirmation(self, task: TaskStep, confirmed: bool):
        """Handle user confirmation response"""
        if confirmed:
            self.get_logger().info(f'Confirmation received, executing task: {task.action}')
            self.execute_task(task)
        else:
            self.get_logger().info(f'Confirmation denied for task: {task.action}')
            # Publish cancellation
            cancel_msg = String()
            cancel_msg.data = f"Task {task.action} cancelled by user"
            self.thought_pub.publish(cancel_msg)

        self.waiting_for_confirmation = False
        self.confirmation_callback_func = None

    def confirmation_callback(self, msg):
        """Handle confirmation responses from user"""
        if self.waiting_for_confirmation and self.confirmation_callback_func:
            confirmed = msg.data
            self.confirmation_callback_func(confirmed)

    def execute_task(self, task: TaskStep):
        """Execute a single task"""
        self.get_logger().info(f'Executing task: {task.action} with params: {task.parameters}')

        # Publish the action for other nodes to execute
        action_msg = String()
        action_msg.data = json.dumps({
            'action': task.action,
            'parameters': task.parameters
        })
        self.action_pub.publish(action_msg)


def main(args=None):
    rclpy.init(args=args)

    node = LLMThinkingSystem()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLM Thinking System')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()