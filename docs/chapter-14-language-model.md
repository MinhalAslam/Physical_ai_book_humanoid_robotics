# Chapter 14: Language Model Thinking - Cognitive Planning with Large Language Models

## The AI Brain for Robot Decision Making

In this chapter, we implement cognitive planning systems that use large language models (LLMs) to enable our robot to think, reason, and plan complex sequences of actions. This fulfills our project's functional requirement **FR-002**: System MUST understand natural language intent and convert to actionable tasks, and aligns with our constitution's principle that "Language is Control"—natural language is not just an interface, but a planning tool.

## Understanding Language Model Integration

Language models provide our robot with the ability to understand context, reason about complex commands, and plan multi-step tasks. This represents the cognitive layer of our "Vision-Language-Action" architecture, enabling sophisticated human-robot interaction.

### Key Tasks from Our Plan:
- T050: Implement language model thinking in src/cognitive_planning/chapter_llm.py

## Language Model Interface Node

Let's create a comprehensive language model interface node that handles cognitive planning:

```python
#!/usr/bin/env python3
"""
Language Model Interface Node for Physical AI System
Implements LLM-based cognitive planning and reasoning
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import openai
import json
import time
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class TaskPlan:
    """Represents a planned sequence of tasks"""
    id: str
    description: str
    steps: List[Dict]
    dependencies: List[str]
    priority: int
    estimated_time: float

class LanguageModelNode(Node):
    def __init__(self):
        super().__init__('language_model_node')

        # Create subscribers
        self.voice_command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )
        self.nlu_intent_sub = self.create_subscription(
            String,
            '/nlu/intent',
            self.nlu_intent_callback,
            10
        )
        self.task_status_sub = self.create_subscription(
            String,
            '/task_status',
            self.task_status_callback,
            10
        )

        # Create publishers
        self.task_plan_pub = self.create_publisher(
            String,
            '/task_plan',
            10
        )
        self.llm_response_pub = self.create_publisher(
            String,
            '/llm_response',
            10
        )
        self.cognitive_status_pub = self.create_publisher(
            String,
            '/cognitive_status',
            10
        )
        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )

        # LLM configuration
        self.llm_model = "gpt-3.5-turbo"  # Can be configured to use other models
        self.llm_temperature = 0.3
        self.max_tokens = 500
        self.api_key = None  # Should be set via environment variable

        # Task planning parameters
        self.task_queue = []
        self.active_tasks = {}
        self.task_history = []
        self.max_history_length = 50

        # System context
        self.system_context = self.create_system_context()

        # Thread pool for LLM calls
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Rate limiting
        self.last_llm_call = 0
        self.llm_call_interval = 1.0  # seconds

        self.get_logger().info('Language Model Node Started')

    def create_system_context(self):
        """Create the system context for the LLM"""
        return f"""
        You are an AI assistant for a physical robot. Your role is to understand human commands,
        plan appropriate actions, and coordinate with the robot's systems.

        Robot capabilities:
        - Navigation: can move to specific locations (kitchen, living room, bedroom, office)
        - Manipulation: can pick up, bring, and manipulate objects (cup, bottle, book, phone, laptop, box)
        - Perception: can see and identify objects in the environment
        - Communication: can speak and listen to humans

        Available actions:
        - NAVIGATE_TO: Move to a specific location
        - GRASP_OBJECT: Pick up an object
        - PLACE_OBJECT: Place an object down
        - FOLLOW_HUMAN: Follow a person
        - WAIT: Wait for further instructions
        - SPEAK: Say something to the human

        Constraints:
        - Always prioritize safety
        - If a command is unclear, ask for clarification
        - Break complex tasks into simple steps
        - Consider the robot's current state and environment
        - Respond in a helpful and polite manner
        """

    def voice_command_callback(self, msg):
        """Process voice commands through LLM"""
        command = msg.data.lower().strip()

        if not command:
            return

        self.get_logger().info(f'Processing voice command with LLM: {command}')

        # Check rate limiting
        current_time = time.time()
        if current_time - self.last_llm_call < self.llm_call_interval:
            time.sleep(self.llm_call_interval - (current_time - self.last_llm_call))

        # Process command asynchronously
        future = self.executor.submit(self.process_command_with_llm, command)
        future.add_done_callback(self.llm_response_callback)

        self.last_llm_call = time.time()

    def nlu_intent_callback(self, msg):
        """Process NLU intents"""
        intent = msg.data

        # Update cognitive status
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'processing_intent',
            'intent': intent,
            'timestamp': time.time()
        })
        self.cognitive_status_pub.publish(status_msg)

    def task_status_callback(self, msg):
        """Process task status updates"""
        try:
            status_data = json.loads(msg.data)
            task_id = status_data.get('task_id')
            status = status_data.get('status')

            if task_id and status:
                # Update active tasks
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]['status'] = status
                    if status in ['completed', 'failed']:
                        del self.active_tasks[task_id]

                # Add to history
                self.task_history.append(status_data)
                if len(self.task_history) > self.max_history_length:
                    self.task_history.pop(0)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task status message')

    def process_command_with_llm(self, command: str) -> Dict:
        """Process command using language model"""
        try:
            # Prepare the prompt
            prompt = f"""
            {self.system_context}

            Human command: "{command}"

            Please analyze this command and provide a structured response in JSON format:
            {{
                "intent": "classify the main intent (navigation, manipulation, conversation, etc.)",
                "entities": {{"key": "value", "location": "specific location", "object": "specific object"}},
                "action_plan": [
                    {{"action": "action_name", "parameters": {{"param": "value"}}, "description": "what this step does"}}
                ],
                "confidence": "confidence level (0.0 to 1.0)",
                "response": "what the robot should say in response"
            }}

            Be specific about locations and objects. If the command is unclear, set confidence low and suggest clarification.
            """

            # Make API call (in a real system, this would use the actual LLM)
            response = self.simulate_llm_call(prompt)

            return response

        except Exception as e:
            self.get_logger().error(f'LLM processing error: {str(e)}')
            return {
                'intent': 'error',
                'entities': {},
                'action_plan': [],
                'confidence': 0.0,
                'response': 'Sorry, I encountered an error processing your request.'
            }

    def simulate_llm_call(self, prompt: str) -> Dict:
        """Simulate LLM call (replace with actual API call in real system)"""
        # In a real system, this would call the LLM API
        # For simulation, we'll parse common commands
        command_lower = prompt.lower()

        # Common command patterns
        if 'kitchen' in command_lower:
            return {
                'intent': 'navigation',
                'entities': {'location': 'kitchen'},
                'action_plan': [
                    {'action': 'NAVIGATE_TO', 'parameters': {'location': 'kitchen'}, 'description': 'Navigate to kitchen'}
                ],
                'confidence': 0.9,
                'response': 'I\'ll go to the kitchen for you.'
            }
        elif 'living room' in command_lower:
            return {
                'intent': 'navigation',
                'entities': {'location': 'living room'},
                'action_plan': [
                    {'action': 'NAVIGATE_TO', 'parameters': {'location': 'living room'}, 'description': 'Navigate to living room'}
                ],
                'confidence': 0.9,
                'response': 'Heading to the living room now.'
            }
        elif 'pick up' in command_lower or 'get' in command_lower or 'bring' in command_lower:
            # Extract object
            words = command_lower.split()
            obj = None
            for i, word in enumerate(words):
                if word in ['cup', 'bottle', 'book', 'phone', 'laptop', 'box']:
                    obj = word
                    break

            if obj:
                return {
                    'intent': 'manipulation',
                    'entities': {'object': obj},
                    'action_plan': [
                        {'action': 'NAVIGATE_TO', 'parameters': {'location': 'object_location'}, 'description': 'Go to object'},
                        {'action': 'GRASP_OBJECT', 'parameters': {'object': obj}, 'description': f'Pick up the {obj}'},
                        {'action': 'NAVIGATE_TO', 'parameters': {'location': 'delivery_location'}, 'description': 'Go to delivery location'},
                        {'action': 'PLACE_OBJECT', 'parameters': {'object': obj}, 'description': f'Place the {obj}'}
                    ],
                    'confidence': 0.85,
                    'response': f'I\'ll get the {obj} for you.'
                }

        # Default response for unrecognized commands
        return {
            'intent': 'unknown',
            'entities': {},
            'action_plan': [],
            'confidence': 0.3,
            'response': 'I\'m not sure I understood. Could you please rephrase your request?'
        }

    def llm_response_callback(self, future):
        """Handle LLM response"""
        try:
            response_data = future.result()

            if response_data:
                # Publish LLM response
                response_msg = String()
                response_msg.data = json.dumps(response_data)
                self.llm_response_pub.publish(response_msg)

                # Process action plan if confidence is high enough
                if response_data.get('confidence', 0) > 0.5:
                    self.execute_action_plan(response_data.get('action_plan', []))

                # Publish cognitive status
                status_msg = String()
                status_msg.data = json.dumps({
                    'status': 'llm_processed',
                    'command': response_data.get('original_command', 'unknown'),
                    'intent': response_data.get('intent', 'unknown'),
                    'confidence': response_data.get('confidence', 0.0)
                })
                self.cognitive_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'LLM response processing error: {str(e)}')

    def execute_action_plan(self, action_plan: List[Dict]):
        """Execute the planned actions"""
        if not action_plan:
            return

        self.get_logger().info(f'Executing action plan with {len(action_plan)} steps')

        for i, action_step in enumerate(action_plan):
            action = action_step.get('action', '').upper()
            params = action_step.get('parameters', {})

            self.get_logger().info(f'Executing step {i+1}: {action} with params {params}')

            # Execute the action
            if action == 'NAVIGATE_TO':
                self.execute_navigation_action(params)
            elif action == 'GRASP_OBJECT':
                self.execute_manipulation_action(params, 'grasp')
            elif action == 'PLACE_OBJECT':
                self.execute_manipulation_action(params, 'place')
            elif action == 'FOLLOW_HUMAN':
                self.execute_follow_action(params)
            elif action == 'SPEAK':
                self.execute_speak_action(params)

            # Simulate time delay between actions
            time.sleep(0.5)

    def execute_navigation_action(self, params: Dict):
        """Execute navigation action"""
        location = params.get('location', 'unknown')

        # Map location to coordinates (in real system, this would come from map)
        location_coords = {
            'kitchen': {'x': 2.0, 'y': 0.0},
            'living room': {'x': -1.0, 'y': 1.0},
            'bedroom': {'x': 0.0, 'y': -2.0},
            'office': {'x': 1.5, 'y': -1.0},
            'object_location': {'x': 0.5, 'y': 0.5},  # Example object location
            'delivery_location': {'x': -0.5, 'y': -0.5}  # Example delivery location
        }

        if location in location_coords:
            coords = location_coords[location]

            # Create and publish navigation goal
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'
            goal_msg.pose.position.x = float(coords['x'])
            goal_msg.pose.position.y = float(coords['y'])
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.w = 1.0

            self.navigation_goal_pub.publish(goal_msg)
            self.get_logger().info(f'Navigating to {location} at ({coords["x"]}, {coords["y"]})')

    def execute_manipulation_action(self, params: Dict, action_type: str):
        """Execute manipulation action"""
        obj = params.get('object', 'unknown')
        self.get_logger().info(f'{action_type.capitalize()}ing object: {obj}')

        # In a real system, this would trigger the manipulation stack
        # For now, just log the action
        manipulation_msg = String()
        manipulation_msg.data = json.dumps({
            'action': f'{action_type}_object',
            'object': obj,
            'status': 'executing'
        })

    def execute_follow_action(self, params: Dict):
        """Execute follow action"""
        self.get_logger().info('Following human')

        # In a real system, this would use person tracking
        follow_msg = String()
        follow_msg.data = json.dumps({
            'action': 'follow_human',
            'status': 'tracking'
        })

    def execute_speak_action(self, params: Dict):
        """Execute speak action"""
        text = params.get('text', 'Hello')
        self.get_logger().info(f'Speaking: {text}')

        # Publish to TTS system
        tts_msg = String()
        tts_msg.data = text

    def plan_complex_task(self, command: str, entities: Dict) -> TaskPlan:
        """Plan a complex multi-step task"""
        task_id = f"task_{int(time.time())}"

        # Create a basic task plan based on command type
        if entities.get('location'):
            # Navigation task
            steps = [
                {'action': 'check_map', 'description': 'Check map for route to destination'},
                {'action': 'plan_path', 'description': 'Plan optimal path to location'},
                {'action': 'navigate', 'description': 'Navigate to specified location'},
                {'action': 'confirm_arrival', 'description': 'Confirm arrival at destination'}
            ]
        elif entities.get('object'):
            # Manipulation task
            steps = [
                {'action': 'locate_object', 'description': 'Locate the specified object'},
                {'action': 'navigate_to_object', 'description': 'Move to object location'},
                {'action': 'grasp_object', 'description': 'Pick up the object'},
                {'action': 'navigate_to_destination', 'description': 'Move to destination'},
                {'action': 'place_object', 'description': 'Place object at destination'}
            ]
        else:
            # Simple task
            steps = [
                {'action': 'process_command', 'description': 'Process the given command'},
                {'action': 'execute_action', 'description': 'Execute appropriate action'}
            ]

        task_plan = TaskPlan(
            id=task_id,
            description=command,
            steps=steps,
            dependencies=[],
            priority=1,
            estimated_time=len(steps) * 2.0  # 2 seconds per step estimate
        )

        # Add to task queue
        self.task_queue.append(task_plan)

        # Publish task plan
        plan_msg = String()
        plan_msg.data = json.dumps({
            'id': task_plan.id,
            'description': task_plan.description,
            'steps': task_plan.steps,
            'priority': task_plan.priority,
            'estimated_time': task_plan.estimated_time
        })
        self.task_plan_pub.publish(plan_msg)

        return task_plan

def main(args=None):
    rclpy.init(args=args)
    llm_node = LanguageModelNode()

    try:
        rclpy.spin(llm_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown executor
        llm_node.executor.shutdown(wait=True)
        llm_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Task Sequencing Node

Now let's create a task sequencing node that manages the execution of planned tasks:

```python
#!/usr/bin/env python3
"""
Task Sequencing Node for Physical AI System
Implements task planning and execution sequencing
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Header
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskStep:
    """Represents a single step in a task"""
    id: str
    action: str
    parameters: Dict
    description: str
    dependencies: List[str]
    timeout: float = 30.0  # seconds

@dataclass
class Task:
    """Represents a complete task"""
    id: str
    name: str
    steps: List[TaskStep]
    status: TaskStatus
    created_time: float
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    current_step_index: int = 0
    result: Optional[Dict] = None

class TaskSequencingNode(Node):
    def __init__(self):
        super().__init__('task_sequencing_node')

        # Create subscribers
        self.task_plan_sub = self.create_subscription(
            String,
            '/task_plan',
            self.task_plan_callback,
            10
        )
        self.task_status_sub = self.create_subscription(
            String,
            '/task_status',
            self.task_status_callback,
            10
        )
        self.cancel_sub = self.create_subscription(
            Bool,
            '/task_cancel',
            self.cancel_callback,
            10
        )

        # Create publishers
        self.task_status_pub = self.create_publisher(
            String,
            '/task_status',
            10
        )
        self.current_task_pub = self.create_publisher(
            String,
            '/current_task',
            10
        )
        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Task management
        self.tasks = {}  # task_id -> Task
        self.active_task_id = None
        self.task_queue = []  # Queue of pending tasks
        self.max_concurrent_tasks = 1  # Only one task at a time for safety

        # Navigation action client
        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        # Create timer for task execution
        self.task_timer = self.create_timer(0.1, self.execute_task_step)

        self.get_logger().info('Task Sequencing Node Started')

    def task_plan_callback(self, msg):
        """Process incoming task plans"""
        try:
            plan_data = json.loads(msg.data)

            task_id = plan_data.get('id', f'task_{int(time.time())}')
            description = plan_data.get('description', 'Unnamed task')
            steps_data = plan_data.get('steps', [])

            # Convert steps to TaskStep objects
            steps = []
            for i, step_data in enumerate(steps_data):
                step_id = f"{task_id}_step_{i}"
                step = TaskStep(
                    id=step_id,
                    action=step_data.get('action', 'unknown'),
                    parameters=step_data.get('parameters', {}),
                    description=step_data.get('description', f'Step {i}'),
                    dependencies=step_data.get('dependencies', []),
                    timeout=step_data.get('timeout', 30.0)
                )
                steps.append(step)

            # Create task
            task = Task(
                id=task_id,
                name=description,
                steps=steps,
                status=TaskStatus.PENDING,
                created_time=time.time()
            )

            # Add to queue
            self.task_queue.append(task)
            self.tasks[task_id] = task

            self.get_logger().info(f'Added task {task_id} to queue: {description}')

            # Publish task status
            self.publish_task_status(task)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task plan message')

    def task_status_callback(self, msg):
        """Process task status updates from other nodes"""
        try:
            status_data = json.loads(msg.data)
            task_id = status_data.get('task_id')
            status = status_data.get('status')

            if task_id and status and task_id in self.tasks:
                task = self.tasks[task_id]
                # Update task status based on external feedback
                if status == 'completed':
                    task.status = TaskStatus.COMPLETED
                    task.completion_time = time.time()
                elif status == 'failed':
                    task.status = TaskStatus.FAILED

                self.publish_task_status(task)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task status message')

    def cancel_callback(self, msg):
        """Cancel current task"""
        if msg.data and self.active_task_id:
            task = self.tasks.get(self.active_task_id)
            if task:
                task.status = TaskStatus.CANCELLED
                task.completion_time = time.time()

                # Stop robot
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)

                self.get_logger().info(f'Cancelled task {self.active_task_id}')
                self.publish_task_status(task)

                self.active_task_id = None

    def execute_task_step(self):
        """Execute one step of the current task"""
        # Check if we need a new task
        if not self.active_task_id and self.task_queue:
            # Get next task from queue
            next_task = self.task_queue.pop(0)
            next_task.status = TaskStatus.EXECUTING
            next_task.start_time = time.time()
            self.active_task_id = next_task.id

            self.get_logger().info(f'Starting task {next_task.id}: {next_task.name}')
            self.publish_task_status(next_task)

        # Execute current task step
        if self.active_task_id:
            task = self.tasks[self.active_task_id]

            if task.status == TaskStatus.EXECUTING:
                # Check if all steps are completed
                if task.current_step_index >= len(task.steps):
                    task.status = TaskStatus.COMPLETED
                    task.completion_time = time.time()
                    self.get_logger().info(f'Task {task.id} completed')
                    self.publish_task_status(task)
                    self.active_task_id = None
                    return

                # Execute current step
                current_step = task.steps[task.current_step_index]
                self.execute_step(current_step, task)

    def execute_step(self, step: TaskStep, task: Task):
        """Execute a single task step"""
        self.get_logger().info(f'Executing step: {step.description}')

        # Execute based on action type
        if step.action.upper() == 'NAVIGATE_TO':
            self.execute_navigation_step(step, task)
        elif step.action.upper() == 'GRASP_OBJECT':
            self.execute_manipulation_step(step, task, 'grasp')
        elif step.action.upper() == 'PLACE_OBJECT':
            self.execute_manipulation_step(step, task, 'place')
        elif step.action.upper() == 'MOVE_ROBOT':
            self.execute_movement_step(step, task)
        else:
            # Unknown action - mark step as completed
            self.mark_step_completed(task)

    def execute_navigation_step(self, step: TaskStep, task: Task):
        """Execute navigation step"""
        location = step.parameters.get('location', 'unknown')

        # Define location coordinates
        location_coords = {
            'kitchen': {'x': 2.0, 'y': 0.0},
            'living room': {'x': -1.0, 'y': 1.0},
            'bedroom': {'x': 0.0, 'y': -2.0},
            'office': {'x': 1.5, 'y': -1.0},
            'object_location': {'x': 0.5, 'y': 0.5},
            'delivery_location': {'x': -0.5, 'y': -0.5}
        }

        if location in location_coords:
            coords = location_coords[location]

            # Create navigation goal
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'
            goal_msg.pose.position.x = float(coords['x'])
            goal_msg.pose.position.y = float(coords['y'])
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.w = 1.0

            self.navigation_goal_pub.publish(goal_msg)
            self.get_logger().info(f'Navigating to {location} at ({coords["x"]}, {coords["y"]})')

            # For simulation, mark as completed after a delay
            time.sleep(2.0)  # Simulate navigation time
            self.mark_step_completed(task)

    def execute_manipulation_step(self, step: TaskStep, task: Task, action_type: str):
        """Execute manipulation step"""
        obj = step.parameters.get('object', 'unknown')
        self.get_logger().info(f'{action_type.capitalize()}ing object: {obj}')

        # For simulation, mark as completed after a delay
        time.sleep(1.0)  # Simulate manipulation time
        self.mark_step_completed(task)

    def execute_movement_step(self, step: TaskStep, task: Task):
        """Execute movement step"""
        direction = step.parameters.get('direction', 'stop')
        speed = step.parameters.get('speed', 0.3)

        cmd_vel = Twist()

        if direction == 'forward':
            cmd_vel.linear.x = speed
        elif direction == 'backward':
            cmd_vel.linear.x = -speed
        elif direction == 'left':
            cmd_vel.angular.z = speed
        elif direction == 'right':
            cmd_vel.angular.z = -speed
        else:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)

        # For simulation, mark as completed after a delay
        time.sleep(1.0)  # Simulate movement time
        self.mark_step_completed(task)

    def mark_step_completed(self, task: Task):
        """Mark current step as completed and move to next step"""
        task.current_step_index += 1

        if task.current_step_index >= len(task.steps):
            # All steps completed
            task.status = TaskStatus.COMPLETED
            task.completion_time = time.time()
            self.get_logger().info(f'Task {task.id} completed')
        else:
            # More steps to go
            self.get_logger().info(f'Task {task.id}: Step {task.current_step_index} completed')

        self.publish_task_status(task)

    def publish_task_status(self, task: Task):
        """Publish task status update"""
        status_msg = String()
        status_msg.data = json.dumps({
            'task_id': task.id,
            'name': task.name,
            'status': task.status.value,
            'current_step': task.current_step_index,
            'total_steps': len(task.steps),
            'created_time': task.created_time,
            'start_time': task.start_time,
            'completion_time': task.completion_time,
            'result': task.result
        })
        self.task_status_pub.publish(status_msg)

        # Also publish current task
        if task.status == TaskStatus.EXECUTING:
            current_task_msg = String()
            current_task_msg.data = json.dumps({
                'task_id': task.id,
                'name': task.name,
                'status': task.status.value,
                'step': task.current_step_index,
                'total_steps': len(task.steps)
            })
            self.current_task_pub.publish(current_task_msg)

def main(args=None):
    rclpy.init(args=args)
    task_node = TaskSequencingNode()

    try:
        rclpy.spin(task_node)
    except KeyboardInterrupt:
        pass
    finally:
        task_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Language Model Configuration

Create a configuration file for language model parameters:

```yaml
# language_model_config.yaml
language_model:
  # LLM service configuration
  service:
    provider: "openai"  # openai, anthropic, huggingface, or custom
    model: "gpt-3.5-turbo"
    temperature: 0.3
    max_tokens: 500
    top_p: 1.0
    frequency_penalty: 0.0
    presence_penalty: 0.0

  # API configuration
  api:
    base_url: "https://api.openai.com/v1"
    timeout: 30.0  # seconds
    max_retries: 3
    retry_delay: 1.0  # seconds

  # Cognitive planning parameters
  cognitive_planning:
    confidence_threshold: 0.7
    max_task_steps: 20
    task_timeout: 300.0  # seconds (5 minutes)
    context_window: 5  # number of previous interactions to remember
    plan_verification: true  # verify plans before execution

  # System context
  system_context:
    capabilities:
      - "navigation to known locations"
      - "object manipulation (grasping, placing)"
      - "human following"
      - "environment perception"
      - "spoken communication"

    constraints:
      - "always prioritize safety"
      - "ask for clarification if uncertain"
      - "break complex tasks into simple steps"
      - "consider current robot state and environment"
      - "maintain polite and helpful communication"

  # Task planning
  task_planning:
    step_templates:
      navigation:
        action: "NAVIGATE_TO"
        parameters:
          location: "{location}"
        description: "Navigate to {location}"

      manipulation:
        action: "MANIPULATE_OBJECT"
        parameters:
          object: "{object}"
          action_type: "{action_type}"
        description: "{action_type} the {object}"

      communication:
        action: "SPEAK"
        parameters:
          text: "{text}"
        description: "Say: {text}"

  # Performance parameters
  performance:
    max_concurrent_requests: 2
    request_queue_size: 10
    response_cache_size: 100
    cache_ttl: 300  # seconds

  # Safety parameters
  safety:
    enabled: true
    safety_check_frequency: 10.0  # seconds
    emergency_stop_keywords:
      - "stop"
      - "emergency"
      - "help"
      - "danger"
    content_filtering: true
```

## Language Model Launch File

Create a launch file for the language model system:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    config_file = LaunchConfiguration('config_file', default='language_model_config.yaml')

    # Language model node
    language_model_node = Node(
        package='physical_ai_cognitive_planning',
        executable='language_model_node',
        name='language_model_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_cognitive_planning'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    # Task sequencing node
    task_sequencing_node = Node(
        package='physical_ai_cognitive_planning',
        executable='task_sequencing_node',
        name='task_sequencing_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_cognitive_planning'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value='language_model_config.yaml',
            description='Configuration file for language model system'
        ),
        language_model_node,
        task_sequencing_node
    ])
```

## Quality Assurance for Language Model System

### Performance Metrics
- **Understanding Accuracy**: Percentage of commands correctly interpreted
- **Planning Quality**: Effectiveness of generated action plans
- **Response Time**: Time from command to plan generation
- **Task Success Rate**: Percentage of tasks completed successfully

### Safety Considerations
Following our constitution's "Safety is Intelligence" principle:

1. **Command Validation**: Verify commands are safe before planning
2. **Plan Verification**: Check action plans for safety before execution
3. **Emergency Override**: Support immediate task cancellation
4. **Content Filtering**: Filter inappropriate language/content

### Testing Scenarios
1. **Simple Commands**: Test basic command interpretation
2. **Complex Tasks**: Test multi-step task planning
3. **Ambiguous Commands**: Test handling of unclear requests
4. **Safety Scenarios**: Test safety constraint enforcement

## Looking Forward

With our language model cognitive planning system in place, the next chapter will focus on creating the complete Vision-Language-Action (VLA) system that integrates perception, language understanding, and action execution.

[Continue to Chapter 15: Vision-Language-Action System](./chapter-15-vla-system.md)