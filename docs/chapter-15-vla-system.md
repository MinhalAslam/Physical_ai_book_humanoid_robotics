# Chapter 15: Vision-Language-Action System - The Complete Cognitive Pipeline

## Integrating Perception, Language, and Action

In this chapter, we implement the complete Vision-Language-Action (VLA) system that represents the convergence of our robot's cognitive capabilities. This system integrates perception, language understanding, and action execution into a unified pipeline that enables our robot to process complex human commands in the context of its visual environment. This fulfills our constitution's principle of "Vision-Language-Action"—the convergence of mind and motion.

## Understanding the VLA Architecture

The Vision-Language-Action system creates a unified cognitive pipeline where:
- **Vision** provides environmental context and object recognition
- **Language** provides command understanding and reasoning
- **Action** provides physical execution of plans

This represents the culmination of our four learning pillars: ROS 2 (communication), Digital Twin (simulation), AI-Robot Brain (cognition), and VLA (integration).

### Key Tasks from Our Plan:
- T051: Create vision-language-action system in src/cognitive_planning/chapter_vla.py

## Vision-Language-Action Integration Node

Let's create a comprehensive VLA integration node that combines all cognitive capabilities:

```python
#!/usr/bin/env python3
"""
Vision-Language-Action Integration Node for Physical AI System
Implements complete VLA cognitive pipeline
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String, Bool, Float64
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
import numpy as np
from cv_bridge import CvBridge
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VLAContext:
    """Represents the complete VLA context"""
    image: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    objects: List[Dict] = None
    locations: List[Dict] = None
    command: str = ""
    intent: str = ""
    entities: Dict = None
    timestamp: float = 0.0

class VLANode(Node):
    def __init__(self):
        super().__init__('vla_node')

        # Create subscribers for all modalities
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )
        self.object_detection_sub = self.create_subscription(
            String,
            '/vision/object_detection',
            self.object_detection_callback,
            10
        )
        self.voice_command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )
        self.intent_sub = self.create_subscription(
            String,
            '/nlu/intent',
            self.intent_callback,
            10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.robot_pose_sub = self.create_subscription(
            PointStamped,
            '/robot_pose',
            self.robot_pose_callback,
            10
        )

        # Create publishers for VLA outputs
        self.vla_plan_pub = self.create_publisher(
            String,
            '/vla/action_plan',
            10
        )
        self.vla_context_pub = self.create_publisher(
            String,
            '/vla/context',
            10
        )
        self.vla_status_pub = self.create_publisher(
            String,
            '/vla/status',
            10
        )
        self.vla_visualization_pub = self.create_publisher(
            MarkerArray,
            '/vla/visualization',
            10
        )
        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )

        # Initialize components
        self.bridge = CvBridge()
        self.context = VLAContext()
        self.context.entities = {}

        # VLA parameters
        self.vla_enabled = True
        self.context_window = 5  # Keep last 5 contexts
        self.context_history = []
        self.processing_frequency = 2.0  # Hz
        self.min_command_confidence = 0.7
        self.min_object_confidence = 0.8

        # Object tracking
        self.tracked_objects = {}

        # Create processing timer
        self.process_timer = self.create_timer(1.0/self.processing_frequency, self.process_vla_cycle)

        self.get_logger().info('VLA Node Started')

    def rgb_callback(self, msg):
        """Process RGB image for visual context"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.context.image = cv_image
            self.context.timestamp = time.time()
        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {str(e)}')

    def depth_callback(self, msg):
        """Process depth image for spatial context"""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.context.depth = depth_image
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def object_detection_callback(self, msg):
        """Process object detection results"""
        try:
            detection_data = json.loads(msg.data)
            self.context.objects = detection_data.get('detections', [])

            # Update tracked objects
            for obj in self.context.objects:
                if obj.get('confidence', 0) >= self.min_object_confidence:
                    obj_id = f"{obj['class_name']}_{obj['center'][0]}_{obj['center'][1]}"
                    self.tracked_objects[obj_id] = obj

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in object detection message')

    def voice_command_callback(self, msg):
        """Process voice command for linguistic context"""
        self.context.command = msg.data.lower().strip()
        self.get_logger().info(f'VLA received command: {self.context.command}')

    def intent_callback(self, msg):
        """Process NLU intent"""
        self.context.intent = msg.data

    def map_callback(self, msg):
        """Process occupancy grid for spatial context"""
        # Store map information for navigation planning
        self.context.locations = self.extract_locations_from_map(msg)

    def robot_pose_callback(self, msg):
        """Process robot pose for current state context"""
        self.context.entities['robot_position'] = {
            'x': msg.point.x,
            'y': msg.point.y,
            'z': msg.point.z
        }

    def extract_locations_from_map(self, map_msg):
        """Extract known locations from occupancy grid"""
        # In a real system, this would identify rooms, corridors, etc.
        # For simulation, return some predefined locations
        return [
            {'name': 'kitchen', 'x': 2.0, 'y': 0.0},
            {'name': 'living_room', 'x': -1.0, 'y': 1.0},
            {'name': 'bedroom', 'y': 0.0, 'y': -2.0},
            {'name': 'office', 'x': 1.5, 'y': -1.0}
        ]

    def process_vla_cycle(self):
        """Process one cycle of VLA integration"""
        if not self.vla_enabled:
            return

        # Check if we have sufficient context
        if not self.context.command or not self.context.image is not None:
            return

        # Process the complete VLA context
        action_plan = self.generate_vla_action_plan()

        if action_plan:
            # Publish action plan
            plan_msg = String()
            plan_msg.data = json.dumps(action_plan)
            self.vla_plan_pub.publish(plan_msg)

            # Publish current context
            context_msg = String()
            context_msg.data = json.dumps({
                'command': self.context.command,
                'intent': self.context.intent,
                'objects': self.context.objects,
                'entities': self.context.entities,
                'timestamp': self.context.timestamp
            })
            self.vla_context_pub.publish(context_msg)

            # Execute the plan
            self.execute_vla_action_plan(action_plan)

            # Update context history
            self.context_history.append(self.context)
            if len(self.context_history) > self.context_window:
                self.context_history.pop(0)

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'vla_processed',
                'command': self.context.command,
                'plan_steps': len(action_plan.get('steps', [])),
                'timestamp': time.time()
            })
            self.vla_status_pub.publish(status_msg)

            self.get_logger().info(f'VLA processed command: {self.context.command}')

    def generate_vla_action_plan(self) -> Dict:
        """Generate action plan based on complete VLA context"""
        command = self.context.command
        intent = self.context.intent
        objects = self.context.objects or []
        entities = self.context.entities or {}

        # Analyze command in visual context
        plan = {
            'command': command,
            'intent': intent,
            'steps': [],
            'confidence': 0.0,
            'context_analysis': {}
        }

        # Analyze objects in the scene
        relevant_objects = self.find_relevant_objects(command, objects)
        plan['context_analysis']['relevant_objects'] = relevant_objects

        # Analyze spatial context
        relevant_locations = self.find_relevant_locations(command, entities)
        plan['context_analysis']['relevant_locations'] = relevant_locations

        # Generate action steps based on command and context
        if intent == 'navigation' and relevant_locations:
            plan['steps'] = self.generate_navigation_plan(relevant_locations[0])
            plan['confidence'] = 0.9
        elif intent == 'manipulation' and relevant_objects:
            plan['steps'] = self.generate_manipulation_plan(relevant_objects[0])
            plan['confidence'] = 0.85
        elif 'follow' in command or 'come' in command:
            plan['steps'] = self.generate_follow_plan()
            plan['confidence'] = 0.8
        else:
            # Default response for unrecognized commands
            plan['steps'] = self.generate_default_plan()
            plan['confidence'] = 0.3

        return plan

    def find_relevant_objects(self, command: str, objects: List[Dict]) -> List[Dict]:
        """Find objects relevant to the command"""
        relevant = []
        command_lower = command.lower()

        for obj in objects:
            if obj.get('confidence', 0) >= self.min_object_confidence:
                obj_name = obj.get('class_name', '').lower()

                # Check if object is mentioned in command
                if obj_name in command_lower:
                    relevant.append(obj)
                # Check for semantic similarity
                elif self.is_semantically_related(obj_name, command_lower):
                    relevant.append(obj)

        return relevant

    def is_semantically_related(self, obj_name: str, command: str) -> bool:
        """Check if object is semantically related to command"""
        # Define semantic relationships
        semantic_map = {
            'cup': ['drink', 'water', 'coffee', 'tea'],
            'bottle': ['drink', 'water', 'liquid'],
            'book': ['read', 'study', 'learn'],
            'phone': ['call', 'message', 'contact'],
            'laptop': ['work', 'computer', 'type'],
            'chair': ['sit', 'rest'],
            'table': ['place', 'put', 'on'],
        }

        for obj, keywords in semantic_map.items():
            if obj_name == obj and any(keyword in command for keyword in keywords):
                return True

        return False

    def find_relevant_locations(self, command: str, entities: Dict) -> List[Dict]:
        """Find locations relevant to the command"""
        relevant = []
        command_lower = command.lower()

        # Check for location mentions in command
        location_keywords = {
            'kitchen': ['kitchen', 'cooking', 'food', 'eat'],
            'living_room': ['living room', 'sit', 'relax', 'tv', 'couch'],
            'bedroom': ['bedroom', 'sleep', 'bed', 'rest'],
            'office': ['office', 'work', 'desk', 'computer']
        }

        for location, keywords in location_keywords.items():
            if any(keyword in command_lower for keyword in keywords):
                # Find the location in entities
                for loc in entities.get('locations', []):
                    if loc.get('name') == location:
                        relevant.append(loc)
                        break

        return relevant

    def generate_navigation_plan(self, location: Dict) -> List[Dict]:
        """Generate navigation action plan"""
        return [
            {
                'action': 'NAVIGATE_TO',
                'parameters': {
                    'x': location['x'],
                    'y': location['y'],
                    'location_name': location['name']
                },
                'description': f'Navigate to {location["name"]}'
            }
        ]

    def generate_manipulation_plan(self, obj: Dict) -> List[Dict]:
        """Generate manipulation action plan"""
        return [
            {
                'action': 'NAVIGATE_TO',
                'parameters': {
                    'x': obj['center'][0] * 0.01,  # Convert pixel to meters (simplified)
                    'y': obj['center'][1] * 0.01,
                    'object_id': obj.get('id', 'unknown')
                },
                'description': f'Navigate to {obj["class_name"]}'
            },
            {
                'action': 'GRASP_OBJECT',
                'parameters': {
                    'object_class': obj['class_name'],
                    'confidence': obj['confidence']
                },
                'description': f'Grasp the {obj["class_name"]}'
            }
        ]

    def generate_follow_plan(self) -> List[Dict]:
        """Generate follow action plan"""
        return [
            {
                'action': 'ACTIVATE_FOLLOW_MODE',
                'parameters': {},
                'description': 'Activate person following mode'
            }
        ]

    def generate_default_plan(self) -> List[Dict]:
        """Generate default action plan for unrecognized commands"""
        return [
            {
                'action': 'REQUEST_CLARIFICATION',
                'parameters': {},
                'description': 'Request clarification from user'
            }
        ]

    def execute_vla_action_plan(self, plan: Dict):
        """Execute the generated action plan"""
        steps = plan.get('steps', [])
        confidence = plan.get('confidence', 0.0)

        # Only execute if confidence is high enough
        if confidence < self.min_command_confidence:
            self.get_logger().info(f'VLA plan confidence too low ({confidence}), not executing')
            return

        for step in steps:
            action = step.get('action', '').upper()
            params = step.get('parameters', {})

            self.get_logger().info(f'Executing VLA step: {action}')

            if action == 'NAVIGATE_TO':
                self.execute_navigation_step(params)
            elif action == 'GRASP_OBJECT':
                self.execute_grasp_step(params)
            elif action == 'ACTIVATE_FOLLOW_MODE':
                self.execute_follow_step(params)
            elif action == 'REQUEST_CLARIFICATION':
                self.execute_clarification_step(params)

            # Small delay between steps
            time.sleep(0.5)

    def execute_navigation_step(self, params: Dict):
        """Execute navigation step"""
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        location_name = params.get('location_name', 'unknown')

        # Create navigation goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = float(x)
        goal_msg.pose.position.y = float(y)
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.navigation_goal_pub.publish(goal_msg)
        self.get_logger().info(f'VLA navigating to {location_name} at ({x}, {y})')

    def execute_grasp_step(self, params: Dict):
        """Execute grasp step"""
        obj_class = params.get('object_class', 'unknown')
        confidence = params.get('confidence', 0.0)

        self.get_logger().info(f'VLA attempting to grasp {obj_class} (confidence: {confidence})')

        # In a real system, this would trigger the manipulation stack
        # For simulation, just log the action
        grasp_msg = String()
        grasp_msg.data = json.dumps({
            'action': 'grasp_attempt',
            'object_class': obj_class,
            'confidence': confidence
        })

    def execute_follow_step(self, params: Dict):
        """Execute follow step"""
        self.get_logger().info('VLA activating follow mode')

        # In a real system, this would activate person following
        follow_msg = String()
        follow_msg.data = json.dumps({
            'action': 'follow_activated',
            'status': 'tracking'
        })

    def execute_clarification_step(self, params: Dict):
        """Execute clarification step"""
        self.get_logger().info('VLA requesting clarification')

        # Request clarification through TTS
        clarification_msg = String()
        clarification_msg.data = "I'm not sure I understood. Could you please repeat or rephrase your request?"
        # This would go to TTS system

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLANode()

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced VLA Reasoning Node

Let's create an advanced node that adds reasoning capabilities to the VLA system:

```python
#!/usr/bin/env python3
"""
Advanced VLA Reasoning Node for Physical AI System
Implements advanced reasoning and decision-making for VLA system
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import networkx as nx

@dataclass
class ReasoningContext:
    """Represents the reasoning context for VLA decisions"""
    objects: List[Dict]
    locations: List[Dict]
    command: str
    intent: str
    robot_state: Dict
    environment_state: Dict
    goals: List[Dict]
    constraints: List[Dict]

class AdvancedVLAReasoningNode(Node):
    def __init__(self):
        super().__init__('advanced_vla_reasoning_node')

        # Create subscribers
        self.vla_context_sub = self.create_subscription(
            String,
            '/vla/context',
            self.vla_context_callback,
            10
        )
        self.robot_state_sub = self.create_subscription(
            String,
            '/robot/state',
            self.robot_state_callback,
            10
        )
        self.environment_state_sub = self.create_subscription(
            String,
            '/environment/state',
            self.environment_state_callback,
            10
        )

        # Create publishers
        self.reasoned_plan_pub = self.create_publisher(
            String,
            '/vla/reasoned_plan',
            10
        )
        self.reasoning_trace_pub = self.create_publisher(
            String,
            '/vla/reasoning_trace',
            10
        )
        self.reasoning_status_pub = self.create_publisher(
            String,
            '/vla/reasoning_status',
            10
        )
        self.reasoning_visualization_pub = self.create_publisher(
            MarkerArray,
            '/vla/reasoning_visualization',
            10
        )

        # Reasoning context
        self.context = ReasoningContext(
            objects=[],
            locations=[],
            command="",
            intent="",
            robot_state={},
            environment_state={},
            goals=[],
            constraints=[]
        )

        # Reasoning parameters
        self.reasoning_enabled = True
        self.reasoning_frequency = 1.0  # Hz
        self.safety_constraints = [
            "avoid collisions",
            "maintain safe distances",
            "respect human space",
            "operate within physical limits"
        ]
        self.reasoning_depth = 3  # Number of steps to look ahead

        # Create reasoning timer
        self.reason_timer = self.create_timer(1.0/self.reasoning_frequency, self.perform_reasoning)

        self.get_logger().info('Advanced VLA Reasoning Node Started')

    def vla_context_callback(self, msg):
        """Process VLA context for reasoning"""
        try:
            context_data = json.loads(msg.data)
            self.context.command = context_data.get('command', '')
            self.context.intent = context_data.get('intent', '')
            self.context.objects = context_data.get('objects', [])
            self.context.goals = self.extract_goals_from_command(self.context.command)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in VLA context message')

    def robot_state_callback(self, msg):
        """Process robot state for reasoning"""
        try:
            self.context.robot_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in robot state message')

    def environment_state_callback(self, msg):
        """Process environment state for reasoning"""
        try:
            self.context.environment_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in environment state message')

    def extract_goals_from_command(self, command: str) -> List[Dict]:
        """Extract goals from natural language command"""
        goals = []

        # Simple goal extraction based on keywords
        command_lower = command.lower()

        if 'go to' in command_lower or 'navigate to' in command_lower:
            # Extract location goal
            for location in ['kitchen', 'living room', 'bedroom', 'office']:
                if location in command_lower:
                    goals.append({
                        'type': 'navigation',
                        'target': location,
                        'description': f'Navigate to {location}'
                    })
                    break

        if 'pick up' in command_lower or 'get' in command_lower or 'bring' in command_lower:
            # Extract manipulation goal
            for obj in ['cup', 'bottle', 'book', 'phone', 'laptop', 'box']:
                if obj in command_lower:
                    goals.append({
                        'type': 'manipulation',
                        'target': obj,
                        'description': f'Pick up {obj}'
                    })
                    break

        if 'follow' in command_lower:
            goals.append({
                'type': 'following',
                'target': 'human',
                'description': 'Follow the human'
            })

        return goals

    def perform_reasoning(self):
        """Perform advanced reasoning on the current context"""
        if not self.reasoning_enabled or not self.context.command:
            return

        self.get_logger().info(f'Performing reasoning for command: {self.context.command}')

        # Generate reasoning trace
        reasoning_trace = self.generate_reasoning_trace()

        # Create reasoned plan
        reasoned_plan = self.generate_reasoned_plan(reasoning_trace)

        if reasoned_plan:
            # Publish reasoned plan
            plan_msg = String()
            plan_msg.data = json.dumps(reasoned_plan)
            self.reasoned_plan_pub.publish(plan_msg)

            # Publish reasoning trace
            trace_msg = String()
            trace_msg.data = json.dumps(reasoning_trace)
            self.reasoning_trace_pub.publish(trace_msg)

            # Publish reasoning status
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'reasoning_completed',
                'command': self.context.command,
                'plan_steps': len(reasoned_plan.get('steps', [])),
                'reasoning_time': reasoning_trace.get('total_time', 0.0)
            })
            self.reasoning_status_pub.publish(status_msg)

            # Publish visualization
            self.publish_reasoning_visualization(reasoned_plan)

            self.get_logger().info(f'Reasoning completed with {len(reasoned_plan.get("steps", []))} steps')

    def generate_reasoning_trace(self) -> Dict:
        """Generate a trace of the reasoning process"""
        start_time = time.time()

        # Analyze the command
        command_analysis = self.analyze_command(self.context.command)

        # Assess current state
        state_assessment = self.assess_current_state()

        # Evaluate feasibility
        feasibility_analysis = self.evaluate_feasibility()

        # Generate reasoning steps
        reasoning_steps = [
            {
                'step': 'command_analysis',
                'description': f'Analyzed command: {self.context.command}',
                'result': command_analysis
            },
            {
                'step': 'state_assessment',
                'description': 'Assessed current robot and environment state',
                'result': state_assessment
            },
            {
                'step': 'feasibility_evaluation',
                'description': 'Evaluated task feasibility',
                'result': feasibility_analysis
            }
        ]

        total_time = time.time() - start_time

        return {
            'command': self.context.command,
            'intent': self.context.intent,
            'reasoning_steps': reasoning_steps,
            'total_time': total_time,
            'confidence': 0.9
        }

    def analyze_command(self, command: str) -> Dict:
        """Analyze the natural language command"""
        analysis = {
            'command': command,
            'intent': self.context.intent,
            'entities': {},
            'action_type': 'unknown',
            'complexity': 1
        }

        # Extract entities and determine action type
        command_lower = command.lower()

        # Identify objects
        objects = []
        for obj in ['cup', 'bottle', 'book', 'phone', 'laptop', 'box']:
            if obj in command_lower:
                objects.append(obj)
        analysis['entities']['objects'] = objects

        # Identify locations
        locations = []
        for loc in ['kitchen', 'living room', 'bedroom', 'office']:
            if loc in command_lower:
                locations.append(loc)
        analysis['entities']['locations'] = locations

        # Determine action type
        if any(word in command_lower for word in ['go', 'navigate', 'move to']):
            analysis['action_type'] = 'navigation'
        elif any(word in command_lower for word in ['pick', 'get', 'bring', 'grasp']):
            analysis['action_type'] = 'manipulation'
        elif any(word in command_lower for word in ['follow', 'come']):
            analysis['action_type'] = 'following'

        # Estimate complexity
        analysis['complexity'] = len(objects) + len(locations)

        return analysis

    def assess_current_state(self) -> Dict:
        """Assess the current state of the robot and environment"""
        state_assessment = {
            'robot_capabilities': self.context.robot_state.get('capabilities', []),
            'environment_objects': self.context.objects,
            'robot_position': self.context.robot_state.get('position', {}),
            'obstacles': self.context.environment_state.get('obstacles', []),
            'safety_status': 'safe'
        }

        # Check for safety constraints
        obstacles = state_assessment['obstacles']
        robot_pos = state_assessment['robot_position']

        if obstacles and robot_pos:
            for obstacle in obstacles:
                dist = self.calculate_distance(robot_pos, obstacle.get('position', {}))
                if dist < 0.5:  # 50cm safety margin
                    state_assessment['safety_status'] = 'unsafe_obstacle_too_close'

        return state_assessment

    def evaluate_feasibility(self) -> Dict:
        """Evaluate if the task is feasible given current constraints"""
        feasibility = {
            'is_feasible': True,
            'constraints': [],
            'reasons': []
        }

        # Check robot capabilities
        required_capabilities = []
        command_lower = self.context.command.lower()

        if 'navigation' in self.context.intent or any(word in command_lower for word in ['go', 'move', 'navigate']):
            required_capabilities.append('navigation')
        if 'manipulation' in self.context.intent or any(word in command_lower for word in ['pick', 'grasp', 'manipulate']):
            required_capabilities.append('manipulation')

        robot_capabilities = self.context.robot_state.get('capabilities', [])
        for cap in required_capabilities:
            if cap not in robot_capabilities:
                feasibility['is_feasible'] = False
                feasibility['constraints'].append(f'Missing capability: {cap}')
                feasibility['reasons'].append(f'Robot lacks {cap} capability')

        # Check safety constraints
        for constraint in self.safety_constraints:
            if not self.check_safety_constraint(constraint):
                feasibility['is_feasible'] = False
                feasibility['constraints'].append(constraint)
                feasibility['reasons'].append(f'Safety constraint violated: {constraint}')

        return feasibility

    def check_safety_constraint(self, constraint: str) -> bool:
        """Check if a safety constraint is satisfied"""
        # This is a simplified check - in a real system, this would be more complex
        if 'avoid collisions' in constraint:
            obstacles = self.context.environment_state.get('obstacles', [])
            return len(obstacles) == 0 or all(obs.get('distance', float('inf')) > 0.5 for obs in obstacles)

        if 'maintain safe distances' in constraint:
            return True  # Simplified check

        return True

    def generate_reasoned_plan(self, reasoning_trace: Dict) -> Dict:
        """Generate a reasoned action plan based on the reasoning trace"""
        plan = {
            'command': self.context.command,
            'intent': self.context.intent,
            'steps': [],
            'reasoning_trace': reasoning_trace,
            'confidence': reasoning_trace.get('confidence', 0.0),
            'safety_verified': True
        }

        # Generate steps based on intent and context
        if self.context.intent == 'navigation':
            plan['steps'] = self.generate_navigation_reasoned_steps()
        elif self.context.intent == 'manipulation':
            plan['steps'] = self.generate_manipulation_reasoned_steps()
        elif self.context.intent == 'following':
            plan['steps'] = self.generate_following_reasoned_steps()
        else:
            plan['steps'] = self.generate_default_reasoned_steps()

        # Verify safety of the plan
        plan['safety_verified'] = self.verify_plan_safety(plan['steps'])

        return plan

    def generate_navigation_reasoned_steps(self) -> List[Dict]:
        """Generate reasoned navigation steps"""
        steps = []

        # Find target location
        target_location = None
        command_lower = self.context.command.lower()
        for loc in ['kitchen', 'living room', 'bedroom', 'office']:
            if loc in command_lower:
                target_location = loc
                break

        if target_location:
            # Get location coordinates (in real system, this would come from map)
            location_coords = {
                'kitchen': {'x': 2.0, 'y': 0.0},
                'living_room': {'x': -1.0, 'y': 1.0},
                'bedroom': {'x': 0.0, 'y': -2.0},
                'office': {'x': 1.5, 'y': -1.0}
            }

            if target_location in location_coords:
                coords = location_coords[target_location]

                steps = [
                    {
                        'action': 'CHECK_SAFETY',
                        'parameters': {'check_type': 'navigation_path'},
                        'description': 'Verify safe navigation path exists'
                    },
                    {
                        'action': 'PLAN_PATH',
                        'parameters': {'target_x': coords['x'], 'target_y': coords['y']},
                        'description': f'Plan path to {target_location}'
                    },
                    {
                        'action': 'NAVIGATE',
                        'parameters': {'target_x': coords['x'], 'target_y': coords['y']},
                        'description': f'Navigate to {target_location}'
                    },
                    {
                        'action': 'CONFIRM_ARRIVAL',
                        'parameters': {'location': target_location},
                        'description': f'Confirm arrival at {target_location}'
                    }
                ]

        return steps

    def generate_manipulation_reasoned_steps(self) -> List[Dict]:
        """Generate reasoned manipulation steps"""
        steps = []

        # Find target object
        target_object = None
        command_lower = self.context.command.lower()
        for obj in ['cup', 'bottle', 'book', 'phone', 'laptop', 'box']:
            if obj in command_lower:
                target_object = obj
                break

        if target_object:
            # Find the object in the environment
            target_obj_info = None
            for obj_info in self.context.objects:
                if obj_info.get('class_name', '').lower() == target_object:
                    target_obj_info = obj_info
                    break

            if target_obj_info:
                steps = [
                    {
                        'action': 'LOCATE_OBJECT',
                        'parameters': {'object_class': target_object},
                        'description': f'Locate the {target_object}'
                    },
                    {
                        'action': 'APPROACH_OBJECT',
                        'parameters': {
                            'object_position': target_obj_info.get('position', {}),
                            'object_center': target_obj_info.get('center', [0, 0])
                        },
                        'description': f'Approach the {target_object}'
                    },
                    {
                        'action': 'GRASP_OBJECT',
                        'parameters': {
                            'object_class': target_object,
                            'confidence': target_obj_info.get('confidence', 0.0)
                        },
                        'description': f'Grasp the {target_object}'
                    },
                    {
                        'action': 'VERIFY_GRASP',
                        'parameters': {'object_class': target_object},
                        'description': f'Verify successful grasp of {target_object}'
                    }
                ]

        return steps

    def generate_following_reasoned_steps(self) -> List[Dict]:
        """Generate reasoned following steps"""
        return [
            {
                'action': 'ACTIVATE_PERSON_DETECTION',
                'parameters': {},
                'description': 'Activate person detection system'
            },
            {
                'action': 'ESTABLISH_TRACKING',
                'parameters': {},
                'description': 'Establish person tracking'
            },
            {
                'action': 'MAINTAIN_FOLLOW_DISTANCE',
                'parameters': {'distance': 1.0},  # 1 meter
                'description': 'Maintain safe following distance'
            },
            {
                'action': 'ADAPT_TO_MOVEMENT',
                'parameters': {},
                'description': 'Adapt to person\'s movement patterns'
            }
        ]

    def generate_default_reasoned_steps(self) -> List[Dict]:
        """Generate default steps for unrecognized commands"""
        return [
            {
                'action': 'REQUEST_CLARIFICATION',
                'parameters': {'command': self.context.command},
                'description': 'Request clarification from user'
            }
        ]

    def verify_plan_safety(self, steps: List[Dict]) -> bool:
        """Verify that the plan is safe to execute"""
        # Check each step for safety
        for step in steps:
            action = step.get('action', '').upper()

            if action in ['NAVIGATE', 'APPROACH_OBJECT']:
                # Check navigation safety
                if not self.check_navigation_safety(step.get('parameters', {})):
                    return False

            elif action == 'GRASP_OBJECT':
                # Check manipulation safety
                if not self.check_manipulation_safety(step.get('parameters', {})):
                    return False

        return True

    def check_navigation_safety(self, params: Dict) -> bool:
        """Check if navigation is safe"""
        # In a real system, this would check the path for obstacles
        return True

    def check_manipulation_safety(self, params: Dict) -> bool:
        """Check if manipulation is safe"""
        # In a real system, this would check for collisions and proper object handling
        return True

    def calculate_distance(self, pos1: Dict, pos2: Dict) -> float:
        """Calculate distance between two positions"""
        if not pos1 or not pos2:
            return float('inf')

        dx = pos1.get('x', 0) - pos2.get('x', 0)
        dy = pos1.get('y', 0) - pos2.get('y', 0)
        dz = pos1.get('z', 0) - pos2.get('z', 0)

        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def publish_reasoning_visualization(self, plan: Dict):
        """Publish visualization of the reasoning process"""
        marker_array = MarkerArray()

        # Create markers for each step in the plan
        for i, step in enumerate(plan.get('steps', [])):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'vla_reasoning'
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            # Position based on step index
            marker.pose.position.x = float(i * 0.5)
            marker.pose.position.y = 0.0
            marker.pose.position.z = 1.0
            marker.pose.orientation.w = 1.0

            marker.scale.z = 0.2  # Text size
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker.text = f"{i+1}. {step.get('description', 'Unknown step')}"

            marker_array.markers.append(marker)

        self.reasoning_visualization_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    reasoning_node = AdvancedVLAReasoningNode()

    try:
        rclpy.spin(reasoning_node)
    except KeyboardInterrupt:
        pass
    finally:
        reasoning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## VLA System Configuration

Create a configuration file for the VLA system:

```yaml
# vla_system_config.yaml
vla_system:
  # Vision processing parameters
  vision:
    object_detection:
      confidence_threshold: 0.8
      max_objects: 20
      detection_frequency: 10.0  # Hz
      classes_of_interest:
        - "person"
        - "chair"
        - "table"
        - "cup"
        - "bottle"
        - "book"
        - "phone"
        - "laptop"
        - "box"
        - "door"
        - "window"

  # Language processing parameters
  language:
    nlu:
      confidence_threshold: 0.7
      intent_classes:
        - "navigation"
        - "manipulation"
        - "following"
        - "greeting"
        - "help"
        - "conversation"
      entity_extraction:
        locations: ["kitchen", "living room", "bedroom", "office"]
        objects: ["cup", "bottle", "book", "phone", "laptop", "box"]
        actions: ["pick up", "get", "bring", "go to", "navigate to"]

  # Action planning parameters
  action_planning:
    planning_frequency: 2.0  # Hz
    max_plan_steps: 10
    plan_verification: true
    safety_check_frequency: 5.0  # Hz
    timeout: 300.0  # seconds (5 minutes)

  # Reasoning parameters
  reasoning:
    depth: 3  # Look-ahead steps
    confidence_threshold: 0.7
    reasoning_frequency: 1.0  # Hz
    constraints:
      safety:
        - "avoid collisions"
        - "maintain safe distances"
        - "respect human space"
        - "operate within physical limits"
      logical:
        - "verify preconditions"
        - "check feasibility"
        - "validate assumptions"

  # Integration parameters
  integration:
    context_window: 5  # Number of previous interactions to consider
    fusion_method: "attention"  # attention, concatenation, or custom
    modality_weights:
      vision: 0.4
      language: 0.4
      action_history: 0.2

  # Performance parameters
  performance:
    max_processing_time: 1.0  # seconds per cycle
    target_frequency: 10.0  # Hz
    memory_limit: 4096  # MB
    gpu_enabled: true
    cpu_threads: 4

  # Safety parameters
  safety:
    enabled: true
    emergency_stop_keywords:
      - "stop"
      - "emergency"
      - "help"
      - "danger"
      - "unsafe"
    collision_threshold: 0.5  # meters
    velocity_limits:
      linear: 0.5  # m/s
      angular: 1.0  # rad/s
```

## VLA System Launch File

Create a launch file for the VLA system:

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
    config_file = LaunchConfiguration('config_file', default='vla_system_config.yaml')

    # VLA integration node
    vla_node = Node(
        package='physical_ai_cognitive_planning',
        executable='vla_node',
        name='vla_node',
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

    # Advanced VLA reasoning node
    advanced_vla_node = Node(
        package='physical_ai_cognitive_planning',
        executable='advanced_vla_reasoning_node',
        name='advanced_vla_reasoning_node',
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
            default_value='vla_system_config.yaml',
            description='Configuration file for VLA system'
        ),
        vla_node,
        advanced_vla_node
    ])
```

## Quality Assurance for VLA System

### Performance Metrics
- **Integration Quality**: How well vision, language, and action components work together
- **Reasoning Accuracy**: Correctness of logical inferences and decisions
- **Response Time**: Time from input to action execution
- **Task Success Rate**: Percentage of tasks completed successfully

### Safety Considerations
Following our constitution's "Safety is Intelligence" principle:

1. **Multi-Modal Safety**: Safety checks across all modalities
2. **Reasoning Verification**: Validate reasoning steps before action
3. **Emergency Protocols**: Immediate stop on safety-critical situations
4. **Constraint Satisfaction**: Ensure all physical and logical constraints are met

### Testing Scenarios
1. **Simple Integration**: Basic vision-language-action tasks
2. **Complex Reasoning**: Multi-step tasks requiring reasoning
3. **Ambiguous Commands**: Commands requiring clarification
4. **Safety Scenarios**: Situations requiring safety interventions

## Looking Forward

With our complete Vision-Language-Action system in place, the next act will focus on building the complete autonomous robot by integrating all components and implementing sim-to-real transfer.

[Continue to Chapter 16: System Integration](./chapter-16-system-integration.md)