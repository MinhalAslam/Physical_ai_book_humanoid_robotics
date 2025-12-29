# Chapter 9: Human-Robot Interaction Scene Design - Creating Meaningful Encounters

## Designing for Human-Robot Interaction

In this chapter, we focus on creating meaningful human-robot interaction (HRI) scenes that will allow our robot to practice social behaviors in simulation. This aligns with our constitution's principle that "Language is Control"—natural language is not just an interface, but a planning tool that enables humans to direct robot behavior.

## HRI Principles and Design

### Core HRI Concepts
- **Social Presence**: The robot should feel like a social entity, not just a machine
- **Intuitive Communication**: Natural language and gesture-based interaction
- **Predictable Behavior**: The robot should behave in ways humans can anticipate
- **Safety First**: All interactions must prioritize human safety and comfort

### Scene Design Elements
1. **Contextual Environments**: Settings that make sense for specific interactions
2. **Social Cues**: Visual and auditory indicators of robot state and attention
3. **Interaction Patterns**: Predefined ways humans and robots can engage
4. **Feedback Mechanisms**: Clear communication of robot understanding and actions

## Implementation: HRI Scene Manager

Based on our project plan, we need to implement an HRI scene that allows for human-robot interaction in simulation.

### Key Tasks from Our Plan:
- T045: Design human-robot interaction scene in src/simulation/hri_scene.py

### HRI Scene Manager Node

```python
#!/usr/bin/env python3
"""
Human-Robot Interaction Scene Manager for Physical AI System
Manages HRI scenarios and interaction patterns in simulation
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
import numpy as np
from enum import Enum
import json
import time

class InteractionState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"

class HRISceneManager(Node):
    def __init__(self):
        super().__init__('hri_scene_manager')

        # Create subscribers for interaction signals
        self.voice_command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )
        self.gesture_sub = self.create_subscription(
            String,
            '/gesture_recognition',
            self.gesture_callback,
            10
        )
        self.task_status_sub = self.create_subscription(
            String,
            '/task_status',
            self.task_status_callback,
            10
        )

        # Create publishers for HRI feedback
        self.interaction_state_pub = self.create_publisher(
            String,
            '/hri_interaction_state',
            10
        )
        self.robot_attention_pub = self.create_publisher(
            PointStamped,
            '/robot_attention_target',
            10
        )
        self.visual_feedback_pub = self.create_publisher(
            MarkerArray,
            '/hri_visual_feedback',
            10
        )
        self.audio_feedback_pub = self.create_publisher(
            String,
            '/audio_feedback',
            10
        )

        # TF listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Scene state
        self.interaction_state = InteractionState.IDLE
        self.current_task = None
        self.human_positions = {}  # Track human positions
        self.interaction_history = []  # Track interaction history

        # HRI parameters
        self.attention_radius = 2.0  # meters - how far robot pays attention
        self.interaction_timeout = 10.0  # seconds - timeout for interactions
        self.last_interaction_time = time.time()

        # Create timer for state management
        self.state_timer = self.create_timer(0.1, self.state_callback)

        self.get_logger().info('HRI Scene Manager Node Started')

    def voice_command_callback(self, msg):
        """Handle voice commands from humans"""
        command = msg.data.lower().strip()
        self.get_logger().info(f'Received voice command: {command}')

        # Update interaction state
        self.interaction_state = InteractionState.PROCESSING
        self.last_interaction_time = time.time()

        # Publish state update
        state_msg = String()
        state_msg.data = self.interaction_state.value
        self.interaction_state_pub.publish(state_msg)

        # Process the command
        self.process_voice_command(command)

    def gesture_callback(self, msg):
        """Handle gesture recognition from computer vision"""
        gesture = msg.data.lower().strip()
        self.get_logger().info(f'Received gesture: {gesture}')

        # Update interaction state if relevant
        if gesture in ['wave', 'point', 'beckon']:
            self.interaction_state = InteractionState.LISTENING
            self.last_interaction_time = time.time()

            state_msg = String()
            state_msg.data = self.interaction_state.value
            self.interaction_state_pub.publish(state_msg)

            # Process the gesture
            self.process_gesture(gesture)

    def task_status_callback(self, msg):
        """Handle task completion status"""
        try:
            status_data = json.loads(msg.data)
            task_id = status_data.get('task_id', '')
            status = status_data.get('status', '')

            self.get_logger().info(f'Task {task_id} status: {status}')

            if status == 'completed':
                self.interaction_state = InteractionState.COMPLETED
                self.current_task = None
            elif status == 'failed':
                self.interaction_state = InteractionState.ERROR
                self.current_task = None

            # Publish updated state
            state_msg = String()
            state_msg.data = self.interaction_state.value
            self.interaction_state_pub.publish(state_msg)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task status message')

    def process_voice_command(self, command):
        """Process and respond to voice commands"""
        # Simple command processing - in a real system, this would use NLP
        if 'move' in command or 'go' in command:
            self.execute_navigation_task(command)
        elif 'pick' in command or 'get' in command:
            self.execute_manipulation_task(command)
        elif 'stop' in command or 'wait' in command:
            self.stop_robot()
        elif 'hello' in command or 'hi' in command:
            self.greet_human()
        else:
            self.get_logger().info(f'Command not recognized: {command}')
            self.interaction_state = InteractionState.ERROR
            self.provide_audio_feedback('Sorry, I did not understand that command.')

    def process_gesture(self, gesture):
        """Process and respond to gestures"""
        if gesture == 'wave':
            self.greet_human()
        elif gesture == 'point':
            self.acknowledge_pointing()
        elif gesture == 'beckon':
            self.move_towards_human()

    def execute_navigation_task(self, command):
        """Execute navigation-related tasks"""
        # Parse navigation command
        target = self.extract_target_location(command)
        if target:
            self.get_logger().info(f'Navigating to {target}')
            self.current_task = f'navigate_to_{target}'

            # Publish navigation goal
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'

            # Set target coordinates based on location name
            if target == 'kitchen':
                goal_msg.pose.position.x = 2.0
                goal_msg.pose.position.y = 0.0
            elif target == 'living_room':
                goal_msg.pose.position.x = -1.0
                goal_msg.pose.position.y = 1.0
            elif target == 'bedroom':
                goal_msg.pose.position.x = 0.0
                goal_msg.pose.position.y = -2.0
            else:
                # If it's a relative position, use current human position
                human_pos = self.get_nearest_human_position()
                if human_pos is not None:
                    goal_msg.pose.position.x = human_pos.x + 1.0  # 1m in front of human
                    goal_msg.pose.position.y = human_pos.y

            # In a real system, this would be sent to navigation stack
            self.get_logger().info(f'Navigation goal sent: {goal_msg.pose.position.x}, {goal_msg.pose.position.y}')

    def execute_manipulation_task(self, command):
        """Execute manipulation-related tasks"""
        # Parse manipulation command
        object_name = self.extract_object_name(command)
        if object_name:
            self.get_logger().info(f'Attempting to manipulate {object_name}')
            self.current_task = f'manipulate_{object_name}'

            # In a real system, this would be sent to manipulation stack
            self.get_logger().info(f'Manipulation task for {object_name} initiated')

    def extract_target_location(self, command):
        """Extract target location from command"""
        locations = ['kitchen', 'living_room', 'bedroom', 'office', 'bathroom', 'here', 'there']
        for loc in locations:
            if loc in command:
                return loc
        return None

    def extract_object_name(self, command):
        """Extract object name from command"""
        objects = ['cup', 'book', 'phone', 'bottle', 'box', 'ball', 'toy']
        for obj in objects:
            if obj in command:
                return obj
        return None

    def get_nearest_human_position(self):
        """Get the position of the nearest human"""
        if not self.human_positions:
            return None

        # For now, return the first human position
        # In a real system, we'd calculate distances
        for pos in self.human_positions.values():
            return pos

        return None

    def stop_robot(self):
        """Stop current robot activity"""
        self.get_logger().info('Stopping robot')
        self.interaction_state = InteractionState.IDLE
        self.current_task = None

    def greet_human(self):
        """Greet the human with audio and visual feedback"""
        self.get_logger().info('Greeting human')
        self.provide_audio_feedback('Hello! How can I help you?')
        self.create_visual_feedback('greeting', [0, 1, 0])  # Green light for greeting

    def acknowledge_pointing(self):
        """Acknowledge human pointing gesture"""
        self.get_logger().info('Acknowledging pointing gesture')
        self.provide_audio_feedback('I see where you are pointing.')
        self.create_visual_feedback('acknowledgment', [1, 1, 0])  # Yellow for acknowledgment

    def move_towards_human(self):
        """Move towards the human who beckoned"""
        human_pos = self.get_nearest_human_position()
        if human_pos:
            self.get_logger().info('Moving towards human')
            self.provide_audio_feedback('Coming to you now.')
            self.interaction_state = InteractionState.EXECUTING
            # In a real system, this would trigger navigation to human position

    def provide_audio_feedback(self, text):
        """Provide audio feedback to human"""
        feedback_msg = String()
        feedback_msg.data = text
        self.audio_feedback_pub.publish(feedback_msg)

    def create_visual_feedback(self, feedback_type, color):
        """Create visual feedback using markers"""
        marker_array = MarkerArray()

        # Create attention marker
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'hri_feedback'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position relative to robot
        marker.pose.position.x = 0.5  # In front of robot
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.5  # At eye level

        # Set size and color
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        marker_array.markers.append(marker)

        self.visual_feedback_pub.publish(marker_array)

    def state_callback(self):
        """Manage interaction state and timeouts"""
        current_time = time.time()

        # Check for interaction timeout
        if (current_time - self.last_interaction_time) > self.interaction_timeout:
            if self.interaction_state != InteractionState.IDLE:
                self.get_logger().info('Interaction timeout - returning to idle state')
                self.interaction_state = InteractionState.IDLE
                self.current_task = None

                # Publish state update
                state_msg = String()
                state_msg.data = self.interaction_state.value
                self.interaction_state_pub.publish(state_msg)

    def update_human_positions(self, positions):
        """Update tracked human positions"""
        self.human_positions = positions

def main(args=None):
    rclpy.init(args=args)
    hri_manager = HRISceneManager()

    try:
        rclpy.spin(hri_manager)
    except KeyboardInterrupt:
        pass
    finally:
        hri_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## HRI Scene Configuration

Create a configuration file for different HRI scenarios:

```yaml
# hri_scenes.yaml
hri_scenarios:
  # Home assistance scenario
  home_assistance:
    description: "Robot assists with household tasks"
    locations:
      - name: "kitchen"
        coordinates: [2.0, 0.0, 0.0]
        tasks: ["fetch_item", "set_table", "clear_table"]
      - name: "living_room"
        coordinates: [-1.0, 1.0, 0.0]
        tasks: ["greet_visitor", "bring_drink", "play_music"]
      - name: "bedroom"
        coordinates: [0.0, -2.0, 0.0]
        tasks: ["good_morning", "prepare_coffee", "remind_schedule"]

    interaction_patterns:
      greeting:
        triggers: ["hello", "hi", "hey"]
        responses: ["Hello! How can I help you?", "Good to see you!", "Hi there!"]
        actions: ["turn_towards_human", "light_up_leds"]

      fetch_item:
        triggers: ["bring", "get", "fetch", "carry"]
        response_template: "I'll get the {item} for you."
        actions: ["navigate_to_kitchen", "detect_object", "grasp_object", "return_to_human"]

  # Educational scenario
  education:
    description: "Robot as educational assistant"
    locations:
      - name: "classroom"
        coordinates: [0.0, 0.0, 0.0]
        tasks: ["answer_question", "demonstrate_concept", "guide_activity"]

    interaction_patterns:
      question_answering:
        triggers: ["what", "how", "why", "when", "where", "explain"]
        responses: ["Let me explain...", "Here's how it works...", "That's a great question!"]
        actions: ["search_knowledge_base", "display_information", "demonstrate"]

  # Healthcare scenario
  healthcare:
    description: "Robot for healthcare assistance"
    locations:
      - name: "patient_room"
        coordinates: [1.0, 0.0, 0.0]
        tasks: ["check_vitals", "remind_medication", "provide_companionship"]

    interaction_patterns:
      medication_reminder:
        triggers: ["medication", "pills", "time_to_take"]
        responses: ["It's time for your medication.", "Don't forget your pills."]
        actions: ["navigate_to_medication", "offer_medication", "confirm_taken"]

# HRI parameters
hri_parameters:
  attention_radius: 2.0  # meters
  interaction_timeout: 10.0  # seconds
  minimum_confidence: 0.8  # for voice/gesture recognition
  feedback_delay: 0.5  # seconds between feedback actions

# Safety constraints
safety_constraints:
  minimum_distance_to_human: 0.5  # meters
  maximum_approach_speed: 0.3  # m/s
  emergency_stop_distance: 0.3  # meters
```

## Interactive HRI Scene Launch File

Create a launch file that starts the HRI scene with necessary components:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    scenario = LaunchConfiguration('scenario', default='home_assistance')

    # Include Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ])
    )

    # HRI scene manager
    hri_scene_manager = Node(
        package='physical_ai_simulation',
        executable='hri_scene_manager',
        name='hri_scene_manager',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'scenario': scenario}
        ],
        output='screen'
    )

    # Human detection simulator (for testing)
    human_detector = Node(
        package='physical_ai_simulation',
        executable='human_detector_sim',
        name='human_detector_sim',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Voice command simulator (for testing)
    voice_simulator = Node(
        package='physical_ai_simulation',
        executable='voice_command_sim',
        name='voice_command_sim',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'scenario',
            default_value='home_assistance',
            description='HRI scenario to load'
        ),
        gazebo_launch,
        hri_scene_manager,
        human_detector,
        voice_simulator
    ])
```

## HRI Testing and Validation

### Testing Scenarios
1. **Basic Interaction Test**: Verify voice command recognition and response
2. **Gesture Recognition Test**: Test gesture detection and response
3. **Navigation Test**: Validate robot movement to specified locations
4. **Safety Test**: Ensure robot maintains safe distances from humans
5. **Timeout Test**: Verify proper timeout behavior for unresponsive humans

### Performance Metrics
- **Response Time**: Time from command to robot action
- **Recognition Accuracy**: Percentage of correctly recognized commands/gestures
- **Task Completion Rate**: Percentage of tasks completed successfully
- **Human Satisfaction**: Subjective measure of interaction quality

### Safety Considerations
Following our constitution's "Safety is Intelligence" principle:

1. **Physical Safety**: Robot maintains safe distances and speeds
2. **Privacy Protection**: No unauthorized recording or data collection
3. **Consent Management**: Clear indicators of when robot is listening/observing
4. **Emergency Protocols**: Immediate stop on safety commands

## Looking Forward

With our human-robot interaction scenes established in simulation, the next act will focus on building the robot's AI brain with synthetic vision, navigation systems, and learning capabilities.

[Continue to Chapter 10: Synthetic Vision](./chapter-10-synthetic-vision.md)