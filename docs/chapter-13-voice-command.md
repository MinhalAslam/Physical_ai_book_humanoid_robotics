# Chapter 13: Voice Command Pipeline - Enabling Natural Language Control

## The Voice-to-Action Pipeline

In this chapter, we implement the voice command pipeline that enables our robot to receive and process human voice commands. This fulfills our project's functional requirement **FR-001**: System MUST process voice commands with ≥90% accuracy using speech recognition, and aligns with our constitution's principle that "Language is Control"—natural language is not an interface, but a planning tool.

## Understanding Voice Command Processing

The voice command pipeline transforms spoken human language into actionable robot commands through several stages: speech recognition, natural language understanding, and command execution. This system represents a critical component of our "Vision-Language-Action" learning pillar.

### Key Tasks from Our Plan:
- T049: Create voice command pipeline in src/speech/chapter_voice.py

## Speech Recognition Node

Let's create a comprehensive speech recognition node that handles audio input and converts it to text:

```python
#!/usr/bin/env python3
"""
Speech Recognition Node for Physical AI System
Implements ASR (Automatic Speech Recognition) functionality
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import Twist
import numpy as np
import pyaudio
import webrtcvad
import collections
import threading
import queue
import json
import time

class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('speech_recognition_node')

        # Create publishers
        self.transcript_pub = self.create_publisher(
            String,
            '/speech/transcript',
            10
        )
        self.command_pub = self.create_publisher(
            String,
            '/voice_command',
            10
        )
        self.speech_status_pub = self.create_publisher(
            String,
            '/speech/status',
            10
        )

        # Audio parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Hz
        self.chunk_size = 1024
        self.audio_threshold = 500  # For basic VAD

        # VAD (Voice Activity Detection) parameters
        self.vad = webrtcvad.Vad(2)  # Aggressiveness mode 2
        self.frame_duration = 30  # ms
        self.frame_size = int(self.rate * self.frame_duration / 1000)
        self.speech_buffer_size = 100  # frames

        # Speech recognition parameters
        self.listening = False
        self.listening_timeout = 5.0  # seconds
        self.silence_timeout = 1.5  # seconds
        self.min_speech_duration = 0.5  # seconds

        # Audio buffers
        self.audio_queue = queue.Queue()
        self.speech_buffer = collections.deque(maxlen=self.speech_buffer_size)
        self.recording = False
        self.last_voice_time = time.time()

        # Create timer for speech processing
        self.speech_timer = self.create_timer(0.01, self.process_audio)

        # Start audio input thread
        self.audio_thread = threading.Thread(target=self.audio_input_thread)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        self.get_logger().info('Speech Recognition Node Started')

    def audio_input_thread(self):
        """Audio input thread to capture audio data"""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        try:
            while rclpy.ok():
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queue.put(data)
        except Exception as e:
            self.get_logger().error(f'Audio input error: {str(e)}')
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def process_audio(self):
        """Process incoming audio data"""
        # Process all available audio chunks
        while not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get_nowait()

                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Simple energy-based voice activity detection
                energy = np.mean(np.abs(audio_array.astype(np.float32)))

                # Check if speech is detected
                is_speech = energy > self.audio_threshold

                # Update speech buffer
                self.speech_buffer.append((audio_data, is_speech, time.time()))

                # Update recording state
                if is_speech:
                    self.last_voice_time = time.time()
                    if not self.recording:
                        self.start_recording()
                elif self.recording and (time.time() - self.last_voice_time) > self.silence_timeout:
                    self.stop_recording()

            except queue.Empty:
                break

    def start_recording(self):
        """Start recording speech"""
        if not self.recording:
            self.recording = True
            self.get_logger().info('Started recording speech')

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'recording_started',
                'timestamp': time.time()
            })
            self.speech_status_pub.publish(status_msg)

    def stop_recording(self):
        """Stop recording and process speech"""
        if self.recording:
            self.recording = False
            self.get_logger().info('Stopped recording speech')

            # Extract speech segments
            speech_data = self.extract_speech_segment()

            if len(speech_data) > 0:
                # Simulate speech recognition (in real system, this would call Whisper or similar)
                transcript = self.simulate_speech_recognition(speech_data)

                if transcript and len(transcript.strip()) > 0:
                    # Publish transcript
                    transcript_msg = String()
                    transcript_msg.data = transcript
                    self.transcript_pub.publish(transcript_msg)

                    # Process command
                    self.process_command(transcript)

                    self.get_logger().info(f'Recognized: {transcript}')

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'recording_stopped',
                'timestamp': time.time()
            })
            self.speech_status_pub.publish(status_msg)

    def extract_speech_segment(self):
        """Extract speech segment from buffer"""
        speech_data = b''

        # Extract recent speech segments
        for audio_chunk, is_speech, timestamp in list(self.speech_buffer):
            if is_speech:
                speech_data += audio_chunk

        return speech_data

    def simulate_speech_recognition(self, audio_data):
        """Simulate speech recognition (replace with actual ASR in real system)"""
        # In a real system, this would call Whisper or another ASR model
        # For simulation, we'll recognize some common commands

        # Common voice commands for our robot
        common_commands = {
            b'hello': 'hello',
            b'move forward': 'move forward',
            b'go forward': 'move forward',
            b'move backward': 'move backward',
            b'go backward': 'move backward',
            b'turn left': 'turn left',
            b'turn right': 'turn right',
            b'stop': 'stop',
            b'help': 'help',
            b'come here': 'come here',
            b'follow me': 'follow me',
            b'go to kitchen': 'go to kitchen',
            b'go to living room': 'go to living room',
            b'pick up the cup': 'pick up the cup',
            b'bring me the bottle': 'bring me the bottle'
        }

        # Simple pattern matching for simulation
        audio_str = audio_data.lower() if isinstance(audio_data, bytes) else str(audio_data).lower()

        for pattern, command in common_commands.items():
            if isinstance(pattern, bytes):
                pattern = pattern.decode('utf-8')
            if pattern in audio_str:
                return command

        # If no known command found, return empty string
        return ""

    def process_command(self, transcript):
        """Process the recognized command"""
        # Publish the voice command
        command_msg = String()
        command_msg.data = transcript
        self.command_pub.publish(command_msg)

def main(args=None):
    rclpy.init(args=args)
    speech_node = SpeechRecognitionNode()

    try:
        rclpy.spin(speech_node)
    except KeyboardInterrupt:
        pass
    finally:
        speech_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Natural Language Understanding Node

Now let's create the Natural Language Understanding (NLU) node that processes the recognized text:

```python
#!/usr/bin/env python3
"""
Natural Language Understanding Node for Physical AI System
Implements NLU (Natural Language Understanding) functionality
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ParsedCommand:
    intent: str
    entities: Dict[str, str]
    confidence: float
    original_text: str

class NaturalLanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('nlu_node')

        # Create subscribers
        self.transcript_sub = self.create_subscription(
            String,
            '/speech/transcript',
            self.transcript_callback,
            10
        )

        # Create publishers
        self.intent_pub = self.create_publisher(
            String,
            '/nlu/intent',
            10
        )
        self.command_pub = self.create_publisher(
            String,
            '/parsed_command',
            10
        )
        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )
        self.motion_cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Intent patterns and entities
        self.intent_patterns = {
            'navigation': [
                r'move\s+(?P<direction>forward|backward|left|right)',
                r'go\s+(?P<direction>forward|backward|left|right)',
                r'go\s+to\s+(?P<location>\w+)',
                r'come\s+to\s+(?P<location>\w+)',
                r'go\s+(?P<location>\w+)',
                r'walk\s+(?P<direction>forward|backward|left|right)',
                r'turn\s+(?P<direction>left|right)',
                r'straight',
                r'stop',
                r'wait'
            ],
            'manipulation': [
                r'pick\s+up\s+(?P<object>[\w\s]+)',
                r'get\s+(?P<object>[\w\s]+)',
                r'bring\s+(?P<object>[\w\s]+)',
                r'fetch\s+(?P<object>[\w\s]+)',
                r'grasp\s+(?P<object>[\w\s]+)',
                r'hold\s+(?P<object>[\w\s]+)',
                r'put\s+(?P<object>[\w\s]+)'
            ],
            'greeting': [
                r'hello',
                r'hi',
                r'hey',
                r'good\s+(morning|afternoon|evening)',
                r'how\s+are\s+you',
                r'what\'?s\s+up'
            ],
            'help': [
                r'help',
                r'what\s+can\s+you\s+do',
                r'what\s+are\s+your\s+capabilities',
                r'what\s+commands\s+do\s+you\s+understand'
            ]
        }

        # Location mappings
        self.location_mappings = {
            'kitchen': {'x': 2.0, 'y': 0.0},
            'living room': {'x': -1.0, 'y': 1.0},
            'bedroom': {'x': 0.0, 'y': -2.0},
            'office': {'x': 1.5, 'y': -1.0},
            'here': {'x': 0.5, 'y': 0.5},  # Relative to current position
            'there': {'x': -0.5, 'y': -0.5}
        }

        # Object mappings
        self.object_mappings = {
            'cup': 'cup',
            'bottle': 'bottle',
            'book': 'book',
            'phone': 'phone',
            'laptop': 'laptop',
            'box': 'box'
        }

        self.get_logger().info('NLU Node Started')

    def transcript_callback(self, msg):
        """Process incoming transcript"""
        transcript = msg.data.lower().strip()

        if not transcript:
            return

        self.get_logger().info(f'Processing transcript: {transcript}')

        # Parse the command
        parsed_command = self.parse_command(transcript)

        if parsed_command:
            # Publish parsed command
            command_msg = String()
            command_msg.data = json.dumps({
                'intent': parsed_command.intent,
                'entities': parsed_command.entities,
                'confidence': parsed_command.confidence,
                'original_text': parsed_command.original_text
            })
            self.command_pub.publish(command_msg)

            # Execute the command based on intent
            self.execute_command(parsed_command)

            # Publish intent for other nodes
            intent_msg = String()
            intent_msg.data = parsed_command.intent
            self.intent_pub.publish(intent_msg)

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """Parse the command and extract intent and entities"""
        # Check each intent type
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entities = match.groupdict()

                    # Process entities
                    processed_entities = {}
                    for key, value in entities.items():
                        if value:
                            # Clean up the entity value
                            cleaned_value = value.strip().lower()

                            # Map location entities
                            if key == 'location':
                                if cleaned_value in self.location_mappings:
                                    processed_entities[key] = cleaned_value
                                else:
                                    # Try to find closest match
                                    closest = self.find_closest_location(cleaned_value)
                                    if closest:
                                        processed_entities[key] = closest

                            # Map object entities
                            elif key == 'object':
                                if cleaned_value in self.object_mappings:
                                    processed_entities[key] = cleaned_value
                                else:
                                    # Try to find closest match
                                    closest = self.find_closest_object(cleaned_value)
                                    if closest:
                                        processed_entities[key] = closest
                            else:
                                processed_entities[key] = cleaned_value

                    return ParsedCommand(
                        intent=intent,
                        entities=processed_entities,
                        confidence=0.9,  # High confidence for pattern matching
                        original_text=text
                    )

        # If no pattern matches, return unknown intent
        return ParsedCommand(
            intent='unknown',
            entities={},
            confidence=0.1,
            original_text=text
        )

    def find_closest_location(self, location: str) -> Optional[str]:
        """Find the closest matching location"""
        for loc in self.location_mappings:
            if location in loc or loc in location:
                return loc
        return None

    def find_closest_object(self, obj: str) -> Optional[str]:
        """Find the closest matching object"""
        for o in self.object_mappings:
            if obj in o or o in obj:
                return o
        return None

    def execute_command(self, parsed_command: ParsedCommand):
        """Execute the parsed command"""
        intent = parsed_command.intent
        entities = parsed_command.entities

        self.get_logger().info(f'Executing command: {intent} with entities {entities}')

        if intent == 'navigation':
            self.handle_navigation_command(entities)
        elif intent == 'manipulation':
            self.handle_manipulation_command(entities)
        elif intent == 'greeting':
            self.handle_greeting_command(entities)
        elif intent == 'help':
            self.handle_help_command(entities)
        else:
            self.get_logger().info(f'Unknown command intent: {intent}')

    def handle_navigation_command(self, entities: Dict[str, str]):
        """Handle navigation commands"""
        if 'location' in entities:
            location = entities['location']
            if location in self.location_mappings:
                location_data = self.location_mappings[location]

                # Create navigation goal
                goal_msg = PoseStamped()
                goal_msg.header.stamp = self.get_clock().now().to_msg()
                goal_msg.header.frame_id = 'map'
                goal_msg.pose.position.x = float(location_data['x'])
                goal_msg.pose.position.y = float(location_data['y'])
                goal_msg.pose.position.z = 0.0
                goal_msg.pose.orientation.w = 1.0

                self.navigation_goal_pub.publish(goal_msg)
                self.get_logger().info(f'Navigating to {location} at ({location_data["x"]}, {location_data["y"]})')

        elif 'direction' in entities:
            direction = entities['direction']

            # Create motion command based on direction
            cmd_vel = Twist()

            if direction == 'forward':
                cmd_vel.linear.x = 0.3
            elif direction == 'backward':
                cmd_vel.linear.x = -0.3
            elif direction == 'left':
                cmd_vel.angular.z = 0.5
            elif direction == 'right':
                cmd_vel.angular.z = -0.5
            elif direction == 'stop':
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
            elif direction == 'straight':
                cmd_vel.linear.x = 0.2
                cmd_vel.angular.z = 0.0

            self.motion_cmd_pub.publish(cmd_vel)
            self.get_logger().info(f'Moving {direction}')

    def handle_manipulation_command(self, entities: Dict[str, str]):
        """Handle manipulation commands"""
        if 'object' in entities:
            obj = entities['object']
            self.get_logger().info(f'Attempting to manipulate object: {obj}')

            # In a real system, this would trigger manipulation planning
            # For now, just log the intent
            manipulation_msg = String()
            manipulation_msg.data = json.dumps({
                'action': 'manipulate',
                'object': obj,
                'status': 'planning'
            })
            # This would go to manipulation system in real implementation

    def handle_greeting_command(self, entities: Dict[str, str]):
        """Handle greeting commands"""
        self.get_logger().info('Received greeting command')

        # In a real system, this might trigger a response
        greeting_response = String()
        greeting_response.data = "Hello! How can I help you today?"
        # This would go to TTS system in real implementation

    def handle_help_command(self, entities: Dict[str, str]):
        """Handle help commands"""
        self.get_logger().info('Received help command')

        # In a real system, this might trigger a help response
        help_response = String()
        help_response.data = "I can help with navigation, object manipulation, and general assistance. Try saying 'move forward' or 'go to kitchen'."
        # This would go to TTS system in real implementation

def main(args=None):
    rclpy.init(args=args)
    nlu_node = NaturalLanguageUnderstandingNode()

    try:
        rclpy.spin(nlu_node)
    except KeyboardInterrupt:
        pass
    finally:
        nlu_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Text-to-Speech Node

Let's also create a TTS node for verbal responses:

```python
#!/usr/bin/env python3
"""
Text-to-Speech Node for Physical AI System
Implements TTS (Text-to-Speech) functionality
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import pyttsx3
import threading
import queue

class TextToSpeechNode(Node):
    def __init__(self):
        super().__init__('tts_node')

        # Create subscribers
        self.response_sub = self.create_subscription(
            String,
            '/tts/response',
            self.response_callback,
            10
        )
        self.status_sub = self.create_subscription(
            String,
            '/system_status',
            self.status_callback,
            10
        )

        # Create publishers
        self.tts_status_pub = self.create_publisher(
            String,
            '/tts/status',
            10
        )

        # TTS engine
        self.tts_engine = pyttsx3.init()

        # Configure TTS properties
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

        # Get available voices
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)  # Use first available voice

        # TTS queue for thread-safe operation
        self.tts_queue = queue.Queue()
        self.tts_lock = threading.Lock()

        # Start TTS processing thread
        self.tts_thread = threading.Thread(target=self.tts_worker)
        self.tts_thread.daemon = True
        self.tts_thread.start()

        self.get_logger().info('TTS Node Started')

    def response_callback(self, msg):
        """Process text-to-speech requests"""
        text = msg.data

        if text and len(text.strip()) > 0:
            self.get_logger().info(f'Queuing TTS: {text}')

            # Add to TTS queue
            self.tts_queue.put(text)

    def status_callback(self, msg):
        """Process system status for TTS announcements"""
        try:
            status_data = json.loads(msg.data)

            # Convert status to verbal announcement
            announcement = self.convert_status_to_speech(status_data)

            if announcement:
                self.tts_queue.put(announcement)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in status message')

    def convert_status_to_speech(self, status_data):
        """Convert system status to speech announcement"""
        if 'task_status' in status_data:
            task_status = status_data['task_status']
            if task_status == 'completed':
                return "Task completed successfully."
            elif task_status == 'failed':
                return "Task failed. Please try again."
            elif task_status == 'started':
                return "Task started. Working on it now."

        elif 'navigation_status' in status_data:
            nav_status = status_data['navigation_status']
            if nav_status == 'reached_goal':
                return "I have reached the destination."
            elif nav_status == 'path_planning':
                return "Planning the path now."
            elif nav_status == 'obstacle_detected':
                return "Obstacle detected. Finding alternative route."

        elif 'voice_command' in status_data:
            cmd = status_data['voice_command']
            return f"Got it. I will {cmd}."

        return None

    def tts_worker(self):
        """Worker thread for TTS processing"""
        while rclpy.ok():
            try:
                # Wait for text to speak
                text = self.tts_queue.get(timeout=1.0)

                if text:
                    with self.tts_lock:
                        # Speak the text
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()

                    # Publish status
                    status_msg = String()
                    status_msg.data = json.dumps({
                        'status': 'speech_completed',
                        'text': text
                    })
                    self.tts_status_pub.publish(status_msg)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'TTS error: {str(e)}')

    def destroy_node(self):
        """Clean up TTS resources"""
        with self.tts_lock:
            self.tts_engine.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    tts_node = TextToSpeechNode()

    try:
        rclpy.spin(tts_node)
    except KeyboardInterrupt:
        pass
    finally:
        tts_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Voice Command Configuration

Create a configuration file for voice command parameters:

```yaml
# voice_command_config.yaml
voice_command:
  # Audio input parameters
  audio:
    sample_rate: 16000  # Hz
    channels: 1
    format: "int16"
    chunk_size: 1024
    threshold: 500  # For basic VAD

  # VAD (Voice Activity Detection) parameters
  vad:
    aggressiveness: 2  # 0-3, higher = more aggressive
    frame_duration: 30  # ms
    min_speech_duration: 0.5  # seconds
    silence_timeout: 1.5  # seconds
    listening_timeout: 5.0  # seconds

  # ASR (Automatic Speech Recognition) parameters
  asr:
    model: "whisper-tiny"  # Model size: tiny, base, small, medium, large
    language: "en"
    temperature: 0.0
    patience: 0.5
    compression_ratio_threshold: 2.4
    logprob_threshold: -1.0
    no_speech_threshold: 0.6

  # NLU (Natural Language Understanding) parameters
  nlu:
    confidence_threshold: 0.7
    intent_mappings:
      navigation:
        - "move forward"
        - "go forward"
        - "move backward"
        - "go backward"
        - "turn left"
        - "turn right"
        - "go to kitchen"
        - "go to living room"
        - "go to bedroom"
        - "stop"
        - "wait"
      manipulation:
        - "pick up cup"
        - "get bottle"
        - "bring book"
        - "fetch phone"
        - "grasp object"
      greeting:
        - "hello"
        - "hi"
        - "good morning"
        - "good afternoon"
        - "good evening"
      help:
        - "help"
        - "what can you do"
        - "how to use"

  # TTS (Text-to-Speech) parameters
  tts:
    rate: 150  # Words per minute
    volume: 0.9  # 0.0 to 1.0
    voice: "default"  # Use default system voice
    response_prefixes:
      confirmation: "Got it. "
      error: "Sorry, I couldn't understand that. "
      help: "I can help you with "

  # Performance parameters
  performance:
    max_processing_time: 2.0  # seconds per command
    queue_size: 10
    retry_attempts: 3
    timeout: 10.0  # seconds for command timeout
```

## Voice Command Launch File

Create a launch file for the voice command system:

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
    config_file = LaunchConfiguration('config_file', default='voice_command_config.yaml')

    # Speech recognition node
    speech_recognition_node = Node(
        package='physical_ai_speech',
        executable='speech_recognition_node',
        name='speech_recognition_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_speech'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    # Natural Language Understanding node
    nlu_node = Node(
        package='physical_ai_speech',
        executable='nlu_node',
        name='nlu_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_speech'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    # Text-to-Speech node
    tts_node = Node(
        package='physical_ai_speech',
        executable='tts_node',
        name='tts_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_speech'),
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
            default_value='voice_command_config.yaml',
            description='Configuration file for voice command system'
        ),
        speech_recognition_node,
        nlu_node,
        tts_node
    ])
```

## Quality Assurance for Voice Command System

### Performance Metrics
- **Recognition Accuracy**: Percentage of commands correctly recognized
- **Response Time**: Time from speech input to action execution
- **Understanding Rate**: Percentage of commands correctly parsed
- **Robustness**: Performance in various acoustic environments

### Safety Considerations
Following our constitution's "Safety is Intelligence" principle:

1. **Command Validation**: Verify commands are safe before execution
2. **Ambiguity Handling**: Request clarification for unclear commands
3. **Emergency Commands**: Support immediate stop/abort commands
4. **Privacy Protection**: Secure handling of audio data

### Testing Scenarios
1. **Basic Commands**: Test recognition of fundamental commands
2. **Noisy Environment**: Test performance with background noise
3. **Multiple Speakers**: Test ability to handle multiple voices
4. **Edge Cases**: Test handling of unknown or ambiguous commands

## Looking Forward

With our voice command pipeline established, the next chapter will focus on language model integration that will enable our robot to think and reason about commands before executing them.

[Continue to Chapter 14: Language Model Thinking](./chapter-14-language-model.md)