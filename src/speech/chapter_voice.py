#!/usr/bin/env python3
"""
Voice Command Pipeline

This node implements the complete voice command processing pipeline:
- Audio input capture
- Speech-to-text conversion
- Natural language understanding
- Command execution
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from audio_common_msgs.msg import AudioData
import speech_recognition as sr
import threading
import queue
import time


class VoiceCommandPipeline(Node):
    def __init__(self):
        super().__init__('voice_command_pipeline')

        # Publishers
        self.speech_text_pub = self.create_publisher(String, '/speech_text', 10)
        self.command_pub = self.create_publisher(String, '/parsed_command', 10)
        self.listening_status_pub = self.create_publisher(Bool, '/is_listening', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/voice_command', self.voice_command_callback, 10)

        self.get_logger().info('Voice Command Pipeline Node Started')

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Set energy threshold for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        self.recognizer.energy_threshold = 300  # Adjust based on environment
        self.recognizer.dynamic_energy_threshold = True

        # Command queue for processing
        self.command_queue = queue.Queue()
        self.listening = True

        # Start listening thread
        self.listening_thread = threading.Thread(target=self.listen_continuously, daemon=True)
        self.listening_thread.start()

        # Timer to publish listening status
        self.status_timer = self.create_timer(1.0, self.publish_listening_status)

    def publish_listening_status(self):
        """Publish current listening status"""
        status_msg = Bool()
        status_msg.data = self.listening
        self.listening_status_pub.publish(status_msg)

    def listen_continuously(self):
        """Continuously listen for voice commands"""
        while self.listening:
            try:
                with self.microphone as source:
                    self.get_logger().debug('Listening for voice input...')
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=5.0)

                # Process the audio in a separate thread to avoid blocking
                processing_thread = threading.Thread(
                    target=self.process_audio,
                    args=(audio,),
                    daemon=True
                )
                processing_thread.start()

            except sr.WaitTimeoutError:
                # This is normal - just means no speech was detected
                continue
            except Exception as e:
                self.get_logger().error(f'Listening error: {e}')
                time.sleep(0.1)  # Brief pause before retrying

    def process_audio(self, audio):
        """Process captured audio to extract text"""
        try:
            # Use Google Web Speech API for recognition
            # For offline capability, you could use recognize_sphinx or other engines
            text = self.recognizer.recognize_google(audio)
            self.get_logger().info(f'Recognized: {text}')

            # Publish recognized text
            text_msg = String()
            text_msg.data = text
            self.speech_text_pub.publish(text_msg)

            # Parse and publish command
            command = self.parse_command(text)
            if command:
                cmd_msg = String()
                cmd_msg.data = command
                self.command_pub.publish(cmd_msg)
                self.get_logger().info(f'Parsed command: {command}')

        except sr.UnknownValueError:
            self.get_logger().warn('Speech recognition could not understand audio')
        except sr.RequestError as e:
            self.get_logger().error(f'Speech recognition request error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def parse_command(self, text):
        """Parse natural language text into robot commands"""
        text = text.lower().strip()

        # Define command patterns
        command_patterns = {
            'move_forward': ['move forward', 'go forward', 'forward', 'go ahead', 'move ahead'],
            'move_backward': ['move backward', 'go backward', 'backward', 'go back', 'move back'],
            'turn_left': ['turn left', 'go left', 'left', 'turn anticlockwise'],
            'turn_right': ['turn right', 'go right', 'right', 'turn clockwise'],
            'stop': ['stop', 'halt', 'freeze', 'stand still'],
            'follow_me': ['follow me', 'follow', 'come with me'],
            'come_here': ['come here', 'come to me', 'here'],
            'find_person': ['find person', 'find someone', 'look for person'],
            'take_picture': ['take picture', 'photo', 'take photo', 'capture image'],
            'what_time': ['what time', 'time', 'what is the time'],
            'hello': ['hello', 'hi', 'hey', 'greetings'],
            'goodbye': ['goodbye', 'bye', 'see you', 'good bye']
        }

        # Match text to command
        for command, patterns in command_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    return command

        # If no direct match, try to extract intent
        if 'move' in text or 'go' in text:
            if 'forward' in text or 'ahead' in text:
                return 'move_forward'
            elif 'back' in text or 'backward' in text:
                return 'move_backward'
            elif 'left' in text:
                return 'turn_left'
            elif 'right' in text:
                return 'turn_right'

        # If still no match, return as custom command
        return f'custom_{text.replace(" ", "_")}'

    def voice_command_callback(self, msg):
        """Handle incoming voice commands from other nodes"""
        self.get_logger().info(f'Processing voice command: {msg.data}')
        # In a real system, this might trigger specific behaviors
        pass


def main(args=None):
    rclpy.init(args=args)

    node = VoiceCommandPipeline()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Voice Command Pipeline')
        node.listening = False
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()