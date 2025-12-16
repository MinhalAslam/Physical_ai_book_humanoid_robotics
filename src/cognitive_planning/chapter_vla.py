#!/usr/bin/env python3
"""
Vision-Language-Action System

This node integrates vision, language understanding, and action execution
to create a complete VLA pipeline for the robot.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time
from typing import Dict, List, Optional


class VisionLanguageActionSystem(Node):
    def __init__(self):
        super().__init__('vla_system')

        # Initialize CvBridge for image processing
        self.bridge = CvBridge()

        # Publishers
        self.vla_output_pub = self.create_publisher(String, '/vla_output', 10)
        self.vision_analysis_pub = self.create_publisher(String, '/vision_analysis', 10)
        self.action_request_pub = self.create_publisher(String, '/vla_action_request', 10)
        self.scene_description_pub = self.create_publisher(String, '/scene_description', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(
            String, '/parsed_command', self.command_callback, 10)
        self.thought_sub = self.create_subscription(
            String, '/llm_thought', self.thought_callback, 10)

        self.get_logger().info('Vision-Language-Action System Node Started')

        # VLA state
        self.latest_image = None
        self.latest_command = ""
        self.latest_thought = ""
        self.scene_objects = []
        self.command_queue = []

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
            self.get_logger().debug('Received image for VLA processing')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming voice commands"""
        self.latest_command = msg.data
        self.get_logger().info(f'Received command for VLA: {msg.data}')

        # If we have an image, process the VLA pipeline
        if self.latest_image is not None:
            self.process_vla_pipeline()

    def thought_callback(self, msg):
        """Receive LLM thoughts for integration"""
        self.latest_thought = msg.data
        self.get_logger().debug('Received LLM thought for VLA integration')

    def process_vla_pipeline(self):
        """Process the complete Vision-Language-Action pipeline"""
        if self.latest_image is None or not self.latest_command:
            return

        # Step 1: Analyze the visual scene
        vision_analysis = self.analyze_scene(self.latest_image)

        # Step 2: Integrate with language command
        integrated_analysis = self.integrate_vision_language(
            vision_analysis, self.latest_command, self.latest_thought
        )

        # Step 3: Generate action plan
        action_plan = self.generate_action_plan(integrated_analysis)

        # Step 4: Publish results
        self.publish_vla_results(vision_analysis, integrated_analysis, action_plan)

    def analyze_scene(self, image):
        """Analyze the visual scene to identify objects and their properties"""
        analysis = {
            'timestamp': time.time(),
            'objects': [],
            'scene_description': '',
            'spatial_relationships': [],
            'color_analysis': {},
            'shape_analysis': {}
        }

        # Detect objects using color and shape analysis
        objects = self.detect_objects(image)

        # Analyze colors in the scene
        color_analysis = self.analyze_colors(image)

        # Analyze shapes and spatial relationships
        shape_analysis = self.analyze_shapes(image)

        analysis['objects'] = objects
        analysis['color_analysis'] = color_analysis
        analysis['shape_analysis'] = shape_analysis

        # Create scene description
        object_names = [obj['type'] for obj in objects]
        analysis['scene_description'] = f"Scene contains: {', '.join(object_names) if object_names else 'no recognizable objects'}"

        # Analyze spatial relationships
        if len(objects) >= 2:
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    rel = self.calculate_spatial_relationship(objects[i], objects[j])
                    analysis['spatial_relationships'].append(rel)

        return analysis

    def detect_objects(self, image):
        """Detect objects in the image using computer vision techniques"""
        objects = []

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define common color ranges
        color_ranges = [
            {'name': 'red', 'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            {'name': 'red2', 'lower': np.array([170, 50, 50]), 'upper': np.array([180, 255, 255])},  # Wrapping red
            {'name': 'green', 'lower': np.array([36, 50, 50]), 'upper': np.array([86, 255, 255])},
            {'name': 'blue', 'lower': np.array([100, 50, 50]), 'upper': np.array([130, 255, 255])},
            {'name': 'yellow', 'lower': np.array([20, 50, 50]), 'upper': np.array([35, 255, 255])},
        ]

        height, width = image.shape[:2]

        for color_info in color_ranges:
            mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small areas
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Calculate relative position
                    rel_x = center_x / width
                    rel_y = center_y / height

                    obj_info = {
                        'type': f"{color_info['name']}_object",
                        'color': color_info['name'],
                        'area': area,
                        'position': {'x': center_x, 'y': center_y},
                        'relative_position': {'x': rel_x, 'y': rel_y},
                        'size': {'width': w, 'height': h},
                        'confidence': min(0.95, area / 10000)
                    }

                    objects.append(obj_info)

        # Also detect large geometric shapes
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Large shapes only
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                shape_name = "unknown"
                if len(approx) == 3:
                    shape_name = "triangle"
                elif len(approx) == 4:
                    # Check if it's square or rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    if 0.95 <= aspect_ratio <= 1.05:
                        shape_name = "square"
                    else:
                        shape_name = "rectangle"
                elif len(approx) > 4:
                    shape_name = "circle"  # Approximate circles/polygons with many sides

                if shape_name != "unknown":
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    rel_x = center_x / width
                    rel_y = center_y / height

                    obj_info = {
                        'type': f"{shape_name}",
                        'shape': shape_name,
                        'area': area,
                        'position': {'x': center_x, 'y': center_y},
                        'relative_position': {'x': rel_x, 'y': rel_y},
                        'size': {'width': w, 'height': h},
                        'confidence': min(0.8, area / 5000)
                    }

                    # Avoid duplicate detections
                    is_duplicate = False
                    for existing_obj in objects:
                        dist = np.sqrt(
                            (existing_obj['position']['x'] - center_x) ** 2 +
                            (existing_obj['position']['y'] - center_y) ** 2
                        )
                        if dist < 50:  # Close enough to be the same object
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        objects.append(obj_info)

        return objects

    def analyze_colors(self, image):
        """Analyze color distribution in the image"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        # Find dominant colors
        dominant_hue = np.argmax(h_hist)
        dominant_sat = np.argmax(s_hist)
        dominant_val = np.argmax(v_hist)

        return {
            'dominant_hue': int(dominant_hue),
            'dominant_saturation': int(dominant_sat),
            'dominant_value': int(dominant_val),
            'colorfulness': float(np.std(hsv)),
            'brightness': float(np.mean(hsv[:, :, 2]))
        }

    def analyze_shapes(self, image):
        """Analyze shapes and spatial relationships"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Detect edges and lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

        line_count = 0
        if lines is not None:
            line_count = len(lines)

        return {
            'edge_density': float(np.sum(edges) / (height * width)),
            'line_count': line_count,
            'complexity': float(np.sum(edges) / 255.0)  # Normalized edge count
        }

    def calculate_spatial_relationship(self, obj1, obj2):
        """Calculate spatial relationship between two objects"""
        dx = obj2['position']['x'] - obj1['position']['x']
        dy = obj2['position']['y'] - obj1['position']['y']
        distance = np.sqrt(dx**2 + dy**2)

        # Determine direction
        angle = np.arctan2(dy, dx) * 180 / np.pi

        direction = "unknown"
        if -45 <= angle < 45:
            direction = "right"
        elif 45 <= angle < 135:
            direction = "down"
        elif -135 <= angle < -45:
            direction = "up"
        else:
            direction = "left"

        return {
            'object1': obj1['type'],
            'object2': obj2['type'],
            'distance': distance,
            'direction': direction,
            'angle': angle
        }

    def integrate_vision_language(self, vision_analysis, command, thought):
        """Integrate visual information with language command and LLM thought"""
        integration = {
            'command': command,
            'thought': thought,
            'vision_analysis': vision_analysis,
            'integration_result': '',
            'relevant_objects': [],
            'action_context': {}
        }

        # Find objects relevant to the command
        command_lower = command.lower()
        relevant_objects = []

        for obj in vision_analysis['objects']:
            # Check if object type is relevant to command
            obj_type = obj['type'].lower()

            # Simple relevance matching
            relevance_keywords = {
                'red_object': ['red', 'red object'],
                'green_object': ['green', 'green object'],
                'blue_object': ['blue', 'blue object'],
                'square': ['square', 'box', 'container'],
                'circle': ['circle', 'round', 'ball'],
                'triangle': ['triangle', 'pyramid', 'cone']
            }

            for keyword in relevance_keywords.get(obj_type, []):
                if keyword in command_lower:
                    relevant_objects.append(obj)
                    break

        integration['relevant_objects'] = relevant_objects

        # Create integration result
        if relevant_objects:
            obj_names = [obj['type'] for obj in relevant_objects]
            integration['integration_result'] = f"Command '{command}' relates to objects: {', '.join(obj_names)}"
        else:
            integration['integration_result'] = f"Command '{command}' - no directly relevant objects detected in scene"

        # Set action context based on integration
        integration['action_context'] = {
            'has_relevant_objects': len(relevant_objects) > 0,
            'object_count': len(vision_analysis['objects']),
            'command_understood': len(command) > 0
        }

        return integration

    def generate_action_plan(self, integrated_analysis):
        """Generate executable action plan based on integrated analysis"""
        command = integrated_analysis['command'].lower()
        relevant_objects = integrated_analysis['relevant_objects']

        action_plan = {
            'command': integrated_analysis['command'],
            'actions': [],
            'confidence': 0.0,
            'reasoning': ''
        }

        # Generate actions based on command type
        if 'move' in command or 'go' in command:
            # Movement commands
            actions = []
            if 'forward' in command:
                actions = [
                    {'action': 'check_environment', 'params': {}},
                    {'action': 'move_forward', 'params': {'distance': 1.0}},
                    {'action': 'stop', 'params': {}}
                ]
            elif 'backward' in command:
                actions = [
                    {'action': 'check_environment', 'params': {}},
                    {'action': 'move_backward', 'params': {'distance': 1.0}},
                    {'action': 'stop', 'params': {}}
                ]
            elif 'left' in command:
                actions = [
                    {'action': 'check_environment', 'params': {}},
                    {'action': 'turn_left', 'params': {'angle': 90}},
                    {'action': 'stop', 'params': {}}
                ]
            elif 'right' in command:
                actions = [
                    {'action': 'check_environment', 'params': {}},
                    {'action': 'turn_right', 'params': {'angle': 90}},
                    {'action': 'stop', 'params': {}}
                ]
            else:
                # General movement
                actions = [
                    {'action': 'check_environment', 'params': {}},
                    {'action': 'move_forward', 'params': {'distance': 0.5}},
                    {'action': 'stop', 'params': {}}
                ]

            action_plan['actions'] = actions
            action_plan['confidence'] = 0.9

        elif 'find' in command or 'look' in command:
            # Object finding commands
            target_color = None
            if 'red' in command:
                target_color = 'red'
            elif 'green' in command:
                target_color = 'green'
            elif 'blue' in command:
                target_color = 'blue'

            if target_color:
                # Look for specific colored object
                found_obj = None
                for obj in relevant_objects:
                    if target_color in obj['type']:
                        found_obj = obj
                        break

                if found_obj:
                    action_plan['actions'] = [
                        {'action': 'look_at_object', 'params': {'object_id': found_obj['type']}},
                        {'action': 'report_location', 'params': {'object': found_obj}}
                    ]
                    action_plan['confidence'] = 0.85
                else:
                    action_plan['actions'] = [
                        {'action': 'scan_environment', 'params': {'color': target_color}},
                        {'action': 'report_not_found', 'params': {'color': target_color}}
                    ]
                    action_plan['confidence'] = 0.7
            else:
                # General find command
                action_plan['actions'] = [
                    {'action': 'scan_environment', 'params': {}},
                    {'action': 'report_objects', 'params': {'count': len(relevant_objects)}}
                ]
                action_plan['confidence'] = 0.75

        elif 'follow' in command:
            # Following commands
            action_plan['actions'] = [
                {'action': 'detect_person', 'params': {}},
                {'action': 'start_following', 'params': {'target': 'person'}},
                {'action': 'maintain_distance', 'params': {'distance': 1.0}}
            ]
            action_plan['confidence'] = 0.8

        elif 'stop' in command:
            # Stop command
            action_plan['actions'] = [
                {'action': 'emergency_stop', 'params': {}},
                {'action': 'report_status', 'params': {'status': 'stopped'}}
            ]
            action_plan['confidence'] = 0.95

        else:
            # Default action for unrecognized commands
            action_plan['actions'] = [
                {'action': 'acknowledge_command', 'params': {'command': command}},
                {'action': 'request_clarification', 'params': {}}
            ]
            action_plan['confidence'] = 0.6

        action_plan['reasoning'] = f"Generated plan for command '{command}' based on visual scene analysis"

        return action_plan

    def publish_vla_results(self, vision_analysis, integrated_analysis, action_plan):
        """Publish the complete VLA results"""
        # Publish vision analysis
        vision_msg = String()
        vision_msg.data = json.dumps(vision_analysis, indent=2)
        self.vision_analysis_pub.publish(vision_msg)

        # Publish scene description
        scene_msg = String()
        scene_msg.data = integrated_analysis['integration_result']
        self.scene_description_pub.publish(scene_msg)

        # Publish complete VLA output
        vla_output = {
            'vision_analysis': vision_analysis,
            'integration_analysis': integrated_analysis,
            'action_plan': action_plan,
            'timestamp': time.time()
        }

        output_msg = String()
        output_msg.data = json.dumps(vla_output, indent=2)
        self.vla_output_pub.publish(output_msg)

        # Publish action requests
        for action in action_plan['actions']:
            action_msg = String()
            action_msg.data = json.dumps(action)
            self.action_request_pub.publish(action_msg)

        self.get_logger().info(f'VLA Pipeline completed - {len(action_plan["actions"])} actions generated')


def main(args=None):
    rclpy.init(args=args)

    node = VisionLanguageActionSystem()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VLA System')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()