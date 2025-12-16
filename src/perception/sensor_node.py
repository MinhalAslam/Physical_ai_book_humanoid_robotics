#!/usr/bin/env python3
"""
Sensor Awareness System with RGB-D

This node processes data from RGB-D sensors and creates awareness of the environment.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, LaserScan
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')

        # Initialize CvBridge for image processing
        self.bridge = CvBridge()

        # Publishers
        self.environment_map_pub = self.create_publisher(String, '/environment_map', 10)
        self.object_detection_pub = self.create_publisher(String, '/object_detections', 10)

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        # Timer for processing
        self.process_timer = self.create_timer(0.5, self.process_sensors)

        self.get_logger().info('Sensor Awareness Node Started')

        # Sensor data storage
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_laser_scan = None
        self.environment_map = {}

    def rgb_callback(self, msg):
        """Process RGB camera data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_rgb_image = cv_image
            self.get_logger().debug('Received RGB image')
        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        """Process depth camera data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")  # 32-bit float depth
            self.latest_depth_image = cv_image
            self.get_logger().debug('Received depth image')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.latest_laser_scan = msg
        self.get_logger().debug('Received laser scan')

    def process_sensors(self):
        """Process all sensor data to create environmental awareness"""
        if self.latest_rgb_image is not None and self.latest_depth_image is not None:
            # Perform basic object detection on RGB image
            objects = self.detect_objects(self.latest_rgb_image)

            # Create environment map using depth data
            environment_info = self.create_environment_map(
                self.latest_depth_image,
                self.latest_laser_scan
            )

            # Publish environment map
            env_msg = String()
            env_msg.data = str(environment_info)
            self.environment_map_pub.publish(env_msg)

            # Publish object detections
            obj_msg = String()
            obj_msg.data = str(objects)
            self.object_detection_pub.publish(obj_msg)

            self.get_logger().info(f'Detected {len(objects)} objects, environment mapped')

    def detect_objects(self, image):
        """Simple object detection using color and shape analysis"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for common objects
        color_ranges = [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),    # Red
            (np.array([170, 50, 50]), np.array([180, 255, 255])), # Red (wrapping)
            (np.array([36, 50, 50]), np.array([86, 255, 255])),   # Green
            (np.array([100, 50, 50]), np.array([130, 255, 255])), # Blue
        ]

        objects = []

        for i, (lower, upper) in enumerate(color_ranges):
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small areas
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    obj_info = {
                        'type': f'color_object_{i}',
                        'area': area,
                        'position': {'x': x, 'y': y, 'width': w, 'height': h},
                        'confidence': min(0.95, area / 10000)  # Normalize confidence
                    }
                    objects.append(obj_info)

        return objects

    def create_environment_map(self, depth_image, laser_scan):
        """Create environment map using depth and laser data"""
        env_map = {
            'timestamp': self.get_clock().now().to_msg().sec,
            'obstacles': [],
            'free_space': [],
            'depth_analysis': {}
        }

        if depth_image is not None:
            # Analyze depth image for obstacles
            height, width = depth_image.shape
            center_x, center_y = width // 2, height // 2

            # Sample depth values in different regions
            center_depth = depth_image[center_y, center_x]
            left_depth = depth_image[center_y, width // 4]
            right_depth = depth_image[center_y, 3 * width // 4]

            env_map['depth_analysis'] = {
                'center': float(center_depth) if not np.isnan(center_depth) else float('inf'),
                'left': float(left_depth) if not np.isnan(left_depth) else float('inf'),
                'right': float(right_depth) if not np.isnan(right_depth) else float('inf')
            }

        if laser_scan is not None:
            # Analyze laser scan for nearby obstacles
            min_distance = min([r for r in laser_scan.ranges if r > 0 and not math.isinf(r)], default=float('inf'))
            env_map['closest_obstacle'] = float(min_distance)

            # Categorize distances
            front_distances = laser_scan.ranges[:len(laser_scan.ranges)//8] + laser_scan.ranges[-len(laser_scan.ranges)//8:]
            close_obstacles = [d for d in front_distances if 0 < d < 1.0]  # Within 1 meter

            env_map['nearby_obstacles'] = len(close_obstacles)

        return env_map


def main(args=None):
    rclpy.init(args=args)

    node = SensorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Sensor Awareness Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()