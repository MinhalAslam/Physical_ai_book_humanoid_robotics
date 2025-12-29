# Chapter 8: Sensor Awareness - The Robot's Senses in Simulation

## Perception in the Digital Twin

In this chapter, we implement sensor awareness systems that enable our robot to perceive its simulated environment. This fulfills our constitution's principle that "Perception Precedes Action"—our robot must first understand its environment before it can act intelligently within it.

## Sensor Integration Architecture

Based on our project specification, we need to implement comprehensive sensor systems that include:
- **FR-006**: System MUST identify objects using RGB + Depth vision with ≥90% recognition accuracy
- **FR-005**: System MUST support SLAM-based mapping and localization for unknown environments

### Key Tasks from Our Plan:
- T044: Create sensor awareness system with RGB-D in src/perception/sensor_node.py

## RGB-D Camera Sensor Node

Let's implement a comprehensive sensor node that handles RGB-D camera data:

```python
#!/usr/bin/env python3
"""
RGB-D Sensor Node for Physical AI System
Handles RGB and depth camera data processing
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Header
import struct
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class RGBDSensorNode(Node):
    def __init__(self):
        super().__init__('rgbd_sensor_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create subscribers for RGB and depth images
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
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Create publishers for processed data
        self.object_detection_pub = self.create_publisher(
            MarkerArray,
            '/detected_objects',
            10
        )
        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            '/point_cloud',
            10
        )
        self.depth_processed_pub = self.create_publisher(
            Image,
            '/depth/processed',
            10
        )

        # Camera parameters (will be updated from camera_info)
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.image_width = 640
        self.image_height = 480

        # Storage for latest images
        self.latest_rgb = None
        self.latest_depth = None

        # Object detection parameters
        self.object_detection_enabled = True
        self.min_object_size = 30  # pixels
        self.depth_threshold = 2.0  # meters

        self.get_logger().info('RGB-D Sensor Node Started')

    def camera_info_callback(self, msg):
        """Update camera parameters from camera info"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        self.image_width = msg.width
        self.image_height = msg.height

        self.get_logger().info(f'Camera parameters updated: {self.image_width}x{self.image_height}')

    def rgb_callback(self, msg):
        """Process RGB image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = cv_image.copy()

            # Process image for object detection
            if self.object_detection_enabled:
                detected_objects = self.detect_objects(cv_image)
                self.publish_detected_objects(detected_objects)

            # Publish processed image if needed
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            processed_msg.header = msg.header
            self.depth_processed_pub.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {str(e)}')

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            # Convert ROS Image to OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth = depth_image.copy()

            # Process depth data
            processed_depth = self.process_depth_data(depth_image)

            # Convert back to ROS message
            processed_msg = self.bridge.cv2_to_imgmsg(processed_depth, encoding='32FC1')
            processed_msg.header = msg.header
            self.depth_processed_pub.publish(processed_msg)

            # If we have both RGB and depth, create point cloud
            if self.latest_rgb is not None:
                point_cloud = self.create_point_cloud(self.latest_rgb, depth_image, msg.header)
                if point_cloud is not None:
                    self.point_cloud_pub.publish(point_cloud)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def detect_objects(self, image):
        """Detect objects in the RGB image"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different objects
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'yellow': ([20, 50, 50], [40, 255, 255])
        }

        detected_objects = []

        for color_name, (lower, upper) in color_ranges.items():
            # Create mask for the color
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_object_size:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center
                    center_x, center_y = x + w//2, y + h//2

                    # Get depth at center point (if depth is available)
                    depth_value = 0.0
                    if self.latest_depth is not None and 0 <= center_y < self.latest_depth.shape[0] and 0 <= center_x < self.latest_depth.shape[1]:
                        depth_value = self.latest_depth[center_y, center_x]
                        if np.isnan(depth_value) or np.isinf(depth_value):
                            depth_value = 0.0

                    # Only add objects within depth threshold
                    if 0 < depth_value <= self.depth_threshold:
                        detected_objects.append({
                            'name': color_name,
                            'area': area,
                            'bbox': (x, y, w, h),
                            'center': (center_x, center_y),
                            'depth': depth_value
                        })

        return detected_objects

    def publish_detected_objects(self, detected_objects):
        """Publish detected objects as visualization markers"""
        marker_array = MarkerArray()

        for i, obj in enumerate(detected_objects):
            # Create marker for the object
            marker = Marker()
            marker.header.frame_id = 'camera_rgb_optical_frame'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_objects'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Set position based on camera matrix and depth
            # Convert pixel coordinates to 3D world coordinates
            u, v = obj['center']
            z = obj['depth']

            # Use camera matrix to convert to 3D
            if self.camera_matrix is not None:
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx = self.camera_matrix[0, 2]
                cy = self.camera_matrix[1, 2]

                # Convert pixel to 3D coordinates
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z

            # Set size based on bounding box
            marker.scale.x = obj['bbox'][2] * z / fx  # width
            marker.scale.y = obj['bbox'][3] * z / fy  # height
            marker.scale.z = 0.1  # depth of the marker

            # Set color based on object color
            color_map = {
                'red': (1.0, 0.0, 0.0),
                'blue': (0.0, 0.0, 1.0),
                'green': (0.0, 1.0, 0.0),
                'yellow': (1.0, 1.0, 0.0)
            }

            r, g, b = color_map.get(obj['name'], (1.0, 1.0, 1.0))
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.7

            marker_array.markers.append(marker)

        self.object_detection_pub.publish(marker_array)

    def process_depth_data(self, depth_image):
        """Process depth image to filter and enhance"""
        # Apply median filter to reduce noise
        filtered_depth = cv2.medianBlur(depth_image, 5)

        # Set invalid depth values to 0
        filtered_depth[np.isnan(filtered_depth)] = 0
        filtered_depth[np.isinf(filtered_depth)] = 0

        # Clamp depth values to reasonable range
        filtered_depth = np.clip(filtered_depth, 0, self.depth_threshold * 2)

        return filtered_depth

    def create_point_cloud(self, rgb_image, depth_image, header):
        """Create point cloud from RGB and depth images"""
        if self.camera_matrix is None:
            return None

        # Get camera parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Create coordinate grids
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D coordinates
        z = depth_image
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()

        # Get colors
        if rgb_image is not None:
            bgr_flat = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        else:
            bgr_flat = np.zeros((height * width, 3), dtype=np.uint8)

        # Create point cloud
        points = []
        for i in range(len(x_flat)):
            if 0 < z_flat[i] < self.depth_threshold * 2:  # Valid depth
                # Pack x, y, z, and RGB
                point_data = struct.pack('fffI',
                    float(x_flat[i]),
                    float(y_flat[i]),
                    float(z_flat[i]),
                    (int(bgr_flat[i][0]) << 16) | (int(bgr_flat[i][1]) << 8) | int(bgr_flat[i][2])
                )
                points.append(point_data)

        if not points:
            return None

        # Create PointCloud2 message
        point_cloud = PointCloud2()
        point_cloud.header = header
        point_cloud.height = 1
        point_cloud.width = len(points)
        point_cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]
        point_cloud.is_bigendian = False
        point_cloud.point_step = 16
        point_cloud.row_step = point_cloud.point_step * point_cloud.width
        point_cloud.is_dense = True
        point_cloud.data = b''.join(points)

        return point_cloud

def main(args=None):
    rclpy.init(args=args)
    sensor_node = RGBDSensorNode()

    try:
        rclpy.spin(sensor_node)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## IMU Sensor Node

Let's also implement an IMU sensor node for orientation and acceleration data:

```python
#!/usr/bin/env python3
"""
IMU Sensor Node for Physical AI System
Handles IMU data for orientation and motion detection
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUSensorNode(Node):
    def __init__(self):
        super().__init__('imu_sensor_node')

        # Create subscriber for IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Create publishers for processed data
        self.roll_pub = self.create_publisher(Float64, '/imu/roll', 10)
        self.pitch_pub = self.create_publisher(Float64, '/imu/pitch', 10)
        self.yaw_pub = self.create_publisher(Float64, '/imu/yaw', 10)
        self.orientation_pub = self.create_publisher(Imu, '/imu/orientation_filtered', 10)

        # IMU parameters
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w
        self.linear_acceleration = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])

        # Filtering parameters
        self.orientation_filter_alpha = 0.1  # For complementary filter
        self.acceleration_threshold = 0.5  # m/s² for motion detection

        self.get_logger().info('IMU Sensor Node Started')

    def imu_callback(self, msg):
        """Process IMU data"""
        # Extract orientation (quaternion)
        orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Extract linear acceleration
        acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Extract angular velocity
        angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Apply complementary filter to orientation
        self.orientation = self.complementary_filter(
            self.orientation, orientation, angular_vel, 0.01  # assuming 100Hz
        )

        # Update other values
        self.linear_acceleration = acceleration
        self.angular_velocity = angular_vel

        # Publish Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(self.orientation)

        roll_msg = Float64()
        roll_msg.data = roll
        self.roll_pub.publish(roll_msg)

        pitch_msg = Float64()
        pitch_msg.data = pitch
        self.pitch_pub.publish(pitch_msg)

        yaw_msg = Float64()
        yaw_msg.data = yaw
        self.yaw_pub.publish(yaw_msg)

        # Publish filtered orientation
        filtered_msg = Imu()
        filtered_msg.header = msg.header
        filtered_msg.orientation.x = self.orientation[0]
        filtered_msg.orientation.y = self.orientation[1]
        filtered_msg.orientation.z = self.orientation[2]
        filtered_msg.orientation.w = self.orientation[3]
        filtered_msg.linear_acceleration = msg.linear_acceleration
        filtered_msg.angular_velocity = msg.angular_velocity

        self.orientation_pub.publish(filtered_msg)

        # Check for significant motion
        if np.linalg.norm(acceleration) > self.acceleration_threshold:
            self.get_logger().info(f'Significant motion detected: {np.linalg.norm(acceleration):.2f} m/s²')

    def complementary_filter(self, current_orientation, measured_orientation, angular_velocity, dt):
        """Apply complementary filter to orientation data"""
        # Convert quaternions to rotation vectors for interpolation
        current_rot = R.from_quat(current_orientation)
        measured_rot = R.from_quat(measured_orientation)

        # Integrate angular velocity
        angular_speed = np.linalg.norm(angular_velocity)
        if angular_speed > 1e-6:  # Avoid division by zero
            axis = angular_velocity / angular_speed
            angle = angular_speed * dt
            integrated_rotation = R.from_rotvec(axis * angle)

            # Apply complementary filter
            predicted_orientation = integrated_rotation * current_rot
            filtered_rot = R.slerp(predicted_orientation, measured_rot)(self.orientation_filter_alpha)
        else:
            # No rotation, just use measured orientation
            filtered_rot = R.slerp(current_rot, measured_rot)(self.orientation_filter_alpha)

        return filtered_rot.as_quat()

    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        r = R.from_quat(quat)
        return r.as_euler('xyz')

def main(args=None):
    rclpy.init(args=args)
    imu_node = IMUSensorNode()

    try:
        rclpy.spin(imu_node)
    except KeyboardInterrupt:
        pass
    finally:
        imu_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Fusion Node

Now let's create a sensor fusion node that combines data from multiple sensors:

```python
#!/usr/bin/env python3
"""
Sensor Fusion Node for Physical AI System
Combines data from multiple sensors for coherent perception
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String, Float64
from visualization_msgs.msg import MarkerArray
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Create subscribers for various sensors
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            TwistStamped,
            '/odom',
            self.odom_callback,
            10
        )
        self.object_sub = self.create_subscription(
            MarkerArray,
            '/detected_objects',
            self.object_callback,
            10
        )

        # Create publishers for fused data
        self.fused_pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)
        self.environment_status_pub = self.create_publisher(String, '/environment_status', 10)

        # Robot state tracking
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, z, w
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])

        # Sensor data buffers
        self.imu_buffer = deque(maxlen=10)
        self.odom_buffer = deque(maxlen=10)

        # Object tracking
        self.tracked_objects = {}

        self.get_logger().info('Sensor Fusion Node Started')

    def imu_callback(self, msg):
        """Process IMU data for orientation and acceleration"""
        # Extract orientation
        self.orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Extract linear acceleration
        linear_acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Extract angular velocity
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Add to buffer for averaging
        self.imu_buffer.append({
            'timestamp': msg.header.stamp,
            'linear_acc': linear_acc,
            'angular_vel': self.angular_velocity
        })

    def odom_callback(self, msg):
        """Process odometry data for position and velocity"""
        # Extract velocity
        self.velocity = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])

        # Update position by integrating velocity
        dt = 0.01  # Assume 100Hz update rate
        self.position += self.velocity * dt

        # Add to buffer
        self.odom_buffer.append({
            'timestamp': msg.header.stamp,
            'velocity': self.velocity
        })

    def object_callback(self, msg):
        """Process detected objects"""
        for marker in msg.markers:
            # Update tracked object positions
            obj_id = marker.id
            obj_pos = np.array([
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            ])

            self.tracked_objects[obj_id] = {
                'position': obj_pos,
                'timestamp': marker.header.stamp,
                'type': marker.ns
            }

        # Publish environment status
        status_msg = String()
        status_msg.data = f'Detected {len(self.tracked_objects)} objects'
        self.environment_status_pub.publish(status_msg)

    def create_fused_pose(self):
        """Create a fused pose estimate combining all sensor data"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # Set position
        pose_msg.pose.position.x = float(self.position[0])
        pose_msg.pose.position.y = float(self.position[1])
        pose_msg.pose.position.z = float(self.position[2])

        # Set orientation
        pose_msg.pose.orientation.x = float(self.orientation[0])
        pose_msg.pose.orientation.y = float(self.orientation[1])
        pose_msg.pose.orientation.z = float(self.orientation[2])
        pose_msg.pose.orientation.w = float(self.orientation[3])

        return pose_msg

    def publish_fused_data(self):
        """Publish the fused sensor data"""
        fused_pose = self.create_fused_pose()
        self.fused_pose_pub.publish(fused_pose)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    # Create timer for publishing fused data
    timer = fusion_node.create_timer(0.05, fusion_node.publish_fused_data)  # 20Hz

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch File for Sensor System

Create a launch file to start all sensor nodes:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # RGB-D sensor node
    rgbd_sensor_node = Node(
        package='physical_ai_perception',
        executable='rgbd_sensor_node',
        name='rgbd_sensor_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # IMU sensor node
    imu_sensor_node = Node(
        package='physical_ai_perception',
        executable='imu_sensor_node',
        name='imu_sensor_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Sensor fusion node
    sensor_fusion_node = Node(
        package='physical_ai_perception',
        executable='sensor_fusion_node',
        name='sensor_fusion_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        rgbd_sensor_node,
        imu_sensor_node,
        sensor_fusion_node
    ])
```

## Quality Assurance for Sensor Systems

### Accuracy Testing
1. **Calibration**: Regularly calibrate sensors using standard procedures
2. **Cross-validation**: Compare sensor readings with ground truth when available
3. **Drift Detection**: Monitor for sensor drift over time
4. **Noise Analysis**: Characterize sensor noise and implement appropriate filtering

### Performance Metrics
- **Update Rate**: Ensure sensors publish at required frequencies
- **Latency**: Minimize delay between sensing and processing
- **Reliability**: Maintain consistent sensor operation
- **Power Consumption**: Monitor sensor power usage in battery-powered robots

## Looking Forward

With our comprehensive sensor awareness system in place, the next chapter will focus on designing human-robot interaction scenes that will allow our robot to engage meaningfully with humans in the simulated environment.

[Continue to Chapter 9: Human-Robot Interaction Scene Design](./chapter-9-hri-scene.md)