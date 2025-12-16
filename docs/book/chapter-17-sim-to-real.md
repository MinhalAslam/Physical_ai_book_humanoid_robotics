# Chapter 17: Sim-to-Real Transfer - Bridging Simulation and Reality

## From Digital Twin to Physical Robot

In this chapter, we focus on the critical process of transferring our robot's capabilities from simulation to the real world. This aligns with our constitution's principle that "Simulation is Truth, Reality is the Test." We'll explore techniques and methodologies to ensure that our robot can successfully operate in the physical world after being trained and validated in simulation.

## Understanding Sim-to-Real Transfer

Sim-to-Real transfer involves adapting models, behaviors, and control strategies developed in simulation to work effectively in the real world. This process addresses the "reality gap"—the differences between simulated and real environments that can cause trained systems to fail when deployed.

### Key Tasks from Our Plan:
- T053: Implement sim-to-real transfer in src/sim_to_real.py

## Domain Randomization Node

Let's create a node that implements domain randomization techniques to improve sim-to-real transfer:

```python
#!/usr/bin/env python3
"""
Domain Randomization Node for Physical AI System
Implements domain randomization techniques for sim-to-real transfer
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose
from cv_bridge import CvBridge
import numpy as np
import cv2
import random
import json
from typing import Dict, List, Tuple
import threading
import time

class DomainRandomizationNode(Node):
    def __init__(self):
        super().__init__('domain_randomization_node')

        # Create subscribers for sensor data
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
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        self.sim_control_sub = self.create_subscription(
            String,
            '/sim_control',
            self.sim_control_callback,
            10
        )

        # Create publishers for randomized data
        self.randomized_rgb_pub = self.create_publisher(
            Image,
            '/camera/rgb/image_randomized',
            10
        )
        self.randomized_depth_pub = self.create_publisher(
            Image,
            '/camera/depth/image_randomized',
            10
        )
        self.randomized_laser_pub = self.create_publisher(
            LaserScan,
            '/scan_randomized',
            10
        )
        self.domain_params_pub = self.create_publisher(
            String,
            '/domain_parameters',
            10
        )

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Domain randomization parameters
        self.randomization_enabled = True
        self.randomization_frequency = 1.0  # Hz
        self.domain_parameters = {
            'lighting': {
                'brightness_range': [0.5, 1.5],
                'contrast_range': [0.8, 1.2],
                'color_temperature_range': [3000, 8000]  # Kelvin
            },
            'textures': {
                'roughness_range': [0.0, 1.0],
                'specular_range': [0.0, 1.0]
            },
            'dynamics': {
                'friction_range': [0.1, 0.9],
                'restitution_range': [0.0, 0.5]
            },
            'sensors': {
                'noise_level_range': [0.0, 0.1],
                'bias_range': [-0.05, 0.05],
                'drift_range': [-0.01, 0.01]
            }
        }

        # Current domain parameters
        self.current_params = self.generate_random_parameters()

        # Create timer for parameter updates
        self.param_timer = self.create_timer(1.0/self.randomization_frequency, self.update_parameters)

        # Storage for sensor data
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_laser = None

        self.get_logger().info('Domain Randomization Node Started')

    def rgb_callback(self, msg):
        """Process RGB image and apply randomization"""
        if not self.randomization_enabled:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Apply domain randomization to image
            randomized_image = self.randomize_image(cv_image, self.current_params)

            # Publish randomized image
            randomized_msg = self.bridge.cv2_to_imgmsg(randomized_image, encoding='bgr8')
            randomized_msg.header = msg.header
            self.randomized_rgb_pub.publish(randomized_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {str(e)}')

    def depth_callback(self, msg):
        """Process depth image and apply randomization"""
        if not self.randomization_enabled:
            return

        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

            # Apply domain randomization to depth
            randomized_depth = self.randomize_depth(depth_image, self.current_params)

            # Publish randomized depth
            randomized_msg = self.bridge.cv2_to_imgmsg(randomized_depth, encoding='32FC1')
            randomized_msg.header = msg.header
            self.randomized_depth_pub.publish(randomized_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def laser_callback(self, msg):
        """Process laser scan and apply randomization"""
        if not self.randomization_enabled:
            return

        # Apply domain randomization to laser data
        randomized_scan = self.randomize_laser_scan(msg, self.current_params)

        # Publish randomized laser scan
        self.randomized_laser_pub.publish(randomized_scan)

    def sim_control_callback(self, msg):
        """Handle simulation control commands"""
        try:
            control_data = json.loads(msg.data)
            command = control_data.get('command', '')

            if command == 'enable_randomization':
                self.randomization_enabled = True
                self.get_logger().info('Domain randomization enabled')
            elif command == 'disable_randomization':
                self.randomization_enabled = False
                self.get_logger().info('Domain randomization disabled')
            elif command == 'update_parameters':
                self.current_params = self.generate_random_parameters()
                self.get_logger().info('Domain parameters updated')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in sim control message')

    def randomize_image(self, image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply domain randomization to an image"""
        randomized = image.copy().astype(np.float32)

        # Apply brightness variation
        brightness_factor = random.uniform(
            params['lighting']['brightness_range'][0],
            params['lighting']['brightness_range'][1]
        )
        randomized *= brightness_factor

        # Apply contrast variation
        contrast_factor = random.uniform(
            params['lighting']['contrast_range'][0],
            params['lighting']['contrast_range'][1]
        )
        randomized = (randomized - 127.5) * contrast_factor + 127.5

        # Apply noise
        noise_level = random.uniform(
            params['sensors']['noise_level_range'][0],
            params['sensors']['noise_level_range'][1]
        )
        noise = np.random.normal(0, noise_level * 255, randomized.shape)
        randomized += noise

        # Clip values to valid range
        randomized = np.clip(randomized, 0, 255).astype(np.uint8)

        return randomized

    def randomize_depth(self, depth_image: np.ndarray, params: Dict) -> np.ndarray:
        """Apply domain randomization to depth image"""
        randomized = depth_image.copy().astype(np.float32)

        # Apply noise to depth values
        noise_level = random.uniform(
            params['sensors']['noise_level_range'][0],
            params['sensors']['noise_level_range'][1]
        )
        noise = np.random.normal(0, noise_level, randomized.shape)
        randomized += noise

        # Apply bias
        bias = random.uniform(
            params['sensors']['bias_range'][0],
            params['sensors']['bias_range'][1]
        )
        randomized += bias

        # Ensure no negative depths
        randomized = np.maximum(randomized, 0)

        return randomized

    def randomize_laser_scan(self, scan_msg: LaserScan, params: Dict) -> LaserScan:
        """Apply domain randomization to laser scan"""
        randomized_scan = LaserScan()
        randomized_scan.header = scan_msg.header
        randomized_scan.angle_min = scan_msg.angle_min
        randomized_scan.angle_max = scan_msg.angle_max
        randomized_scan.angle_increment = scan_msg.angle_increment
        randomized_scan.time_increment = scan_msg.time_increment
        randomized_scan.scan_time = scan_msg.scan_time
        randomized_scan.range_min = scan_msg.range_min
        randomized_scan.range_max = scan_msg.range_max

        # Convert ranges to numpy array for processing
        ranges = np.array(scan_msg.ranges)

        # Apply noise
        noise_level = random.uniform(
            params['sensors']['noise_level_range'][0],
            params['sensors']['noise_level_range'][1]
        )
        noise = np.random.normal(0, noise_level, ranges.shape)
        randomized_ranges = ranges + noise

        # Apply bias
        bias = random.uniform(
            params['sensors']['bias_range'][0],
            params['sensors']['bias_range'][1]
        )
        randomized_ranges += bias

        # Ensure valid range values
        randomized_ranges = np.clip(randomized_ranges, scan_msg.range_min, scan_msg.range_max)

        # Handle invalid values (nan, inf)
        randomized_ranges = np.nan_to_num(randomized_ranges, nan=scan_msg.range_max, posinf=scan_msg.range_max, neginf=scan_msg.range_min)

        randomized_scan.ranges = randomized_ranges.tolist()
        randomized_scan.intensities = scan_msg.intensities  # Keep original intensities

        return randomized_scan

    def generate_random_parameters(self) -> Dict:
        """Generate random domain parameters"""
        params = {}

        for category, param_ranges in self.domain_parameters.items():
            params[category] = {}
            for param_name, value_range in param_ranges.items():
                if isinstance(value_range, list) and len(value_range) == 2:
                    params[category][param_name] = random.uniform(value_range[0], value_range[1])
                else:
                    params[category][param_name] = value_range

        # Publish parameters
        params_msg = String()
        params_msg.data = json.dumps(params, indent=2)
        self.domain_params_pub.publish(params_msg)

        return params

    def update_parameters(self):
        """Update domain parameters periodically"""
        if self.randomization_enabled:
            self.current_params = self.generate_random_parameters()
            self.get_logger().info('Domain parameters updated for sim-to-real transfer')

def main(args=None):
    rclpy.init(args=args)
    dr_node = DomainRandomizationNode()

    try:
        rclpy.spin(dr_node)
    except KeyboardInterrupt:
        pass
    finally:
        dr_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Reality Gap Compensation Node

Now let's create a node that compensates for the reality gap:

```python
#!/usr/bin/env python3
"""
Reality Gap Compensation Node for Physical AI System
Implements techniques to compensate for the reality gap between simulation and reality
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from std_msgs.msg import String, Bool, Float64
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import cv2
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class RealityGapMetrics:
    """Metrics for measuring reality gap"""
    visual_difference: float = 0.0
    sensor_noise: float = 0.0
    dynamics_mismatch: float = 0.0
    timestamp: float = 0.0

class RealityGapCompensationNode(Node):
    def __init__(self):
        super().__init__('reality_gap_compensation_node')

        # Create subscribers
        self.sim_image_sub = self.create_subscription(
            Image,
            '/sim/camera/rgb/image_raw',
            self.sim_image_callback,
            10
        )
        self.real_image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.real_image_callback,
            10
        )
        self.sim_laser_sub = self.create_subscription(
            LaserScan,
            '/sim/scan',
            self.sim_laser_callback,
            10
        )
        self.real_laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.real_laser_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Create publishers
        self.compensated_cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel_compensated',
            10
        )
        self.reality_gap_metrics_pub = self.create_publisher(
            String,
            '/reality_gap_metrics',
            10
        )
        self.adaptation_params_pub = self.create_publisher(
            String,
            '/adaptation_parameters',
            10
        )

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Reality gap compensation parameters
        self.compensation_enabled = True
        self.adaptation_rate = 0.1  # How quickly to adapt
        self.visual_threshold = 0.1  # Threshold for visual difference
        self.sensor_threshold = 0.05  # Threshold for sensor difference

        # Storage for comparison
        self.last_sim_image = None
        self.last_real_image = None
        self.last_sim_laser = None
        self.last_real_laser = None

        # Adaptation parameters
        self.adaptation_params = {
            'visual_compensation': 1.0,
            'sensor_compensation': 1.0,
            'dynamics_compensation': 1.0,
            'control_scaling': 1.0
        }

        # Metrics tracking
        self.metrics_history = []
        self.max_metrics_history = 100

        # Create timer for gap analysis
        self.analysis_timer = self.create_timer(1.0, self.analyze_reality_gap)

        self.get_logger().info('Reality Gap Compensation Node Started')

    def sim_image_callback(self, msg):
        """Process simulated image"""
        try:
            self.last_sim_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing sim image: {str(e)}')

    def real_image_callback(self, msg):
        """Process real image"""
        try:
            self.last_real_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing real image: {str(e)}')

    def sim_laser_callback(self, msg):
        """Process simulated laser scan"""
        self.last_sim_laser = msg

    def real_laser_callback(self, msg):
        """Process real laser scan"""
        self.last_real_laser = msg

    def odom_callback(self, msg):
        """Process odometry for dynamics comparison"""
        # Store for dynamics analysis
        pass

    def analyze_reality_gap(self):
        """Analyze the reality gap and update compensation parameters"""
        if not self.compensation_enabled:
            return

        gap_metrics = RealityGapMetrics()
        gap_metrics.timestamp = time.time()

        # Analyze visual difference
        if self.last_sim_image is not None and self.last_real_image is not None:
            gap_metrics.visual_difference = self.calculate_visual_difference(
                self.last_sim_image, self.last_real_image
            )

        # Analyze sensor difference
        if self.last_sim_laser is not None and self.last_real_laser is not None:
            gap_metrics.sensor_noise = self.calculate_sensor_difference(
                self.last_sim_laser, self.last_real_laser
            )

        # Calculate dynamics mismatch (simplified)
        gap_metrics.dynamics_mismatch = abs(
            gap_metrics.visual_difference - gap_metrics.sensor_noise
        ) * 0.5

        # Add to history
        self.metrics_history.append(gap_metrics)
        if len(self.metrics_history) > self.max_metrics_history:
            self.metrics_history.pop(0)

        # Update adaptation parameters based on gap metrics
        self.update_adaptation_parameters(gap_metrics)

        # Publish metrics
        metrics_msg = String()
        metrics_msg.data = json.dumps({
            'visual_difference': gap_metrics.visual_difference,
            'sensor_noise': gap_metrics.sensor_noise,
            'dynamics_mismatch': gap_metrics.dynamics_mismatch,
            'timestamp': gap_metrics.timestamp
        })
        self.reality_gap_metrics_pub.publish(metrics_msg)

        self.get_logger().info(
            f'Reality gap - Visual: {gap_metrics.visual_difference:.3f}, '
            f'Sensor: {gap_metrics.sensor_noise:.3f}, '
            f'Dynamics: {gap_metrics.dynamics_mismatch:.3f}'
        )

    def calculate_visual_difference(self, sim_img: np.ndarray, real_img: np.ndarray) -> float:
        """Calculate visual difference between sim and real images"""
        # Resize images to same size if needed
        if sim_img.shape != real_img.shape:
            real_img = cv2.resize(real_img, (sim_img.shape[1], sim_img.shape[0]))

        # Convert to grayscale for comparison
        sim_gray = cv2.cvtColor(sim_img, cv2.COLOR_BGR2GRAY)
        real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)

        # Calculate mean squared difference
        diff = cv2.absdiff(sim_gray.astype(np.float32), real_gray.astype(np.float32))
        mse = np.mean(diff ** 2)

        # Normalize to [0, 1] range
        normalized_diff = mse / (255.0 ** 2)

        return float(normalized_diff)

    def calculate_sensor_difference(self, sim_scan: LaserScan, real_scan: LaserScan) -> float:
        """Calculate difference between sim and real laser scans"""
        if len(sim_scan.ranges) != len(real_scan.ranges):
            return 0.5  # Default difference if lengths don't match

        # Calculate mean absolute difference
        sim_ranges = np.array(sim_scan.ranges)
        real_ranges = np.array(real_scan.ranges)

        # Handle invalid values
        sim_ranges = np.nan_to_num(sim_ranges, nan=np.inf)
        real_ranges = np.nan_to_num(real_ranges, nan=np.inf)

        # Calculate difference, ignoring infinite values
        valid_mask = (sim_ranges != np.inf) & (real_ranges != np.inf)
        if np.any(valid_mask):
            diff = np.abs(sim_ranges[valid_mask] - real_ranges[valid_mask])
            mean_diff = np.mean(diff)
            # Normalize by max range
            normalized_diff = min(mean_diff / sim_scan.range_max, 1.0)
            return float(normalized_diff)
        else:
            return 0.0

    def update_adaptation_parameters(self, metrics: RealityGapMetrics):
        """Update adaptation parameters based on reality gap metrics"""
        # Adjust visual compensation
        if metrics.visual_difference > self.visual_threshold:
            self.adaptation_params['visual_compensation'] *= (1.0 - self.adaptation_rate)
        else:
            self.adaptation_params['visual_compensation'] *= (1.0 + self.adaptation_rate * 0.1)

        # Adjust sensor compensation
        if metrics.sensor_noise > self.sensor_threshold:
            self.adaptation_params['sensor_compensation'] *= (1.0 - self.adaptation_rate)
        else:
            self.adaptation_params['sensor_compensation'] *= (1.0 + self.adaptation_rate * 0.1)

        # Adjust dynamics compensation based on overall gap
        overall_gap = (metrics.visual_difference + metrics.sensor_noise + metrics.dynamics_mismatch) / 3.0
        if overall_gap > 0.3:  # High gap
            self.adaptation_params['dynamics_compensation'] = max(0.5, self.adaptation_params['dynamics_compensation'] * (1.0 - self.adaptation_rate))
            self.adaptation_params['control_scaling'] = max(0.7, self.adaptation_params['control_scaling'] * (1.0 - self.adaptation_rate * 0.5))
        elif overall_gap < 0.1:  # Low gap
            self.adaptation_params['dynamics_compensation'] = min(1.5, self.adaptation_params['dynamics_compensation'] * (1.0 + self.adaptation_rate * 0.1))
            self.adaptation_params['control_scaling'] = min(1.2, self.adaptation_params['control_scaling'] * (1.0 + self.adaptation_rate * 0.1))

        # Ensure parameters stay within reasonable bounds
        for key in self.adaptation_params:
            self.adaptation_params[key] = max(0.1, min(2.0, self.adaptation_params[key]))

        # Publish adaptation parameters
        params_msg = String()
        params_msg.data = json.dumps(self.adaptation_params, indent=2)
        self.adaptation_params_pub.publish(params_msg)

    def compensate_command(self, original_cmd: Twist) -> Twist:
        """Apply compensation to a command based on adaptation parameters"""
        compensated_cmd = Twist()

        # Apply control scaling
        scaling = self.adaptation_params['control_scaling']
        compensated_cmd.linear.x = original_cmd.linear.x * scaling
        compensated_cmd.linear.y = original_cmd.linear.y * scaling
        compensated_cmd.linear.z = original_cmd.linear.z * scaling
        compensated_cmd.angular.x = original_cmd.angular.x * scaling
        compensated_cmd.angular.y = original_cmd.angular.y * scaling
        compensated_cmd.angular.z = original_cmd.angular.z * scaling

        return compensated_cmd

def main(args=None):
    rclpy.init(args=args)
    rgc_node = RealityGapCompensationNode()

    try:
        rclpy.spin(rgc_node)
    except KeyboardInterrupt:
        pass
    finally:
        rgc_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Calibraton and Adaptation Node

Let's create a node for calibration and adaptation:

```python
#!/usr/bin/env python3
"""
Calibration and Adaptation Node for Physical AI System
Handles sensor calibration and system adaptation for sim-to-real transfer
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan, Imu, JointState
from std_msgs.msg import String, Bool, Float64
from geometry_msgs.msg import Twist, Pose, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import cv2
from cv_bridge import CvBridge
import json
import time
from typing import Dict, List, Tuple
import threading

class CalibrationAdaptationNode(Node):
    def __init__(self):
        super().__init__('calibration_adaptation_node')

        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.calibrate_cmd_sub = self.create_subscription(
            String,
            '/calibration/command',
            self.calibrate_command_callback,
            10
        )

        # Create publishers
        self.calibrated_camera_info_pub = self.create_publisher(
            CameraInfo,
            '/camera/rgb/camera_info_calibrated',
            10
        )
        self.calibrated_laser_pub = self.create_publisher(
            LaserScan,
            '/scan_calibrated',
            10
        )
        self.calibration_status_pub = self.create_publisher(
            String,
            '/calibration/status',
            10
        )
        self.adaptation_params_pub = self.create_publisher(
            String,
            '/adaptation/parameters',
            10
        )

        # Initialize CV bridge and TF broadcaster
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        # Calibration parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.laser_calibration = {'offset': 0.0, 'scale': 1.0}
        self.imu_calibration = {'bias': np.array([0.0, 0.0, 0.0])}
        self.joint_calibration = {}

        # Calibration state
        self.calibration_in_progress = False
        self.calibration_data = {
            'chessboard_corners': [],
            'laser_measurements': [],
            'imu_readings': []
        }

        # Adaptation parameters
        self.adaptation_params = {
            'sensor_fusion_weights': {'camera': 0.6, 'laser': 0.4},
            'control_gains': {'linear': 1.0, 'angular': 1.0},
            'safety_margins': {'distance': 0.5, 'velocity': 0.3}
        }

        # Create timer for continuous adaptation
        self.adaptation_timer = self.create_timer(2.0, self.update_adaptation)

        self.get_logger().info('Calibration and Adaptation Node Started')

    def rgb_callback(self, msg):
        """Process RGB image for calibration"""
        if not self.calibration_in_progress:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Look for calibration pattern (chessboard)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, (9, 6), None,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )

                self.calibration_data['chessboard_corners'].append(corners_refined)

                # If we have enough data, perform calibration
                if len(self.calibration_data['chessboard_corners']) >= 10:
                    self.perform_camera_calibration()

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image for calibration: {str(e)}')

    def camera_info_callback(self, msg):
        """Process camera info"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)

    def laser_callback(self, msg):
        """Process laser scan for calibration"""
        if self.calibration_in_progress:
            self.calibration_data['laser_measurements'].append(msg)

    def imu_callback(self, msg):
        """Process IMU data for calibration"""
        if self.calibration_in_progress:
            imu_reading = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            self.calibration_data['imu_readings'].append(imu_reading)

    def joint_state_callback(self, msg):
        """Process joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                if name not in self.joint_calibration:
                    self.joint_calibration[name] = {'offset': 0.0, 'scale': 1.0}

    def calibrate_command_callback(self, msg):
        """Handle calibration commands"""
        try:
            command_data = json.loads(msg.data)
            command = command_data.get('command', '')

            if command == 'start_camera_calibration':
                self.start_camera_calibration()
            elif command == 'start_laser_calibration':
                self.start_laser_calibration()
            elif command == 'start_imu_calibration':
                self.start_imu_calibration()
            elif command == 'apply_calibration':
                self.apply_calibration()
            elif command == 'reset_calibration':
                self.reset_calibration()

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in calibration command')

    def start_camera_calibration(self):
        """Start camera calibration process"""
        self.calibration_in_progress = True
        self.calibration_data['chessboard_corners'] = []
        self.get_logger().info('Starting camera calibration - present chessboard pattern to camera')

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'camera_calibration_started',
            'instructions': 'Present chessboard pattern to camera'
        })
        self.calibration_status_pub.publish(status_msg)

    def start_laser_calibration(self):
        """Start laser calibration process"""
        self.calibration_in_progress = True
        self.calibration_data['laser_measurements'] = []
        self.get_logger().info('Starting laser calibration')

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'laser_calibration_started',
            'instructions': 'Move robot to known positions for laser calibration'
        })
        self.calibration_status_pub.publish(status_msg)

    def start_imu_calibration(self):
        """Start IMU calibration process"""
        self.calibration_in_progress = True
        self.calibration_data['imu_readings'] = []
        self.get_logger().info('Starting IMU calibration - keep robot stationary')

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'imu_calibration_started',
            'instructions': 'Keep robot stationary for IMU calibration'
        })
        self.calibration_status_pub.publish(status_msg)

    def perform_camera_calibration(self):
        """Perform camera calibration using collected data"""
        if len(self.calibration_data['chessboard_corners']) < 10:
            return

        # Prepare object points (3D points in real world space)
        objp = np.zeros((9*6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all images
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane

        # Filter valid corner detections
        for corners in self.calibration_data['chessboard_corners']:
            if corners is not None:
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) < 10:
            self.get_logger().warn('Not enough valid corner detections for calibration')
            return

        # Perform calibration
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, (640, 480), None, None
            )

            if ret:
                self.camera_matrix = camera_matrix
                self.distortion_coeffs = dist_coeffs.flatten()

                # Publish calibrated camera info
                calibrated_info = self.create_calibrated_camera_info()
                self.calibrated_camera_info_pub.publish(calibrated_info)

                self.get_logger().info(f'Camera calibration completed - Reprojection error: {ret:.4f}')

                # Publish status
                status_msg = String()
                status_msg.data = json.dumps({
                    'status': 'camera_calibration_completed',
                    'reprojection_error': ret,
                    'valid_points': len(objpoints)
                })
                self.calibration_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Camera calibration failed: {str(e)}')

    def create_calibrated_camera_info(self):
        """Create calibrated camera info message"""
        from sensor_msgs.msg import CameraInfo

        calibrated_info = CameraInfo()
        calibrated_info.header.frame_id = 'camera_rgb_optical_frame'
        calibrated_info.height = 480
        calibrated_info.width = 640

        if self.camera_matrix is not None:
            calibrated_info.k = self.camera_matrix.flatten().tolist()

        if self.distortion_coeffs is not None:
            calibrated_info.d = self.distortion_coeffs.tolist()

        calibrated_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        calibrated_info.p = [self.camera_matrix[0,0], 0.0, self.camera_matrix[0,2], 0.0,
                            0.0, self.camera_matrix[1,1], self.camera_matrix[1,2], 0.0,
                            0.0, 0.0, 1.0, 0.0]

        calibrated_info.distortion_model = 'plumb_bob'
        return calibrated_info

    def apply_calibration(self):
        """Apply all calibrations"""
        self.calibration_in_progress = False

        # Update adaptation parameters based on calibration results
        self.adaptation_params['sensor_fusion_weights']['camera'] = 0.7  # Increased after calibration
        self.adaptation_params['sensor_fusion_weights']['laser'] = 0.3

        # Publish adaptation parameters
        params_msg = String()
        params_msg.data = json.dumps(self.adaptation_params, indent=2)
        self.adaptation_params_pub.publish(params_msg)

        self.get_logger().info('Calibration applied and adaptation parameters updated')

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'calibration_applied',
            'camera_matrix': self.camera_matrix.flatten().tolist() if self.camera_matrix is not None else None,
            'distortion_coeffs': self.distortion_coeffs.tolist() if self.distortion_coeffs is not None else None
        })
        self.calibration_status_pub.publish(status_msg)

    def reset_calibration(self):
        """Reset all calibration data"""
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.laser_calibration = {'offset': 0.0, 'scale': 1.0}
        self.imu_calibration = {'bias': np.array([0.0, 0.0, 0.0])}
        self.joint_calibration = {}

        self.calibration_data = {
            'chessboard_corners': [],
            'laser_measurements': [],
            'imu_readings': []
        }

        self.get_logger().info('Calibration data reset')

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({'status': 'calibration_reset'})
        self.calibration_status_pub.publish(status_msg)

    def update_adaptation(self):
        """Update adaptation parameters based on system performance"""
        # In a real system, this would analyze performance metrics and adjust parameters
        # For now, we'll just log the current adaptation parameters

        # Publish current adaptation parameters
        params_msg = String()
        params_msg.data = json.dumps(self.adaptation_params, indent=2)
        self.adaptation_params_pub.publish(params_msg)

        self.get_logger().info(f'Adaptation parameters updated - Camera weight: {self.adaptation_params["sensor_fusion_weights"]["camera"]}')

def main(args=None):
    rclpy.init(args=args)
    cal_node = CalibrationAdaptationNode()

    try:
        rclpy.spin(cal_node)
    except KeyboardInterrupt:
        pass
    finally:
        cal_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sim-to-Real Configuration

Create a configuration file for sim-to-real transfer:

```yaml
# sim_to_real_config.yaml
sim_to_real:
  # Domain randomization parameters
  domain_randomization:
    enabled: true
    frequency: 1.0  # Hz
    lighting:
      brightness_range: [0.5, 1.5]
      contrast_range: [0.8, 1.2]
      color_temperature_range: [3000, 8000]  # Kelvin
    textures:
      roughness_range: [0.0, 1.0]
      specular_range: [0.0, 1.0]
    dynamics:
      friction_range: [0.1, 0.9]
      restitution_range: [0.0, 0.5]
    sensors:
      noise_level_range: [0.0, 0.1]
      bias_range: [-0.05, 0.05]
      drift_range: [-0.01, 0.01]

  # Reality gap compensation
  reality_gap:
    compensation_enabled: true
    adaptation_rate: 0.1
    visual_threshold: 0.1
    sensor_threshold: 0.05
    max_history: 100

  # Calibration parameters
  calibration:
    camera_calibration:
      pattern_size: [9, 6]  # Chessboard pattern
      square_size: 0.025  # meters
      min_points: 10
    laser_calibration:
      enabled: true
      reference_distance: 1.0  # meters
      accuracy_threshold: 0.01  # meters
    imu_calibration:
      enabled: true
      stationary_duration: 5.0  # seconds
      bias_threshold: 0.1

  # Adaptation parameters
  adaptation:
    sensor_fusion_weights:
      camera: 0.6
      laser: 0.4
      imu: 0.2
    control_gains:
      linear: 1.0
      angular: 1.0
    safety_margins:
      distance: 0.5  # meters
      velocity: 0.3  # m/s

  # Performance parameters
  performance:
    max_processing_time: 0.1  # seconds per frame
    target_frequency: 30.0  # Hz for image processing
    memory_limit: 2048  # MB
    gpu_enabled: true

  # Safety parameters
  safety:
    enabled: true
    reality_gap_threshold: 0.5  # Stop if gap exceeds this
    emergency_calibration: true
    fallback_behavior: "safe_stop"
```

## Sim-to-Real Launch File

Create a launch file for the sim-to-real transfer system:

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
    config_file = LaunchConfiguration('config_file', default='sim_to_real_config.yaml')

    # Domain randomization node
    domain_randomization_node = Node(
        package='physical_ai_sim_to_real',
        executable='domain_randomization_node',
        name='domain_randomization_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_sim_to_real'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    # Reality gap compensation node
    reality_gap_compensation_node = Node(
        package='physical_ai_sim_to_real',
        executable='reality_gap_compensation_node',
        name='reality_gap_compensation_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_sim_to_real'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    # Calibration and adaptation node
    calibration_adaptation_node = Node(
        package='physical_ai_sim_to_real',
        executable='calibration_adaptation_node',
        name='calibration_adaptation_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_sim_to_real'),
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
            default_value='sim_to_real_config.yaml',
            description='Configuration file for sim-to-real transfer'
        ),
        domain_randomization_node,
        reality_gap_compensation_node,
        calibration_adaptation_node
    ])
```

## Quality Assurance for Sim-to-Real Transfer

### Performance Metrics
- **Transfer Success Rate**: Percentage of simulation-trained behaviors that work in reality
- **Reality Gap Size**: Quantified difference between simulation and reality
- **Adaptation Speed**: How quickly the system adapts to real-world conditions
- **Calibration Accuracy**: Precision of sensor and system calibration

### Safety Considerations
Following our constitution's "Safety is Intelligence" principle:

1. **Safe Transfer**: Ensure safe operation during sim-to-real transition
2. **Fallback Mechanisms**: Maintain safe behaviors when transfer fails
3. **Continuous Monitoring**: Monitor for reality gap during operation
4. **Calibration Validation**: Verify calibration before deployment

### Testing Scenarios
1. **Domain Randomization**: Test system with various simulated conditions
2. **Reality Gap Analysis**: Measure and analyze differences between sim and real
3. **Calibration Procedures**: Test sensor and system calibration processes
4. **Adaptation Performance**: Test system adaptation in real environments

## Looking Forward

With our sim-to-real transfer capabilities established, the next chapter will focus on creating the final demonstration of our autonomous robot completing the complete challenge.

[Continue to Chapter 18: Final Demo Mission](./chapter-18-final-demo.md)