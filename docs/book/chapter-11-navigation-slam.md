# Chapter 11: Navigation & SLAM - Teaching Robots to Find Their Way

## The Foundation of Autonomous Movement

In this chapter, we implement navigation and SLAM (Simultaneous Localization and Mapping) systems that enable our robot to autonomously navigate through unknown environments. This fulfills our project's functional requirement **FR-004**: System MUST perform autonomous indoor navigation with ≥95% collision-free success rate, and **FR-005**: System MUST support SLAM-based mapping and localization for unknown environments.

## Understanding Navigation and SLAM

SLAM is a fundamental capability that allows robots to build a map of an unknown environment while simultaneously keeping track of their location within that map. This is crucial for autonomous navigation and aligns with our constitution's principle that "Simulation is Truth, Reality is the Test."

### Key Tasks from Our Plan:
- T047: Create navigation and SLAM system in src/navigation/chapter_nav.py

## SLAM Implementation Node

Let's create a comprehensive SLAM node that handles mapping and localization:

```python
#!/usr/bin/env python3
"""
SLAM Node for Physical AI System
Implements Simultaneous Localization and Mapping
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PointStamped
from std_msgs.msg import String, Float64
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_dilation, binary_erosion
import cv2
from collections import deque
import json

class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        # Create subscribers for sensor data
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.imu_sub = self.create_subscription(
            PointCloud2,
            '/point_cloud',
            self.pointcloud_callback,
            10
        )

        # Create publishers for SLAM results
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            10
        )
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            10
        )
        self.localization_status_pub = self.create_publisher(
            String,
            '/localization_status',
            10
        )

        # TF listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # SLAM parameters
        self.map_resolution = 0.05  # meters per cell
        self.map_width = 400  # cells (20m x 20m map)
        self.map_height = 400  # cells
        self.map_origin_x = -10.0  # meters
        self.map_origin_y = -10.0  # meters

        # Initialize occupancy grid
        self.occupancy_grid = np.full((self.map_height, self.map_width), -1, dtype=np.int8)  # -1 = unknown
        self.occupancy_grid_updated = False

        # Robot state
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.robot_covariance = np.eye(3) * 0.1  # Initial uncertainty

        # SLAM state
        self.is_localized = False
        self.localization_confidence = 0.0
        self.map_building_enabled = True

        # Sensor data buffers
        self.laser_buffer = deque(maxlen=10)
        self.odom_buffer = deque(maxlen=10)

        # Mapping parameters
        self.laser_max_range = 10.0  # meters
        self.occupancy_threshold = 0.6  # probability threshold for occupied cells
        self.free_space_threshold = 0.3  # probability threshold for free space

        # Particle filter parameters for localization
        self.num_particles = 100
        self.particles = self.initialize_particles()
        self.particle_weights = np.ones(self.num_particles) / self.num_particles

        self.get_logger().info('SLAM Node Started')

    def initialize_particles(self):
        """Initialize particles for Monte Carlo localization"""
        particles = np.zeros((self.num_particles, 3))  # x, y, theta
        # Initially spread particles across the map
        for i in range(self.num_particles):
            particles[i, 0] = np.random.uniform(-5, 5)  # x
            particles[i, 1] = np.random.uniform(-5, 5)  # y
            particles[i, 2] = np.random.uniform(-np.pi, np.pi)  # theta
        return particles

    def laser_callback(self, msg):
        """Process laser scan data for mapping and localization"""
        # Convert laser scan to world coordinates
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges
        valid_indices = (ranges > msg.range_min) & (ranges < msg.range_max) & (~np.isnan(ranges))
        valid_angles = angles[valid_indices]
        valid_ranges = ranges[valid_indices]

        # Convert to Cartesian coordinates in laser frame
        laser_x = valid_ranges * np.cos(valid_angles)
        laser_y = valid_ranges * np.sin(valid_angles)

        # Transform to world frame using current robot pose
        cos_theta = np.cos(self.robot_pose[2])
        sin_theta = np.sin(self.robot_pose[2])

        world_x = self.robot_pose[0] + cos_theta * laser_x - sin_theta * laser_y
        world_y = self.robot_pose[1] + sin_theta * laser_x + cos_theta * laser_y

        # Update occupancy grid
        if self.map_building_enabled:
            self.update_occupancy_grid(world_x, world_y, self.robot_pose)

        # Update particle filter for localization
        self.update_particle_filter(msg, world_x, world_y)

        # Publish map if updated
        if self.occupancy_grid_updated:
            self.publish_map()
            self.occupancy_grid_updated = False

    def odom_callback(self, msg):
        """Process odometry data for motion prediction"""
        # Extract pose from odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Extract orientation
        orientation = msg.pose.pose.orientation
        r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        theta = r.as_euler('xyz')[2]  # yaw angle

        # Update robot pose
        self.robot_pose = np.array([x, y, theta])

        # Add to buffer
        self.odom_buffer.append({
            'timestamp': msg.header.stamp,
            'pose': self.robot_pose.copy()
        })

    def pointcloud_callback(self, msg):
        """Process point cloud data for 3D mapping"""
        # In a real system, this would integrate 3D point cloud data into the map
        # For now, we'll just log the event
        self.get_logger().info('Received point cloud data for 3D mapping')

    def update_occupancy_grid(self, laser_x, laser_y, robot_pose):
        """Update occupancy grid based on laser scan"""
        # Convert world coordinates to grid coordinates
        grid_x = np.floor((laser_x - self.map_origin_x) / self.map_resolution).astype(int)
        grid_y = np.floor((laser_y - self.map_origin_y) / self.map_resolution).astype(int)

        # Filter points within map bounds
        valid_points = (grid_x >= 0) & (grid_x < self.map_width) & \
                      (grid_y >= 0) & (grid_y < self.map_height)

        grid_x = grid_x[valid_points]
        grid_y = grid_y[valid_points]

        if len(grid_x) == 0:
            return

        # Mark laser endpoints as occupied
        self.occupancy_grid[grid_y, grid_x] = 100  # occupied

        # Mark free space along laser rays using ray tracing
        robot_grid_x = int((robot_pose[0] - self.map_origin_x) / self.map_resolution)
        robot_grid_y = int((robot_pose[1] - self.map_origin_y) / self.map_resolution)

        # Ensure robot position is within bounds
        if 0 <= robot_grid_x < self.map_width and 0 <= robot_grid_y < self.map_height:
            for i in range(len(grid_x)):
                # Simple Bresenham line algorithm to mark free space
                self.mark_free_space(robot_grid_x, robot_grid_y, grid_x[i], grid_y[i])

        self.occupancy_grid_updated = True

    def mark_free_space(self, x0, y0, x1, y1):
        """Mark free space along a line using Bresenham's algorithm"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            # Check bounds
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                # Only mark as free if not already occupied
                if self.occupancy_grid[y, x] != 100:
                    self.occupancy_grid[y, x] = 0  # free space

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def update_particle_filter(self, laser_msg, laser_x, laser_y):
        """Update particle filter for Monte Carlo localization"""
        # Predict step: move particles based on odometry
        self.predict_particles()

        # Update step: weight particles based on sensor likelihood
        self.update_particle_weights(laser_msg, laser_x, laser_y)

        # Resample particles
        self.resample_particles()

        # Estimate pose from particles
        self.estimate_pose_from_particles()

    def predict_particles(self):
        """Predict particle motion based on odometry"""
        # In a real system, this would use odometry to predict particle motion
        # For simplicity, we'll add small random noise to simulate motion
        noise_scale = 0.05  # meters
        angle_noise_scale = 0.01  # radians

        self.particles[:, 0] += np.random.normal(0, noise_scale, self.num_particles)
        self.particles[:, 1] += np.random.normal(0, noise_scale, self.num_particles)
        self.particles[:, 2] += np.random.normal(0, angle_noise_scale, self.num_particles)

    def update_particle_weights(self, laser_msg, laser_x, laser_y):
        """Update particle weights based on sensor likelihood"""
        # For each particle, calculate likelihood of laser readings
        for i in range(self.num_particles):
            particle_pose = self.particles[i, :]
            likelihood = self.calculate_laser_likelihood(particle_pose, laser_x, laser_y)
            self.particle_weights[i] *= likelihood

        # Normalize weights
        total_weight = np.sum(self.particle_weights)
        if total_weight > 0:
            self.particle_weights /= total_weight
        else:
            # Reset weights if they become too small
            self.particle_weights.fill(1.0 / self.num_particles)

    def calculate_laser_likelihood(self, particle_pose, laser_x, laser_y):
        """Calculate likelihood of laser readings given particle pose"""
        # Transform laser points to particle's frame of reference
        cos_theta = np.cos(-particle_pose[2])
        sin_theta = np.sin(-particle_pose[2])

        dx = laser_x - particle_pose[0]
        dy = laser_y - particle_pose[1]

        transformed_x = cos_theta * dx - sin_theta * dy
        transformed_y = sin_theta * dx + cos_theta * dy

        # Calculate likelihood based on map occupancy
        likelihood = 1.0
        for x, y in zip(transformed_x, transformed_y):
            grid_x = int((x - self.map_origin_x) / self.map_resolution)
            grid_y = int((y - self.map_origin_y) / self.map_resolution)

            if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
                if self.occupancy_grid[grid_y, grid_x] == 100:  # occupied
                    likelihood *= 0.9  # high likelihood for occupied cells
                elif self.occupancy_grid[grid_y, grid_x] == 0:  # free
                    likelihood *= 0.7  # medium likelihood for free space
                else:  # unknown
                    likelihood *= 0.5  # low likelihood for unknown

        return max(likelihood, 1e-6)  # prevent zero likelihood

    def resample_particles(self):
        """Resample particles based on weights"""
        # Systematic resampling
        indices = self.systematic_resample(self.particle_weights)
        self.particles = self.particles[indices]
        self.particle_weights.fill(1.0 / self.num_particles)

    def systematic_resample(self, weights):
        """Systematic resampling algorithm"""
        N = len(weights)
        indices = np.zeros(N, dtype=int)

        # Generate random start point
        random_start = np.random.random() / N

        # Generate sampling points
        points = (np.arange(N) + random_start) / N

        # Find corresponding particles
        cumsum = np.cumsum(weights)
        i, j = 0, 0
        while i < N and j < N:
            if points[i] <= cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        return indices

    def estimate_pose_from_particles(self):
        """Estimate robot pose from particles"""
        # Calculate weighted mean of particles
        mean_pose = np.average(self.particles, axis=0, weights=self.particle_weights)

        # Calculate covariance
        diff = self.particles - mean_pose
        covariance = np.cov(diff.T, aweights=self.particle_weights)

        # Update robot pose estimate
        self.robot_pose = mean_pose
        self.robot_covariance = covariance

        # Calculate localization confidence
        self.localization_confidence = np.mean(self.particle_weights)
        self.is_localized = self.localization_confidence > 0.5

        # Publish localization status
        status_msg = String()
        status_msg.data = json.dumps({
            'is_localized': self.is_localized,
            'confidence': float(self.localization_confidence),
            'particle_count': self.num_particles
        })
        self.localization_status_pub.publish(status_msg)

    def publish_map(self):
        """Publish the occupancy grid map"""
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'map'

        # Set map metadata
        map_msg.info.resolution = float(self.map_resolution)
        map_msg.info.width = int(self.map_width)
        map_msg.info.height = int(self.map_height)
        map_msg.info.origin.position.x = float(self.map_origin_x)
        map_msg.info.origin.position.y = float(self.map_origin_y)
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # Flatten grid for message
        flat_grid = self.occupancy_grid.flatten()
        map_msg.data = flat_grid.astype(np.int8).tolist()

        self.map_pub.publish(map_msg)

def main(args=None):
    rclpy.init(args=args)
    slam_node = SLAMNode()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Navigation Node

Now let's create a navigation node that uses the SLAM map for path planning:

```python
#!/usr/bin/env python3
"""
Navigation Node for Physical AI System
Implements path planning and navigation using SLAM map
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseArray, Twist
from std_msgs.msg import String, Bool
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.spatial import distance
import heapq
import json

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Create subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )
        self.cancel_sub = self.create_subscription(
            Bool,
            '/move_base/cancel',
            self.cancel_callback,
            10
        )

        # Create publishers
        self.path_pub = self.create_publisher(
            Path,
            '/plan',
            10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.status_pub = self.create_publisher(
            String,
            '/navigation_status',
            10
        )

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation parameters
        self.map_resolution = 0.05  # meters per cell
        self.robot_radius = 0.3  # meters (for collision checking)
        self.inflation_radius = 0.5  # meters (for path planning)
        self.planning_frequency = 5.0  # Hz
        self.control_frequency = 10.0  # Hz

        # Map data
        self.occupancy_grid = None
        self.map_width = 0
        self.map_height = 0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_inflated = None

        # Navigation state
        self.current_goal = None
        self.current_path = []
        self.path_index = 0
        self.is_navigating = False
        self.navigation_cancelled = False

        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        # Create timers
        self.planning_timer = self.create_timer(1.0/self.planning_frequency, self.plan_path)
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.execute_navigation)

        self.get_logger().info('Navigation Node Started')

    def map_callback(self, msg):
        """Process occupancy grid map"""
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        # Convert map data to numpy array
        self.occupancy_grid = np.array(msg.data).reshape((self.map_height, self.map_width)).astype(np.int8)

        # Inflate obstacles
        self.map_inflated = self.inflate_obstacles(self.occupancy_grid)

        self.get_logger().info(f'Map received: {self.map_width}x{self.map_height}, resolution: {self.map_resolution}')

    def inflate_obstacles(self, grid):
        """Inflate obstacles in the map to account for robot size"""
        inflated = grid.copy()

        # Create inflation mask based on robot radius
        inflation_cells = int(self.inflation_radius / self.map_resolution)

        for i in range(self.map_height):
            for j in range(self.map_width):
                if grid[i, j] > 50:  # Occupied cell
                    # Inflate around this cell
                    for di in range(-inflation_cells, inflation_cells + 1):
                        for dj in range(-inflation_cells, inflation_cells + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.map_height and 0 <= nj < self.map_width:
                                dist = np.sqrt(di**2 + dj**2) * self.map_resolution
                                if dist <= self.inflation_radius:
                                    inflated[ni, nj] = 100  # Mark as occupied

        return inflated

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.current_goal = msg
        self.navigation_cancelled = False

        # Extract goal position
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        self.get_logger().info(f'New navigation goal: ({goal_x:.2f}, {goal_y:.2f})')

        # Plan path immediately
        self.plan_path()

    def cancel_callback(self, msg):
        """Cancel current navigation"""
        if msg.data:
            self.navigation_cancelled = True
            self.is_navigating = False
            self.current_goal = None
            self.current_path = []
            self.path_index = 0

            # Stop the robot
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)

            self.get_logger().info('Navigation cancelled')

    def plan_path(self):
        """Plan path using A* algorithm"""
        if self.map_inflated is None or self.current_goal is None:
            return

        # Get robot position (in a real system, this would come from localization)
        robot_grid_x = int((self.robot_x - self.map_origin_x) / self.map_resolution)
        robot_grid_y = int((self.robot_y - self.map_origin_y) / self.map_resolution)

        # Get goal position
        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y
        goal_grid_x = int((goal_x - self.map_origin_x) / self.map_resolution)
        goal_grid_y = int((goal_y - self.map_origin_y) / self.map_resolution)

        # Check bounds
        if not (0 <= robot_grid_x < self.map_width and 0 <= robot_grid_y < self.map_height and
                0 <= goal_grid_x < self.map_width and 0 <= goal_grid_y < self.map_height):
            self.get_logger().error('Robot or goal position outside map bounds')
            return

        # Check if start or goal is occupied
        if self.map_inflated[robot_grid_y, robot_grid_x] > 50:
            self.get_logger().error('Robot position is in occupied space')
            return

        if self.map_inflated[goal_grid_y, goal_grid_x] > 50:
            self.get_logger().error('Goal position is in occupied space')
            return

        # Run A* path planning
        path = self.a_star_planning(robot_grid_x, robot_grid_y, goal_grid_x, goal_grid_y)

        if path:
            # Convert grid path to world coordinates
            world_path = []
            for grid_x, grid_y in path:
                world_x = grid_x * self.map_resolution + self.map_origin_x
                world_y = grid_y * self.map_resolution + self.map_origin_y

                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = 'map'
                pose.pose.position.x = float(world_x)
                pose.pose.position.y = float(world_y)
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0

                world_path.append(pose)

            self.current_path = world_path
            self.path_index = 0
            self.is_navigating = True

            # Publish path
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'
            path_msg.poses = self.current_path

            self.path_pub.publish(path_msg)

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'path_planned',
                'path_length': len(self.current_path),
                'goal': [goal_x, goal_y]
            })
            self.status_pub.publish(status_msg)

            self.get_logger().info(f'Path planned with {len(self.current_path)} waypoints')
        else:
            self.get_logger().error('Failed to find path to goal')
            self.is_navigating = False

    def a_star_planning(self, start_x, start_y, goal_x, goal_y):
        """A* path planning algorithm"""
        # Define possible movements (8-connected)
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        movement_costs = [
            np.sqrt(2), 1, np.sqrt(2),
            1,          1,
            np.sqrt(2), 1, np.sqrt(2)
        ]

        # Initialize open set with start position
        open_set = [(0, start_x, start_y)]
        heapq.heapify(open_set)

        # Initialize g_score (cost from start)
        g_score = np.full((self.map_height, self.map_width), np.inf)
        g_score[start_y, start_x] = 0

        # Initialize f_score (g_score + heuristic)
        f_score = np.full((self.map_height, self.map_width), np.inf)
        f_score[start_y, start_x] = self.heuristic(start_x, start_y, goal_x, goal_y)

        # Initialize came_from for path reconstruction
        came_from = {}

        while open_set:
            current_f, current_x, current_y = heapq.heappop(open_set)

            # Check if we reached the goal
            if current_x == goal_x and current_y == goal_y:
                # Reconstruct path
                path = []
                current = (current_x, current_y)
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append((start_x, start_y))
                path.reverse()
                return path

            # Skip if we've already found a better path to this node
            if current_f > f_score[current_y, current_x]:
                continue

            # Explore neighbors
            for i, (dx, dy) in enumerate(movements):
                neighbor_x, neighbor_y = current_x + dx, current_y + dy

                # Check bounds
                if not (0 <= neighbor_x < self.map_width and 0 <= neighbor_y < self.map_height):
                    continue

                # Check if neighbor is occupied
                if self.map_inflated[neighbor_y, neighbor_x] > 50:
                    continue

                # Calculate tentative g_score
                tentative_g = g_score[current_y, current_x] + movement_costs[i]

                # Check if this path to neighbor is better
                if tentative_g < g_score[neighbor_y, neighbor_x]:
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_score[neighbor_y, neighbor_x] = tentative_g
                    f_score[neighbor_y, neighbor_x] = tentative_g + self.heuristic(neighbor_x, neighbor_y, goal_x, goal_y)
                    heapq.heappush(open_set, (f_score[neighbor_y, neighbor_x], neighbor_x, neighbor_y))

        # No path found
        return None

    def heuristic(self, x1, y1, x2, y2):
        """Calculate heuristic distance (Euclidean)"""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def execute_navigation(self):
        """Execute navigation along planned path"""
        if not self.is_navigating or self.navigation_cancelled:
            return

        if not self.current_path or self.path_index >= len(self.current_path):
            # Reached goal or path is empty
            self.is_navigating = False
            self.current_path = []
            self.path_index = 0

            # Stop the robot
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'goal_reached' if self.current_path else 'no_path',
                'completed': True
            })
            self.status_pub.publish(status_msg)

            self.get_logger().info('Navigation completed')
            return

        # Get current target waypoint
        target_pose = self.current_path[self.path_index]
        target_x = target_pose.pose.position.x
        target_y = target_pose.pose.position.y

        # Calculate distance to target
        dist_to_target = np.sqrt((target_x - self.robot_x)**2 + (target_y - self.robot_y)**2)

        # Check if we've reached the current waypoint
        if dist_to_target < 0.2:  # 20cm threshold
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                # Reached the end of the path
                self.is_navigating = False
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                return

        # Calculate control command to reach target
        cmd_vel = self.calculate_control_command(target_x, target_y)

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'status': 'navigating',
            'current_waypoint': self.path_index,
            'total_waypoints': len(self.current_path),
            'distance_to_goal': dist_to_target
        })
        self.status_pub.publish(status_msg)

    def calculate_control_command(self, target_x, target_y):
        """Calculate velocity command to reach target position"""
        cmd = Twist()

        # Calculate error
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate desired angle
        desired_angle = np.arctan2(dy, dx)
        angle_error = desired_angle - self.robot_theta

        # Normalize angle error to [-π, π]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        # Simple proportional controller
        linear_kp = 0.5
        angular_kp = 1.0

        # Set velocities (with limits)
        cmd.linear.x = min(linear_kp * distance, 0.5)  # Max 0.5 m/s
        cmd.angular.z = angular_kp * angle_error  # Limit handled by robot

        # Ensure robot doesn't move too fast when close to target
        if distance < 0.3:
            cmd.linear.x *= 0.5  # Slow down when close

        return cmd

def main(args=None):
    rclpy.init(args=args)
    nav_node = NavigationNode()

    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot on shutdown
        stop_cmd = Twist()
        nav_node.cmd_vel_pub.publish(stop_cmd)
        nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Navigation Configuration

Create a configuration file for navigation parameters:

```yaml
# navigation_config.yaml
navigation:
  # Global planner parameters
  global_planner:
    planner_frequency: 1.0  # Hz
    plan_resolution: 0.05  # meters
    visualize_planner: true

  # Local planner parameters
  local_planner:
    controller_frequency: 10.0  # Hz
    max_vel_x: 0.5  # m/s
    min_vel_x: 0.1  # m/s
    max_vel_theta: 1.0  # rad/s
    min_vel_theta: 0.1  # rad/s
    acc_lim_x: 2.5  # m/s²
    acc_lim_theta: 3.2  # rad/s²

  # Costmap parameters
  costmap:
    resolution: 0.05  # meters per cell
    inflation_radius: 0.55  # meters
    obstacle_range: 2.5  # meters
    raytrace_range: 3.0  # meters
    transform_tolerance: 0.2  # seconds

  # Recovery behaviors
  recovery:
    enabled: true
    rotate_recovery: true
    rotate_recovery_angle: 1.57  # radians (90 degrees)
    clear_costmap_recovery: true
    oscillation_timeout: 10.0  # seconds
    oscillation_distance: 0.5  # meters

  # Goal tolerance
  goal_tolerance:
    xy_goal_tolerance: 0.2  # meters
    yaw_goal_tolerance: 0.1  # radians
    latch_xy_goal_tolerance: false

  # Obstacle avoidance
  obstacle_avoidance:
    enabled: true
    min_obstacle_dist: 0.3  # meters
    max_obstacle_dist: 1.0  # meters
    obstacle_cost: 254  # cost value for obstacles
    inflation_cost: 128  # cost value for inflated area

  # SLAM integration
  slam_integration:
    map_topic: "/map"
    localization_topic: "/amcl_pose"
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_frame: "odom"
```

## Navigation Launch File

Create a launch file for the navigation system:

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
    config_file = LaunchConfiguration('config_file', default='navigation_config.yaml')

    # SLAM node
    slam_node = Node(
        package='physical_ai_navigation',
        executable='slam_node',
        name='slam_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_navigation'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    # Navigation node
    navigation_node = Node(
        package='physical_ai_navigation',
        executable='navigation_node',
        name='navigation_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_navigation'),
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
            default_value='navigation_config.yaml',
            description='Configuration file for navigation'
        ),
        slam_node,
        navigation_node
    ])
```

## Quality Assurance for Navigation

### Performance Metrics
- **Success Rate**: Percentage of goals reached successfully
- **Path Efficiency**: Ratio of actual path length to optimal path length
- **Computation Time**: Time taken for path planning and execution
- **Collision Avoidance**: Percentage of collision-free navigation

### Safety Considerations
Following our constitution's "Safety is Intelligence" principle:

1. **Obstacle Detection**: Reliable detection and avoidance of obstacles
2. **Emergency Stops**: Immediate stop on safety-critical situations
3. **Speed Limiting**: Controlled speeds appropriate for environment
4. **Localization Accuracy**: Maintained localization confidence above threshold

### Testing Scenarios
1. **Simple Navigation**: Basic point-to-point navigation
2. **Dynamic Obstacles**: Navigation around moving obstacles
3. **Narrow Passages**: Navigation through confined spaces
4. **Recovery Behaviors**: Handling of navigation failures

## Looking Forward

With our navigation and SLAM systems in place, the next chapter will focus on reinforcement learning and teaching our robot to "learn to move" through various locomotion strategies.

[Continue to Chapter 12: Learning to Move](./chapter-12-learning-to-move.md)