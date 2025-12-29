# Chapter 5: First Motion Node - Bringing the Robot to Life

## From Communication to Action

In this chapter, we transition from the communication infrastructure we built in Chapter 4 to actual robot motion. This is where our "nervous system" begins to control the "body" of our robot, fulfilling the "Embodiment First" principle from our constitution.

## The Motion Control Architecture

Based on our project plan, we'll implement a motion control node that can receive commands and execute basic movements. This node will be part of the navigation system and will interface with the robot's actuators.

### Key Tasks from Our Plan:
- T041: Create first moving robot Python node in src/ros_nodes/motion_node.py

## Basic Motion Node Implementation

Let's create a simple motion control node that can move our robot forward, backward, left, and right:

```python
#!/usr/bin/env python3
"""
First Motion Node for Physical AI System
Implements basic movement commands for the robot
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import math

class MotionController(Node):
    def __init__(self):
        super().__init__('motion_controller')

        # Create publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create subscriber for motion commands
        self.command_sub = self.create_subscription(
            String,
            'motion_command',
            self.command_callback,
            10
        )

        # Create subscriber for safety sensors
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Emergency stop subscriber
        self.emergency_stop_sub = self.create_subscription(
            String,
            'emergency_stop',
            self.emergency_stop_callback,
            10
        )

        # Robot state
        self.is_safe_to_move = True
        self.emergency_stop_activated = False

        # Motion parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s

        self.get_logger().info('Motion Controller Node Started')

    def command_callback(self, msg):
        """Handle motion commands"""
        if self.emergency_stop_activated:
            self.get_logger().warn('Emergency stop active - ignoring motion commands')
            return

        if not self.is_safe_to_move:
            self.get_logger().warn('Safety check failed - not moving')
            return

        command = msg.data.lower().strip()

        twist = Twist()

        if command == 'forward':
            twist.linear.x = self.linear_speed
            self.get_logger().info('Moving forward')
        elif command == 'backward':
            twist.linear.x = -self.linear_speed
            self.get_logger().info('Moving backward')
        elif command == 'left':
            twist.angular.z = self.angular_speed
            self.get_logger().info('Turning left')
        elif command == 'right':
            twist.angular.z = -self.angular_speed
            self.get_logger().info('Turning right')
        elif command == 'stop':
            # Already zero values
            self.get_logger().info('Stopping')
        else:
            self.get_logger().warn(f'Unknown command: {command}')
            return

        self.cmd_vel_pub.publish(twist)

    def laser_callback(self, msg):
        """Check laser scan for obstacles"""
        # Check for obstacles in front of the robot (within 1 meter)
        min_distance = float('inf')
        front_scan_start = len(msg.ranges) // 2 - 10
        front_scan_end = len(msg.ranges) // 2 + 10

        for i in range(front_scan_start, front_scan_end):
            if 0 < msg.ranges[i] < min_distance:
                min_distance = msg.ranges[i]

        # Update safety status based on obstacle detection
        self.is_safe_to_move = min_distance > 0.5  # Safe if obstacle > 0.5m away

        if not self.is_safe_to_move:
            self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m - movement restricted')

    def emergency_stop_callback(self, msg):
        """Handle emergency stop commands"""
        if msg.data.lower() == 'activate':
            self.emergency_stop_activated = True
            # Stop the robot immediately
            stop_twist = Twist()
            self.cmd_vel_pub.publish(stop_twist)
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
        elif msg.data.lower() == 'deactivate':
            self.emergency_stop_activated = False
            self.get_logger().info('Emergency stop deactivated')

def main(args=None):
    rclpy.init(args=args)
    motion_controller = MotionController()

    try:
        rclpy.spin(motion_controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot before shutting down
        stop_twist = Twist()
        motion_controller.cmd_vel_pub.publish(stop_twist)
        motion_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Motion Planning Node

For more sophisticated movement, let's also implement a path following node:

```python
#!/usr/bin/env python3
"""
Path Following Node for Physical AI System
Implements more sophisticated motion control
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import math
import numpy as np
from collections import deque

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_sub = self.create_subscription(
            Path,
            'path_to_follow',
            self.path_callback,
            10
        )
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        self.emergency_stop_sub = self.create_subscription(
            String,
            'emergency_stop',
            self.emergency_stop_callback,
            10
        )

        # Robot state
        self.current_path = deque()
        self.current_goal = None
        self.is_moving = False
        self.emergency_stop_activated = False

        # Motion parameters
        self.linear_speed = 0.3
        self.angular_speed_limit = 0.5
        self.arrival_threshold = 0.2  # meters
        self.lookahead_distance = 0.5  # meters

        self.get_logger().info('Path Follower Node Started')

    def path_callback(self, msg):
        """Handle incoming path to follow"""
        if self.emergency_stop_activated:
            return

        # Clear current path and load new one
        self.current_path.clear()
        for pose in msg.poses:
            point = Point()
            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = pose.pose.position.z
            self.current_path.append(point)

        self.get_logger().info(f'Loaded path with {len(self.current_path)} waypoints')

    def laser_callback(self, msg):
        """Check for obstacles during navigation"""
        # Check for obstacles in the robot's path
        min_distance = min([r for r in msg.ranges if not math.isnan(r)], default=float('inf'))

        if min_distance < 0.3:  # Emergency stop distance
            self.emergency_stop_activated = True
            stop_twist = Twist()
            self.cmd_vel_pub.publish(stop_twist)
            self.get_logger().error('OBSTACLE TOO CLOSE - EMERGENCY STOP')

    def emergency_stop_callback(self, msg):
        """Handle emergency stop"""
        if msg.data.lower() == 'activate':
            self.emergency_stop_activated = True
            stop_twist = Twist()
            self.cmd_vel_pub.publish(stop_twist)
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
        elif msg.data.lower() == 'deactivate':
            self.emergency_stop_activated = False
            self.get_logger().info('Emergency stop deactivated')

    def follow_path(self):
        """Main path following logic"""
        if self.emergency_stop_activated or not self.current_path:
            return

        # Get current robot position (in a real system, this would come from localization)
        # For now, we'll use a simulated position
        robot_x = 0.0  # This would come from odometry/localization
        robot_y = 0.0

        # Get the next goal from the path
        if self.current_goal is None and self.current_path:
            self.current_goal = self.current_path.popleft()

        if self.current_goal:
            # Calculate direction to goal
            dx = self.current_goal.x - robot_x
            dy = self.current_goal.y - robot_y
            distance_to_goal = math.sqrt(dx*dx + dy*dy)

            if distance_to_goal < self.arrival_threshold:
                # Reached current goal, get next one
                if self.current_path:
                    self.current_goal = self.current_path.popleft()
                else:
                    # Path completed
                    self.current_goal = None
                    self.get_logger().info('Path completed')
                    return

            # Calculate required velocity
            twist = Twist()

            # Linear velocity - proportional to distance (slow down when close)
            twist.linear.x = min(self.linear_speed, distance_to_goal * 0.5)

            # Angular velocity - proportional to angle error
            desired_angle = math.atan2(dy, dx)
            # In a real system, this would come from odometry
            current_angle = 0.0  # This would be the robot's actual heading
            angle_error = desired_angle - current_angle

            # Normalize angle error to [-pi, pi]
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi

            twist.angular.z = max(-self.angular_speed_limit,
                                 min(self.angular_speed_limit, angle_error * 1.0))

            self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    path_follower = PathFollower()

    # Timer for path following logic
    timer = path_follower.create_timer(0.1, path_follower.follow_path)

    try:
        rclpy.spin(path_follower)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot before shutting down
        stop_twist = Twist()
        path_follower.cmd_vel_pub.publish(stop_twist)
        path_follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Safety Considerations

Following our constitution's principle that "Safety is Intelligence," our motion nodes include several safety features:

1. **Emergency Stop System**: Immediate halt on emergency stop command
2. **Obstacle Detection**: Automatic stopping when obstacles are detected
3. **Speed Limiting**: Controlled movement speeds to prevent accidents
4. **State Monitoring**: Continuous monitoring of robot state and environment

## Testing the Motion Nodes

To test our motion nodes:

```bash
# Start the motion controller
ros2 run my_robot_package motion_controller

# Send motion commands
ros2 topic pub /motion_command std_msgs/String "data: 'forward'"
ros2 topic pub /motion_command std_msgs/String "data: 'stop'"

# For path following
ros2 run my_robot_package path_follower
# Then publish a path to /path_to_follow
```

## Integration with Voice Commands

Eventually, these motion nodes will be integrated with our voice processing system. When a user says "move forward," the voice command will be processed by our ASR and NLU nodes, then converted to a motion command that our motion controller can execute.

## Looking Forward

Now that we've implemented basic motion capabilities, the next chapter will focus on creating our robot's skeleton using URDF (Unified Robot Description Format), giving our robot a physical form that matches its nervous system.

[Continue to Chapter 6: Robot Skeleton URDF](./chapter-6-robot-skeleton.md)