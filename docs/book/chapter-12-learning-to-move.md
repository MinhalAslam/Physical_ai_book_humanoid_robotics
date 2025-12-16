# Chapter 12: Learning to Move - Reinforcement Learning for Robot Locomotion

## Teaching Robots to Move Intelligently

In this chapter, we explore reinforcement learning (RL) techniques that enable our robot to learn complex locomotion patterns and movement strategies. This aligns with our constitution's principle of "Embodiment First" - intelligence gains meaning only when grounded in physical interaction. We'll implement systems that allow our robot to learn through trial and error in both simulation and real environments.

## Understanding Robot Locomotion Learning

Reinforcement learning provides a framework for robots to learn movement patterns by receiving rewards for successful actions and penalties for failures. This approach is particularly powerful for learning complex locomotion that's difficult to program explicitly.

### Key Tasks from Our Plan:
- T048: Implement learning to move with RL in src/navigation/rl_balance.py

## Reinforcement Learning Environment Node

Let's create a comprehensive RL environment for locomotion learning:

```python
#!/usr/bin/env python3
"""
Reinforcement Learning Environment for Robot Locomotion
Implements RL training for robot balance and movement
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64, String, Bool
from nav_msgs.msg import Odometry
from tf2_ros import TransformListener, Buffer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json
import time

class RLBalanceNode(Node):
    def __init__(self):
        super().__init__('rl_balance_node')

        # Create subscribers for robot state
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Create publishers for control commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.joint_cmd_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )
        self.rl_status_pub = self.create_publisher(
            String,
            '/rl_status',
            10
        )
        self.reward_pub = self.create_publisher(
            Float64,
            '/rl_reward',
            10
        )

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.imu_data = {'orientation': [0, 0, 0, 1], 'angular_velocity': [0, 0, 0], 'linear_acceleration': [0, 0, 0]}
        self.odom_data = {'position': [0, 0, 0], 'velocity': [0, 0, 0], 'angular_velocity': [0, 0, 0]}
        self.laser_data = []

        # RL parameters
        self.state_size = 24  # Define state space size
        self.action_size = 6  # Define action space size
        self.learning_rate = 0.001
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_size = 10000
        self.batch_size = 32

        # Neural network components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self.create_q_network()
        self.target_network = self.create_q_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Experience replay
        self.memory = deque(maxlen=self.memory_size)

        # Training parameters
        self.is_training = True
        self.training_frequency = 1.0  # Hz
        self.episode_count = 0
        self.step_count = 0
        self.max_steps_per_episode = 1000

        # Locomotion goals
        self.balance_target = [0.0, 0.0, 0.0]  # [roll, pitch, yaw]
        self.movement_target = [0.0, 0.0, 0.0]  # [x, y, theta]

        # Create training timer
        self.train_timer = self.create_timer(1.0/self.training_frequency, self.train_step)

        self.get_logger().info('RL Balance Node Started')

    def create_q_network(self):
        """Create the Q-network for reinforcement learning"""
        class QNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 256)
                self.fc2 = nn.Linear(256, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, action_size)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                return self.fc4(x)

        return QNetwork(self.state_size, self.action_size).to(self.device)

    def joint_state_callback(self, msg):
        """Process joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg):
        """Process IMU data for balance information"""
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def odom_callback(self, msg):
        """Process odometry data for movement tracking"""
        self.odom_data = {
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            'velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            'angular_velocity': [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        }

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        self.laser_data = list(msg.ranges)

    def get_robot_state(self):
        """Get the current state of the robot for RL"""
        # Extract relevant information from sensors
        orientation = self.imu_data['orientation']
        angular_vel = self.imu_data['angular_velocity']
        linear_acc = self.imu_data['linear_acceleration']
        velocity = self.odom_data['velocity']
        angular_velocity = self.odom_data['angular_velocity']

        # Convert quaternion to euler angles
        roll, pitch, yaw = self.quaternion_to_euler(orientation)

        # Create state vector
        state = np.array([
            # Orientation (balance)
            roll, pitch, yaw,
            # Angular velocities
            angular_vel[0], angular_vel[1], angular_vel[2],
            # Linear accelerations
            linear_acc[0], linear_acc[1], linear_acc[2],
            # Velocities
            velocity[0], velocity[1], velocity[2],
            # Angular velocities
            angular_velocity[0], angular_velocity[1], angular_velocity[2],
            # Joint positions (simplified - using a few key joints)
            self.joint_positions.get('left_wheel_joint', 0),
            self.joint_positions.get('right_wheel_joint', 0),
            # Laser scan features (first few readings for simplicity)
            self.laser_data[0] if self.laser_data else 10.0,
            self.laser_data[len(self.laser_data)//2] if self.laser_data else 10.0,
            self.laser_data[-1] if self.laser_data else 10.0,
            # Previous actions (for temporal consistency)
            0, 0, 0,  # Placeholder for previous actions
        ])

        # Ensure state has the correct size
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)), 'constant')
        elif len(state) > self.state_size:
            state = state[:self.state_size]

        return state

    def quaternion_to_euler(self, quat):
        """Convert quaternion to euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.action_size)

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        # Select action with highest Q-value
        action = q_values.max(1)[1].item()
        return action

    def calculate_reward(self, state, action, next_state):
        """Calculate reward based on robot state"""
        # Extract relevant values from states
        roll, pitch, _, _, _, _, _, _, _, vel_x, vel_y, vel_z, _, _, _ = next_state[:15]

        reward = 0.0

        # Balance reward: penalize large roll/pitch angles
        balance_penalty = -(abs(roll) + abs(pitch)) * 10
        reward += balance_penalty

        # Velocity reward: encourage forward movement
        forward_velocity = vel_x  # Assuming x is forward direction
        velocity_reward = max(0, forward_velocity) * 5
        reward += velocity_reward

        # Stability reward: penalize excessive angular velocities
        # (these would be in the state vector at appropriate indices)

        # Safety reward: penalize being too close to obstacles
        if self.laser_data:
            min_distance = min(self.laser_data) if self.laser_data else float('inf')
            if min_distance < 0.5:  # 50cm threshold
                reward -= (0.5 - min_distance) * 100  # Heavy penalty for getting too close

        # Small time penalty to encourage efficiency
        reward -= 0.01

        # Bonus for maintaining balance and moving forward
        if abs(roll) < 0.1 and abs(pitch) < 0.1 and forward_velocity > 0.1:
            reward += 0.1

        return float(reward)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_step(self):
        """Execute one step of training"""
        if not self.is_training:
            return

        # Get current state
        current_state = self.get_robot_state()

        # Select action
        action = self.select_action(current_state)

        # Execute action (this would actually control the robot in a real system)
        self.execute_action(action)

        # Get next state after action
        next_state = self.get_robot_state()

        # Calculate reward
        reward = self.calculate_reward(current_state, action, next_state)

        # Check if episode should end (simplified condition)
        roll, pitch, _, _, _, _, _, _, _, vel_x, _, _, _, _, _ = next_state[:15]
        done = abs(roll) > 1.0 or abs(pitch) > 1.0  # Robot fell over

        # Store experience
        self.remember(current_state, action, reward, next_state, done)

        # Publish reward
        reward_msg = Float64()
        reward_msg.data = reward
        self.reward_pub.publish(reward_msg)

        # Train network
        self.replay()

        # Update step count
        self.step_count += 1

        # Update target network periodically
        if self.step_count % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Check if episode ended
        if done:
            self.episode_count += 1
            self.step_count = 0
            self.get_logger().info(f'Episode {self.episode_count} ended')

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                'episode': self.episode_count,
                'epsilon': self.epsilon,
                'steps': self.step_count
            })
            self.rl_status_pub.publish(status_msg)

    def execute_action(self, action):
        """Execute the selected action on the robot"""
        # Convert discrete action to continuous control command
        cmd_vel = Twist()

        # Define action mapping (simplified example)
        if action == 0:  # Move forward
            cmd_vel.linear.x = 0.2
        elif action == 1:  # Move backward
            cmd_vel.linear.x = -0.2
        elif action == 2:  # Turn left
            cmd_vel.angular.z = 0.3
        elif action == 3:  # Turn right
            cmd_vel.angular.z = -0.3
        elif action == 4:  # Stop
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        elif action == 5:  # Balance adjustment (simplified)
            # In a real system, this would adjust joint positions for balance
            pass

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        self.get_logger().info(f'Model saved to {filepath}')

    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.get_logger().info(f'Model loaded from {filepath}')

def main(args=None):
    rclpy.init(args=args)
    rl_node = RLBalanceNode()

    try:
        rclpy.spin(rl_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save model before shutting down
        rl_node.save_model('/tmp/rl_balance_model.pth')
        rl_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Locomotion Learning Node

Let's create a more sophisticated node for complex locomotion patterns:

```python
#!/usr/bin/env python3
"""
Advanced Locomotion Learning Node for Physical AI System
Implements sophisticated movement pattern learning
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64, String, Bool
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json

class AdvancedLocomotionNode(Node):
    def __init__(self):
        super().__init__('advanced_locomotion_node')

        # Create subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Create publishers
        self.joint_traj_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory',
            10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.locomotion_status_pub = self.create_publisher(
            String,
            '/locomotion_status',
            10
        )
        self.pattern_pub = self.create_publisher(
            String,
            '/locomotion_pattern',
            10
        )

        # Robot state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.imu_data = {'orientation': [0, 0, 0, 1], 'angular_velocity': [0, 0, 0], 'linear_acceleration': [0, 0, 0]}
        self.odom_data = {'position': [0, 0, 0], 'velocity': [0, 0, 0], 'angular_velocity': [0, 0, 0]}
        self.laser_data = []

        # Locomotion learning parameters
        self.state_size = 30  # Extended state space for complex locomotion
        self.action_size = 12  # Multiple joint control actions
        self.learning_rate = 0.0005
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.memory_size = 20000
        self.batch_size = 64

        # Neural network for locomotion control
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = self.create_policy_network()
        self.value_network = self.create_value_network()
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        # Experience replay
        self.memory = deque(maxlen=self.memory_size)

        # Locomotion patterns
        self.locomotion_patterns = {
            'walk': [0, 1, 2, 3, 4, 5],
            'turn_left': [6, 7],
            'turn_right': [8, 9],
            'balance': [10, 11]
        }
        self.current_pattern = 'walk'

        # Training parameters
        self.is_training = True
        self.training_frequency = 2.0  # Higher frequency for locomotion
        self.episode_count = 0
        self.step_count = 0

        # Create training timer
        self.train_timer = self.create_timer(1.0/self.training_frequency, self.train_step)

        self.get_logger().info('Advanced Locomotion Node Started')

    def create_policy_network(self):
        """Create policy network for locomotion"""
        class PolicyNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super(PolicyNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, action_size)
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                x = self.fc4(x)
                return self.softmax(x)

        return PolicyNetwork(self.state_size, self.action_size).to(self.device)

    def create_value_network(self):
        """Create value network for locomotion"""
        class ValueNetwork(nn.Module):
            def __init__(self, state_size):
                super(ValueNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                return self.fc4(x)

        return ValueNetwork(self.state_size).to(self.device)

    def joint_state_callback(self, msg):
        """Process joint state information"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg):
        """Process IMU data for balance information"""
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def odom_callback(self, msg):
        """Process odometry data for movement tracking"""
        self.odom_data = {
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            'velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            'angular_velocity': [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        }

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        self.laser_data = list(msg.ranges)

    def get_robot_state(self):
        """Get the current state for locomotion learning"""
        # Extract comprehensive state information
        orientation = self.imu_data['orientation']
        angular_vel = self.imu_data['angular_velocity']
        linear_acc = self.imu_data['linear_acceleration']
        velocity = self.odom_data['velocity']
        angular_velocity = self.odom_data['angular_velocity']

        # Convert quaternion to euler angles
        roll, pitch, yaw = self.quaternion_to_euler(orientation)

        # Create extended state vector
        state = np.array([
            # Balance state
            roll, pitch, yaw,
            angular_vel[0], angular_vel[1], angular_vel[2],
            linear_acc[0], linear_acc[1], linear_acc[2],
            # Movement state
            velocity[0], velocity[1], velocity[2],
            angular_velocity[0], angular_velocity[1], angular_velocity[2],
            # Joint state (simplified - using key joints)
            self.joint_positions.get('left_wheel_joint', 0),
            self.joint_positions.get('right_wheel_joint', 0),
            self.joint_positions.get('left_arm_joint', 0),
            self.joint_positions.get('right_arm_joint', 0),
            # Joint velocities
            self.joint_velocities.get('left_wheel_joint', 0),
            self.joint_velocities.get('right_wheel_joint', 0),
            # Environment state
            self.laser_data[0] if self.laser_data else 10.0,  # Front distance
            self.laser_data[len(self.laser_data)//4] if self.laser_data else 10.0,  # Left distance
            self.laser_data[3*len(self.laser_data)//4] if self.laser_data else 10.0,  # Right distance
            self.laser_data[len(self.laser_data)//2] if self.laser_data else 10.0,  # Center distance
            # Time-based features
            np.sin(self.step_count * 0.01),  # Oscillation for rhythmic patterns
            np.cos(self.step_count * 0.01),
            # Previous action (one-hot encoded)
            0, 0, 0, 0  # Placeholder for previous action
        ])

        # Ensure state has correct size
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)), 'constant')
        elif len(state) > self.state_size:
            state = state[:self.state_size]

        return state

    def quaternion_to_euler(self, quat):
        """Convert quaternion to euler angles"""
        x, y, z, w = quat

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def select_action(self, state):
        """Select action using policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            action = torch.multinomial(action_probs, 1).item()

        return action

    def calculate_locomotion_reward(self, state, action, next_state):
        """Calculate reward for locomotion learning"""
        reward = 0.0

        # Extract state components
        roll, pitch, _, _, _, _, _, _, _, vel_x, vel_y, vel_z, _, _, _ = next_state[:15]

        # Balance reward
        balance_penalty = -(abs(roll) + abs(pitch)) * 15
        reward += balance_penalty

        # Forward velocity reward (for walking pattern)
        if self.current_pattern == 'walk':
            forward_reward = max(0, vel_x) * 10
            reward += forward_reward

        # Turning reward (for turning patterns)
        if self.current_pattern in ['turn_left', 'turn_right']:
            angular_vel_z = next_state[14] if len(next_state) > 14 else 0
            if (self.current_pattern == 'turn_left' and angular_vel_z > 0) or \
               (self.current_pattern == 'turn_right' and angular_vel_z < 0):
                turn_reward = abs(angular_vel_z) * 10
                reward += turn_reward

        # Energy efficiency reward (penalize excessive joint velocities)
        joint_vel_penalty = 0
        for joint_vel in [next_state[16], next_state[17]]:  # Simplified joint velocity access
            joint_vel_penalty += abs(joint_vel) * 0.1
        reward -= joint_vel_penalty

        # Safety reward (obstacle avoidance)
        if self.laser_data:
            min_distance = min(self.laser_data) if self.laser_data else float('inf')
            if min_distance < 0.3:
                reward -= (0.3 - min_distance) * 200

        # Small time penalty
        reward -= 0.02

        # Bonus for stable, forward movement
        if abs(roll) < 0.2 and abs(pitch) < 0.2 and vel_x > 0.2:
            reward += 0.5

        return float(reward)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def update_networks(self):
        """Update policy and value networks using PPO-style update"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)

        # Compute value targets
        with torch.no_grad():
            next_values = self.value_network(states).squeeze()
            value_targets = rewards + self.gamma * next_values

        # Update value network
        current_values = self.value_network(states).squeeze()
        value_loss = nn.MSELoss()(current_values, value_targets)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        action_probs = self.policy_network(states)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1))

        # Simple policy gradient update (without advantage for simplicity)
        policy_loss = -(torch.log(selected_action_probs) * rewards.unsqueeze(1)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_step(self):
        """Execute one training step"""
        if not self.is_training:
            return

        current_state = self.get_robot_state()
        action = self.select_action(current_state)
        self.execute_locomotion_action(action)
        next_state = self.get_robot_state()

        reward = self.calculate_locomotion_reward(current_state, action, next_state)
        done = self.check_episode_done(next_state)

        self.remember(current_state, action, reward, next_state, done)

        # Update networks
        self.update_networks()

        self.step_count += 1

        if done:
            self.episode_count += 1
            self.step_count = 0
            self.get_logger().info(f'Locomotion episode {self.episode_count} ended')

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                'episode': self.episode_count,
                'epsilon': self.epsilon,
                'pattern': self.current_pattern
            })
            self.locomotion_status_pub.publish(status_msg)

    def execute_locomotion_action(self, action):
        """Execute locomotion action"""
        # Map action to specific locomotion pattern
        if 0 <= action <= 5:  # Walking pattern
            self.current_pattern = 'walk'
            self.execute_walk_pattern()
        elif 6 <= action <= 7:  # Turn left
            self.current_pattern = 'turn_left'
            self.execute_turn_pattern('left')
        elif 8 <= action <= 9:  # Turn right
            self.current_pattern = 'turn_right'
            self.execute_turn_pattern('right')
        elif 10 <= action <= 11:  # Balance adjustment
            self.current_pattern = 'balance'
            self.execute_balance_pattern()

    def execute_walk_pattern(self):
        """Execute walking locomotion pattern"""
        # Publish joint trajectory for walking
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['left_wheel_joint', 'right_wheel_joint']

        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0]  # Simplified
        point.velocities = [0.5, 0.5]  # Move both wheels forward
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 50000000  # 50ms

        traj_msg.points = [point]
        self.joint_traj_pub.publish(traj_msg)

    def execute_turn_pattern(self, direction):
        """Execute turning locomotion pattern"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['left_wheel_joint', 'right_wheel_joint']

        point = JointTrajectoryPoint()
        if direction == 'left':
            point.positions = [0.0, 0.0]
            point.velocities = [-0.3, 0.3]  # Turn left
        else:  # right
            point.positions = [0.0, 0.0]
            point.velocities = [0.3, -0.3]  # Turn right
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 50000000

        traj_msg.points = [point]
        self.joint_traj_pub.publish(traj_msg)

    def execute_balance_pattern(self):
        """Execute balance adjustment pattern"""
        # In a real system, this would adjust joint positions for balance
        # For now, just publish the pattern
        pattern_msg = String()
        pattern_msg.data = json.dumps({
            'pattern': 'balance_adjustment',
            'timestamp': self.get_clock().now().to_msg()
        })
        self.pattern_pub.publish(pattern_msg)

    def check_episode_done(self, state):
        """Check if the episode should end"""
        roll, pitch, _, _, _, _, _, _, _, _, _, _, _, _, _ = state[:15]
        # End episode if robot falls over
        return abs(roll) > 1.0 or abs(pitch) > 1.0

def main(args=None):
    rclpy.init(args=args)
    locomotion_node = AdvancedLocomotionNode()

    try:
        rclpy.spin(locomotion_node)
    except KeyboardInterrupt:
        pass
    finally:
        locomotion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Locomotion Learning Configuration

Create a configuration file for locomotion learning parameters:

```yaml
# locomotion_learning_config.yaml
locomotion_learning:
  # Network architecture
  network:
    state_size: 30
    action_size: 12
    hidden_layers: [512, 256, 128]
    learning_rate: 0.0005
    gamma: 0.95
    epsilon_start: 1.0
    epsilon_min: 0.01
    epsilon_decay: 0.999

  # Training parameters
  training:
    batch_size: 64
    memory_size: 20000
    update_frequency: 2.0  # Hz
    target_network_update_freq: 100  # steps

  # Reward shaping
  rewards:
    balance_weight: 15.0
    velocity_weight: 10.0
    energy_penalty_weight: 0.1
    safety_weight: 200.0
    time_penalty: 0.02
    stability_bonus: 0.5

  # Locomotion patterns
  patterns:
    walk:
      description: "Forward walking motion"
      joints: ["left_wheel", "right_wheel"]
      parameters: {"speed": 0.5, "frequency": 2.0}
    turn_left:
      description: "Left turning motion"
      joints: ["left_wheel", "right_wheel"]
      parameters: {"left_speed": -0.3, "right_speed": 0.3}
    turn_right:
      description: "Right turning motion"
      joints: ["left_wheel", "right_wheel"]
      parameters: {"left_speed": 0.3, "right_speed": -0.3}
    balance:
      description: "Balance adjustment"
      joints: ["left_leg", "right_leg", "torso"]
      parameters: {"adjustment_gain": 0.1}

  # Safety constraints
  safety:
    max_roll: 1.0  # radians
    max_pitch: 1.0  # radians
    min_obstacle_distance: 0.3  # meters
    max_joint_velocity: 2.0  # rad/s

  # Simulation parameters
  simulation:
    physics_accuracy: 0.001  # seconds
    sensor_noise:
      imu: 0.01
      joint: 0.005
      laser: 0.02
```

## Locomotion Learning Launch File

Create a launch file for the locomotion learning system:

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
    config_file = LaunchConfiguration('config_file', default='locomotion_learning_config.yaml')

    # RL balance node
    rl_balance_node = Node(
        package='physical_ai_navigation',
        executable='rl_balance_node',
        name='rl_balance_node',
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

    # Advanced locomotion node
    advanced_locomotion_node = Node(
        package='physical_ai_navigation',
        executable='advanced_locomotion_node',
        name='advanced_locomotion_node',
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
            default_value='locomotion_learning_config.yaml',
            description='Configuration file for locomotion learning'
        ),
        rl_balance_node,
        advanced_locomotion_node
    ])
```

## Quality Assurance for Locomotion Learning

### Performance Metrics
- **Learning Rate**: How quickly the robot learns to perform tasks
- **Stability**: Ability to maintain balance during locomotion
- **Efficiency**: Energy consumption and path efficiency
- **Generalization**: Performance in unseen environments

### Safety Considerations
Following our constitution's "Safety is Intelligence" principle:

1. **Physical Constraints**: Respect joint limits and actuator capabilities
2. **Stability Monitoring**: Continuous balance assessment
3. **Emergency Stops**: Immediate halt on dangerous conditions
4. **Learning Safety**: Safe exploration during learning process

### Testing Scenarios
1. **Basic Balance**: Learning to maintain upright position
2. **Simple Locomotion**: Learning basic movement patterns
3. **Obstacle Navigation**: Learning to navigate around obstacles
4. **Recovery Behaviors**: Learning to recover from perturbations

## Looking Forward

With our robot learning to move intelligently through reinforcement learning, the next act will focus on voice and language processing systems that will enable our robot to understand and respond to human commands.

[Continue to Chapter 13: Voice Command Pipeline](./chapter-13-voice-command.md)