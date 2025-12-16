# Chapter 16: System Integration - Bringing It All Together

## The Complete Autonomous Robot

In this chapter, we implement the complete system integration that brings together all the components we've developed throughout this book. This represents the culmination of our "Autonomous Humanoid Challenge" where our robot must hear a human voice command, understand intent using language models, plan a sequence of actions, navigate a physical environment, identify objects using perception, and manipulate the world safely and successfully.

## Understanding System Integration

System integration is the process of combining all individual components into a cohesive, functional whole. This chapter focuses on:
- Integrating perception, cognition, and action systems
- Creating unified communication between all modules
- Implementing the complete voice→intelligence→motion pipeline
- Ensuring robust operation across all capabilities

### Key Tasks from Our Plan:
- T052: Complete system architecture integration in src/system_integration.py

## Main Integration Node

Let's create the main integration node that coordinates all system components:

```python
#!/usr/bin/env python3
"""
Main Integration Node for Physical AI System
Coordinates all system components into a unified autonomous robot
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float64
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import MarkerArray
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue

class SystemState(Enum):
    """Overall system state"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class SystemStatus:
    """Current status of all system components"""
    perception_active: bool = False
    navigation_active: bool = False
    speech_active: bool = False
    cognitive_active: bool = False
    manipulation_active: bool = False
    system_state: SystemState = SystemState.IDLE
    last_command: str = ""
    execution_progress: float = 0.0
    safety_status: str = "safe"

class SystemIntegrationNode(Node):
    def __init__(self):
        super().__init__('system_integration_node')

        # Create subscribers for all system components
        self.voice_command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )
        self.task_plan_sub = self.create_subscription(
            String,
            '/task_plan',
            self.task_plan_callback,
            10
        )
        self.vla_plan_sub = self.create_subscription(
            String,
            '/vla/action_plan',
            self.vla_plan_callback,
            10
        )
        self.navigation_status_sub = self.create_subscription(
            String,
            '/navigation_status',
            self.navigation_status_callback,
            10
        )
        self.task_status_sub = self.create_subscription(
            String,
            '/task_status',
            self.task_status_callback,
            10
        )
        self.emergency_stop_sub = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_stop_callback,
            10
        )
        self.safety_status_sub = self.create_subscription(
            String,
            '/safety/status',
            self.safety_status_callback,
            10
        )

        # Create publishers for system-wide communication
        self.system_status_pub = self.create_publisher(
            String,
            '/system/status',
            10
        )
        self.system_command_pub = self.create_publisher(
            String,
            '/system/command',
            10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )
        self.system_state_pub = self.create_publisher(
            String,
            '/system/state',
            10
        )

        # System state management
        self.system_status = SystemStatus()
        self.active_plan = None
        self.plan_queue = queue.Queue()
        self.execution_thread = None
        self.execution_active = False

        # Component health tracking
        self.component_health = {
            'perception': {'healthy': True, 'last_update': time.time()},
            'navigation': {'healthy': True, 'last_update': time.time()},
            'speech': {'healthy': True, 'last_update': time.time()},
            'cognitive': {'healthy': True, 'last_update': time.time()},
            'manipulation': {'healthy': True, 'last_update': time.time()}
        }

        # System parameters
        self.max_execution_time = 300.0  # 5 minutes
        self.health_check_interval = 5.0  # seconds
        self.safety_timeout = 10.0  # seconds without safety update

        # Create timers
        self.status_timer = self.create_timer(1.0, self.publish_system_status)
        self.health_timer = self.create_timer(self.health_check_interval, self.check_component_health)

        self.get_logger().info('System Integration Node Started')

    def voice_command_callback(self, msg):
        """Handle incoming voice commands"""
        command = msg.data.lower().strip()

        if not command:
            return

        self.get_logger().info(f'System received voice command: {command}')

        # Update system state
        self.system_status.system_state = SystemState.PROCESSING
        self.system_status.last_command = command

        # Publish system state
        state_msg = String()
        state_msg.data = self.system_status.system_state.value
        self.system_state_pub.publish(state_msg)

    def task_plan_callback(self, msg):
        """Handle incoming task plans"""
        try:
            plan_data = json.loads(msg.data)

            # Add to execution queue
            self.plan_queue.put(plan_data)

            # Start execution if not already running
            if not self.execution_active:
                self.start_execution_thread()

            self.get_logger().info(f'Added plan to execution queue: {plan_data.get("command", "unknown")}')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task plan message')

    def vla_plan_callback(self, msg):
        """Handle VLA action plans"""
        try:
            plan_data = json.loads(msg.data)

            # Add to execution queue
            self.plan_queue.put(plan_data)

            # Start execution if not already running
            if not self.execution_active:
                self.start_execution_thread()

            self.get_logger().info(f'Added VLA plan to execution queue: {plan_data.get("command", "unknown")}')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in VLA plan message')

    def navigation_status_callback(self, msg):
        """Handle navigation status updates"""
        try:
            status_data = json.loads(msg.data)

            # Update navigation component health
            self.component_health['navigation']['last_update'] = time.time()
            self.component_health['navigation']['healthy'] = True

            # Update execution progress if applicable
            if 'distance_to_goal' in status_data:
                # Calculate progress based on distance (simplified)
                max_distance = 10.0  # meters
                distance = status_data.get('distance_to_goal', max_distance)
                self.system_status.execution_progress = max(0.0, 1.0 - (distance / max_distance))

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in navigation status message')

    def task_status_callback(self, msg):
        """Handle task status updates"""
        try:
            status_data = json.loads(msg.data)

            # Update system state based on task status
            status = status_data.get('status', '')
            if status == 'navigating':
                self.system_status.system_state = SystemState.EXECUTING
            elif status == 'goal_reached':
                self.system_status.system_state = SystemState.COMPLETED
            elif status == 'failed':
                self.system_status.system_state = SystemState.ERROR

            # Update manipulation component health
            self.component_health['manipulation']['last_update'] = time.time()
            self.component_health['manipulation']['healthy'] = True

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task status message')

    def emergency_stop_callback(self, msg):
        """Handle emergency stop"""
        if msg.data:
            self.system_status.system_state = SystemState.EMERGENCY_STOP

            # Stop all motion
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)

            # Cancel any navigation goals
            cancel_msg = Bool()
            cancel_msg.data = True
            # This would go to navigation cancel topic in real system

            self.get_logger().warn('EMERGENCY STOP ACTIVATED')

    def safety_status_callback(self, msg):
        """Handle safety system status"""
        try:
            status_data = json.loads(msg.data)
            safety_status = status_data.get('status', 'unknown')

            # Update safety status
            self.system_status.safety_status = safety_status

            # Update safety component health
            self.component_health['perception']['last_update'] = time.time()
            self.component_health['perception']['healthy'] = status_data.get('perception_healthy', True)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in safety status message')

    def start_execution_thread(self):
        """Start the plan execution thread"""
        if self.execution_thread is None or not self.execution_thread.is_alive():
            self.execution_active = True
            self.execution_thread = threading.Thread(target=self.execute_plans)
            self.execution_thread.daemon = True
            self.execution_thread.start()

    def execute_plans(self):
        """Execute plans from the queue in a separate thread"""
        while self.execution_active and rclpy.ok():
            try:
                # Get next plan from queue (non-blocking)
                if not self.plan_queue.empty():
                    plan = self.plan_queue.get_nowait()

                    if plan:
                        self.execute_single_plan(plan)
                else:
                    # Small delay when queue is empty
                    time.sleep(0.1)

            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f'Error in execution thread: {str(e)}')
                time.sleep(0.1)

    def execute_single_plan(self, plan: Dict):
        """Execute a single action plan"""
        self.get_logger().info(f'Executing plan: {plan.get("command", "unknown")}')

        # Update system state
        self.system_status.system_state = SystemState.EXECUTING
        self.active_plan = plan

        steps = plan.get('steps', [])
        total_steps = len(steps)
        completed_steps = 0

        for i, step in enumerate(steps):
            if self.system_status.system_state == SystemState.EMERGENCY_STOP:
                self.get_logger().warn('Plan execution stopped due to emergency stop')
                break

            action = step.get('action', '').upper()
            params = step.get('parameters', {})

            self.get_logger().info(f'Executing step {i+1}/{total_steps}: {action}')

            # Execute the action
            success = self.execute_action(action, params)

            if success:
                completed_steps += 1
                self.system_status.execution_progress = completed_steps / total_steps
            else:
                self.get_logger().error(f'Failed to execute step: {action}')
                self.system_status.system_state = SystemState.ERROR
                break

        # Update final state
        if self.system_status.system_state != SystemState.ERROR:
            self.system_status.system_state = SystemState.COMPLETED
            self.system_status.execution_progress = 1.0

        self.active_plan = None

        self.get_logger().info(f'Plan execution completed with status: {self.system_status.system_state.value}')

    def execute_action(self, action: str, params: Dict) -> bool:
        """Execute a single action"""
        try:
            if action == 'NAVIGATE_TO':
                return self.execute_navigation_action(params)
            elif action == 'GRASP_OBJECT':
                return self.execute_manipulation_action(params, 'grasp')
            elif action == 'PLACE_OBJECT':
                return self.execute_manipulation_action(params, 'place')
            elif action == 'SPEAK':
                return self.execute_speak_action(params)
            elif action == 'MOVE_ROBOT':
                return self.execute_move_action(params)
            elif action == 'ACTIVATE_FOLLOW_MODE':
                return self.execute_follow_action(params)
            else:
                self.get_logger().warn(f'Unknown action: {action}')
                return False

        except Exception as e:
            self.get_logger().error(f'Error executing action {action}: {str(e)}')
            return False

    def execute_navigation_action(self, params: Dict) -> bool:
        """Execute navigation action"""
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        location_name = params.get('location_name', 'unknown')

        # Create navigation goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = float(x)
        goal_msg.pose.position.y = float(y)
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.navigation_goal_pub.publish(goal_msg)
        self.get_logger().info(f'Navigating to {location_name} at ({x}, {y})')

        # Update navigation component health
        self.component_health['navigation']['last_update'] = time.time()

        return True

    def execute_manipulation_action(self, params: Dict, action_type: str) -> bool:
        """Execute manipulation action"""
        obj_class = params.get('object_class', 'unknown')
        self.get_logger().info(f'{action_type.capitalize()}ing object: {obj_class}')

        # Update manipulation component health
        self.component_health['manipulation']['last_update'] = time.time()

        return True

    def execute_speak_action(self, params: Dict) -> bool:
        """Execute speak action"""
        text = params.get('text', 'Hello')
        self.get_logger().info(f'Speaking: {text}')

        # In a real system, this would go to TTS
        return True

    def execute_move_action(self, params: Dict) -> bool:
        """Execute movement action"""
        direction = params.get('direction', 'stop')
        speed = params.get('speed', 0.3)

        cmd_vel = Twist()

        if direction == 'forward':
            cmd_vel.linear.x = speed
        elif direction == 'backward':
            cmd_vel.linear.x = -speed
        elif direction == 'left':
            cmd_vel.angular.z = speed
        elif direction == 'right':
            cmd_vel.angular.z = -speed
        else:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd_vel)
        self.get_logger().info(f'Moving {direction} at speed {speed}')

        return True

    def execute_follow_action(self, params: Dict) -> bool:
        """Execute follow action"""
        self.get_logger().info('Activating follow mode')

        # In a real system, this would activate person following
        return True

    def check_component_health(self):
        """Check health of all system components"""
        current_time = time.time()
        timeout_threshold = self.health_check_interval * 2  # 2x interval

        for component, health_info in self.component_health.items():
            time_since_update = current_time - health_info['last_update']
            is_healthy = time_since_update < timeout_threshold

            health_info['healthy'] = is_healthy

            if not is_healthy:
                self.get_logger().warn(f'{component} component health check failed (last update: {time_since_update:.1f}s ago)')

        # Update system status based on component health
        all_healthy = all(info['healthy'] for info in self.component_health.values())

        if not all_healthy:
            self.get_logger().warn('One or more components are unhealthy')

    def publish_system_status(self):
        """Publish overall system status"""
        status_msg = String()
        status_msg.data = json.dumps({
            'timestamp': time.time(),
            'system_state': self.system_status.system_state.value,
            'component_health': {
                comp: info['healthy'] for comp, info in self.component_health.items()
            },
            'last_command': self.system_status.last_command,
            'execution_progress': self.system_status.execution_progress,
            'safety_status': self.system_status.safety_status,
            'plan_queue_size': self.plan_queue.qsize()
        })
        self.system_status_pub.publish(status_msg)

        # Also publish system state separately
        state_msg = String()
        state_msg.data = self.system_status.system_state.value
        self.system_state_pub.publish(state_msg)

    def destroy_node(self):
        """Clean up before node destruction"""
        self.execution_active = False
        if self.execution_thread:
            self.execution_thread.join(timeout=2.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    integration_node = SystemIntegrationNode()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        pass
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## System Monitoring Node

Let's create a system monitoring node that provides oversight of the entire system:

```python
#!/usr/bin/env python3
"""
System Monitoring Node for Physical AI System
Provides comprehensive system monitoring and diagnostics
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
import json
import time
from typing import Dict, List
import psutil
import os

class SystemMonitoringNode(Node):
    def __init__(self):
        super().__init__('system_monitoring_node')

        # Create subscribers for system status
        self.system_status_sub = self.create_subscription(
            String,
            '/system/status',
            self.system_status_callback,
            10
        )
        self.system_state_sub = self.create_subscription(
            String,
            '/system/state',
            self.system_state_callback,
            10
        )

        # Create publishers for monitoring
        self.monitoring_status_pub = self.create_publisher(
            String,
            '/monitoring/status',
            10
        )
        self.performance_metrics_pub = self.create_publisher(
            String,
            '/monitoring/performance',
            10
        )
        self.health_report_pub = self.create_publisher(
            String,
            '/monitoring/health_report',
            10
        )

        # System monitoring parameters
        self.monitoring_interval = 2.0  # seconds
        self.cpu_threshold = 80.0  # percent
        self.memory_threshold = 85.0  # percent
        self.disk_threshold = 90.0  # percent

        # Performance tracking
        self.performance_history = []
        self.max_history_length = 50

        # Create monitoring timer
        self.monitor_timer = self.create_timer(self.monitoring_interval, self.perform_monitoring)

        self.get_logger().info('System Monitoring Node Started')

    def system_status_callback(self, msg):
        """Process system status updates"""
        try:
            status_data = json.loads(msg.data)

            # Log important status changes
            system_state = status_data.get('system_state', 'unknown')
            if system_state in ['error', 'emergency_stop']:
                self.get_logger().warn(f'System in {system_state} state')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in system status message')

    def system_state_callback(self, msg):
        """Process system state updates"""
        state = msg.data
        self.get_logger().info(f'System state: {state}')

    def perform_monitoring(self):
        """Perform comprehensive system monitoring"""
        # Collect system metrics
        metrics = self.collect_system_metrics()

        # Check for issues
        issues = self.check_system_issues(metrics)

        # Generate health report
        health_report = self.generate_health_report(metrics, issues)

        # Publish monitoring status
        status_msg = String()
        status_msg.data = json.dumps({
            'timestamp': time.time(),
            'metrics': metrics,
            'issues': issues,
            'health_score': health_report['health_score']
        })
        self.monitoring_status_pub.publish(status_msg)

        # Publish performance metrics
        perf_msg = String()
        perf_msg.data = json.dumps(metrics)
        self.performance_metrics_pub.publish(perf_msg)

        # Publish health report
        health_msg = String()
        health_msg.data = json.dumps(health_report)
        self.health_report_pub.publish(health_msg)

        # Log health score
        self.get_logger().info(f'System health score: {health_report["health_score"]:.1f}%')

    def collect_system_metrics(self) -> Dict:
        """Collect system performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids()),
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
            'system_uptime': time.time() - psutil.boot_time(),
            'process_memory_mb': self.get_process_memory_mb()
        }

        # Add to history
        self.performance_history.append(metrics)
        if len(self.performance_history) > self.max_history_length:
            self.performance_history.pop(0)

        return metrics

    def get_process_memory_mb(self) -> float:
        """Get memory usage of this process in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def check_system_issues(self, metrics: Dict) -> List[str]:
        """Check for system issues based on metrics"""
        issues = []

        # Check CPU usage
        if metrics['cpu_percent'] > self.cpu_threshold:
            issues.append(f'High CPU usage: {metrics["cpu_percent"]:.1f}%')

        # Check memory usage
        if metrics['memory_percent'] > self.memory_threshold:
            issues.append(f'High memory usage: {metrics["memory_percent"]:.1f}%')

        # Check disk usage
        if metrics['disk_percent'] > self.disk_threshold:
            issues.append(f'High disk usage: {metrics["disk_percent"]:.1f}%')

        # Check process count (arbitrary threshold)
        if metrics['process_count'] > 1000:
            issues.append(f'High process count: {metrics["process_count"]}')

        return issues

    def generate_health_report(self, metrics: Dict, issues: List[str]) -> Dict:
        """Generate comprehensive health report"""
        # Calculate health score based on various factors
        base_score = 100.0

        # Deduct points for issues
        for issue in issues:
            base_score -= 10.0  # 10 points per issue

        # Deduct points for high resource usage
        cpu_penalty = max(0, metrics['cpu_percent'] - 70) * 0.5  # 0.5 per % above 70
        memory_penalty = max(0, metrics['memory_percent'] - 70) * 0.5
        disk_penalty = max(0, metrics['disk_percent'] - 80) * 0.25

        health_score = max(0, base_score - cpu_penalty - memory_penalty - disk_penalty)

        # Determine health level
        if health_score >= 90:
            health_level = 'excellent'
        elif health_score >= 70:
            health_level = 'good'
        elif health_score >= 50:
            health_level = 'fair'
        else:
            health_level = 'poor'

        return {
            'timestamp': metrics['timestamp'],
            'health_score': health_score,
            'health_level': health_level,
            'metrics': metrics,
            'issues': issues,
            'recommendations': self.generate_recommendations(issues, metrics)
        }

    def generate_recommendations(self, issues: List[str], metrics: Dict) -> List[str]:
        """Generate recommendations based on issues and metrics"""
        recommendations = []

        if any('CPU' in issue for issue in issues):
            recommendations.append('Consider optimizing CPU-intensive processes')

        if any('memory' in issue.lower() for issue in issues):
            recommendations.append('Consider reducing memory usage or adding more RAM')

        if any('disk' in issue.lower() for issue in issues):
            recommendations.append('Clean up disk space or add more storage')

        if not recommendations:
            recommendations.append('System is operating normally')

        return recommendations

def main(args=None):
    rclpy.init(args=args)
    monitoring_node = SystemMonitoringNode()

    try:
        rclpy.spin(monitoring_node)
    except KeyboardInterrupt:
        pass
    finally:
        monitoring_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## System Integration Configuration

Create a configuration file for the system integration:

```yaml
# system_integration_config.yaml
system_integration:
  # Component integration parameters
  integration:
    monitoring_interval: 2.0  # seconds
    execution_timeout: 300.0  # seconds (5 minutes)
    health_check_interval: 5.0  # seconds
    emergency_stop_timeout: 1.0  # seconds

  # Performance thresholds
  performance:
    cpu_threshold: 80.0  # percent
    memory_threshold: 85.0  # percent
    disk_threshold: 90.0  # percent
    process_limit: 1000  # max processes

  # Safety parameters
  safety:
    enabled: true
    emergency_stop_keywords:
      - "stop"
      - "emergency"
      - "help"
      - "danger"
      - "unsafe"
    collision_threshold: 0.5  # meters
    velocity_limits:
      linear: 0.5  # m/s
      angular: 1.0  # rad/s

  # Communication parameters
  communication:
    qos_profile:
      reliability: "reliable"
      durability: "volatile"
      depth: 10
    timeout: 5.0  # seconds
    retry_attempts: 3
    heartbeat_interval: 1.0  # seconds

  # Execution parameters
  execution:
    max_concurrent_plans: 1
    plan_queue_size: 10
    step_timeout: 30.0  # seconds per step
    recovery_attempts: 3

  # Logging parameters
  logging:
    level: "info"
    log_file: "/var/log/physical_ai_system.log"
    rotation_size: "10MB"
    backup_count: 5
    detailed_components:
      - "system_integration"
      - "system_monitoring"
```

## System Integration Launch File

Create a launch file for the complete system integration:

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
    config_file = LaunchConfiguration('config_file', default='system_integration_config.yaml')

    # Include base launch files for all components
    # (In a real system, you would include all the launch files we created earlier)

    # System integration node
    system_integration_node = Node(
        package='physical_ai_system',
        executable='system_integration_node',
        name='system_integration_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_system'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    # System monitoring node
    system_monitoring_node = Node(
        package='physical_ai_system',
        executable='system_monitoring_node',
        name='system_monitoring_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_system'),
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
            default_value='system_integration_config.yaml',
            description='Configuration file for system integration'
        ),
        system_integration_node,
        system_monitoring_node
    ])
```

## Quality Assurance for System Integration

### Performance Metrics
- **Integration Success Rate**: Percentage of tasks completed successfully across all components
- **System Responsiveness**: Time from command to action execution
- **Component Coordination**: How well components work together
- **Resource Utilization**: CPU, memory, and network usage

### Safety Considerations
Following our constitution's "Safety is Intelligence" principle:

1. **System-Wide Safety**: Safety checks across all integrated components
2. **Emergency Protocols**: Unified emergency stop across all systems
3. **Health Monitoring**: Continuous monitoring of all system components
4. **Fail-Safe Mechanisms**: Safe degradation when components fail

### Testing Scenarios
1. **Complete Pipeline**: Test the full voice→intelligence→motion pipeline
2. **Component Failure**: Test system behavior when components fail
3. **Resource Stress**: Test system under high resource usage
4. **Emergency Situations**: Test emergency stop and recovery procedures

## Looking Forward

With our complete system integrated, the next chapter will focus on sim-to-real transfer, where we'll prepare our robot for deployment in the real world.

[Continue to Chapter 17: Sim-to-Real Transfer](./chapter-17-sim-to-real.md)