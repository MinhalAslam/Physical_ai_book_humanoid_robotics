# Chapter 19: Safety & Audit - Ensuring Responsible Robotics

## The Foundation of Ethical AI

In this chapter, we implement comprehensive safety systems and audit procedures that ensure our robot operates responsibly and ethically. This aligns with our constitution's principles that "Safety is Intelligence" and "Human dignity overrides automation." We'll create systems that monitor, verify, and maintain safety throughout all robot operations.

## Understanding Safety in Physical AI

Safety in Physical AI encompasses multiple layers:
- **Physical Safety**: Preventing harm to humans and environment
- **Operational Safety**: Ensuring reliable system operation
- **Ethical Safety**: Respecting human dignity and rights
- **Security Safety**: Protecting against unauthorized access

### Key Tasks from Our Plan:
- T055: Implement safety audit system in src/core/safety_audit.py

## Safety Monitoring Node

Let's create a comprehensive safety monitoring node:

```python
#!/usr/bin/env python3
"""
Safety Monitoring Node for Physical AI System
Implements comprehensive safety monitoring and enforcement
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from std_msgs.msg import String, Bool, Float64
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
import numpy as np
import json
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

class SafetyLevel(Enum):
    """Safety level classification"""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

class SafetyViolation(Enum):
    """Types of safety violations"""
    COLLISION_IMMINENT = "collision_imminent"
    JOINT_LIMIT_EXCEEDED = "joint_limit_exceeded"
    VELOCITY_EXCEEDED = "velocity_exceeded"
    DISTANCE_VIOLATION = "distance_violation"
    HUMAN_PROXIMITY = "human_proximity"
    SYSTEM_ERROR = "system_error"

@dataclass
class SafetyMetrics:
    """Current safety metrics"""
    collision_risk: float = 0.0
    human_proximity: float = 0.0
    velocity_risk: float = 0.0
    joint_limit_risk: float = 0.0
    system_stability: float = 0.0
    overall_safety: float = 0.0
    timestamp: float = 0.0

class SafetyMonitoringNode(Node):
    def __init__(self):
        super().__init__('safety_monitoring_node')

        # Create subscribers for all safety-relevant data
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
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.robot_pose_sub = self.create_subscription(
            PointStamped,
            '/robot_pose',
            self.robot_pose_callback,
            10
        )

        # Create publishers for safety outputs
        self.safety_status_pub = self.create_publisher(
            String,
            '/safety/status',
            10
        )
        self.safety_metrics_pub = self.create_publisher(
            String,
            '/safety/metrics',
            10
        )
        self.emergency_stop_pub = self.create_publisher(
            Bool,
            '/emergency_stop',
            10
        )
        self.safety_violation_pub = self.create_publisher(
            String,
            '/safety/violation',
            10
        )
        self.safety_visualization_pub = self.create_publisher(
            MarkerArray,
            '/safety/visualization',
            10
        )

        # Safety parameters
        self.safety_enabled = True
        self.monitoring_frequency = 10.0  # Hz
        self.collision_distance_threshold = 0.5  # meters
        self.human_proximity_threshold = 1.0  # meters
        self.velocity_threshold = 0.5  # m/s
        self.angular_velocity_threshold = 1.0  # rad/s
        self.tilt_threshold = 0.3  # radians (about 17 degrees)

        # Robot state tracking
        self.current_cmd_vel = Twist()
        self.current_pose = PointStamped()
        self.current_velocity = np.array([0.0, 0.0, 0.0])
        self.current_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.current_imu_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        # Safety history
        self.safety_history = []
        self.max_history_length = 100

        # Create monitoring timer
        self.monitor_timer = self.create_timer(1.0/self.monitoring_frequency, self.monitor_safety)

        self.get_logger().info('Safety Monitoring Node Started')

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Find minimum distance to obstacles
        if msg.ranges:
            valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max and not np.isnan(r)]
            if valid_ranges:
                min_distance = min(valid_ranges)
                self.get_logger().debug(f'Min obstacle distance: {min_distance:.2f}m')

    def imu_callback(self, msg):
        """Process IMU data for orientation and stability"""
        self.current_imu_orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Check for dangerous tilt
        roll, pitch, _ = self.quaternion_to_euler(self.current_imu_orientation)
        tilt_magnitude = np.sqrt(roll**2 + pitch**2)

        if tilt_magnitude > self.tilt_threshold:
            self.get_logger().warn(f'Dangerous tilt detected: {tilt_magnitude:.2f} rad')

    def joint_state_callback(self, msg):
        """Process joint states for joint limit monitoring"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

    def odom_callback(self, msg):
        """Process odometry for velocity monitoring"""
        self.current_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        self.current_angular_velocity = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])

    def cmd_vel_callback(self, msg):
        """Monitor commanded velocities"""
        self.current_cmd_vel = msg

    def robot_pose_callback(self, msg):
        """Update robot pose"""
        self.current_pose = msg

    def monitor_safety(self):
        """Monitor safety metrics and enforce safety"""
        if not self.safety_enabled:
            return

        # Calculate safety metrics
        metrics = self.calculate_safety_metrics()

        # Add to history
        self.safety_history.append(metrics)
        if len(self.safety_history) > self.max_history_length:
            self.safety_history.pop(0)

        # Check for safety violations
        violations = self.check_safety_violations(metrics)

        # Determine overall safety level
        safety_level = self.determine_safety_level(metrics)

        # Enforce safety actions if needed
        self.enforce_safety_actions(safety_level, violations)

        # Publish safety metrics
        metrics_msg = String()
        metrics_msg.data = json.dumps({
            'metrics': metrics.__dict__,
            'safety_level': safety_level.value,
            'violations': [v.value for v in violations],
            'timestamp': time.time()
        })
        self.safety_metrics_pub.publish(metrics_msg)

        # Publish safety status
        status_msg = String()
        status_msg.data = safety_level.value
        self.safety_status_pub.publish(status_msg)

        # Log safety status
        if safety_level != SafetyLevel.SAFE:
            self.get_logger().warn(f'Safety level: {safety_level.value}, Violations: {[v.value for v in violations]}')

    def calculate_safety_metrics(self) -> SafetyMetrics:
        """Calculate current safety metrics"""
        metrics = SafetyMetrics()
        metrics.timestamp = time.time()

        # Collision risk based on laser scan
        metrics.collision_risk = self.calculate_collision_risk()

        # Human proximity (simulated - in real system this would use person detection)
        metrics.human_proximity = self.calculate_human_proximity_risk()

        # Velocity risk
        linear_speed = np.linalg.norm(self.current_velocity)
        angular_speed = np.linalg.norm(self.current_angular_velocity)
        metrics.velocity_risk = max(
            linear_speed / self.velocity_threshold,
            angular_speed / self.angular_velocity_threshold
        )

        # Joint limit risk (simulated - would check actual joint limits)
        metrics.joint_limit_risk = self.calculate_joint_limit_risk()

        # System stability (based on IMU data)
        roll, pitch, _ = self.quaternion_to_euler(self.current_imu_orientation)
        tilt_magnitude = np.sqrt(roll**2 + pitch**2)
        metrics.system_stability = 1.0 - min(tilt_magnitude / self.tilt_threshold, 1.0)

        # Overall safety score (weighted average)
        metrics.overall_safety = (
            0.3 * (1.0 - metrics.collision_risk) +
            0.2 * (1.0 - metrics.human_proximity) +
            0.2 * (1.0 - min(metrics.velocity_risk, 1.0)) +
            0.15 * (1.0 - min(metrics.joint_limit_risk, 1.0)) +
            0.15 * metrics.system_stability
        )

        return metrics

    def calculate_collision_risk(self) -> float:
        """Calculate collision risk based on laser scan"""
        # This would be implemented with actual laser processing
        # For simulation, return a value based on proximity to obstacles
        return 0.1  # Placeholder value

    def calculate_human_proximity_risk(self) -> float:
        """Calculate risk from human proximity"""
        # This would use person detection in a real system
        return 0.05  # Placeholder value

    def calculate_joint_limit_risk(self) -> float:
        """Calculate risk from joint limits"""
        # This would check actual joint positions against limits
        return 0.0  # Placeholder value

    def check_safety_violations(self, metrics: SafetyMetrics) -> List[SafetyViolation]:
        """Check for safety violations"""
        violations = []

        # Check collision risk
        if metrics.collision_risk > 0.8:
            violations.append(SafetyViolation.COLLISION_IMMINENT)

        # Check velocity limits
        linear_speed = np.linalg.norm(self.current_velocity)
        if linear_speed > self.velocity_threshold * 1.2:  # 20% over threshold
            violations.append(SafetyViolation.VELOCITY_EXCEEDED)

        # Check human proximity
        if metrics.human_proximity > 0.9:
            violations.append(SafetyViolation.HUMAN_PROXIMITY)

        return violations

    def determine_safety_level(self, metrics: SafetyMetrics) -> SafetyLevel:
        """Determine overall safety level"""
        if metrics.overall_safety < 0.3:
            return SafetyLevel.CRITICAL
        elif metrics.overall_safety < 0.6:
            return SafetyLevel.DANGER
        elif metrics.overall_safety < 0.8:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.SAFE

    def enforce_safety_actions(self, safety_level: SafetyLevel, violations: List[SafetyViolation]):
        """Enforce safety actions based on safety level"""
        if safety_level == SafetyLevel.CRITICAL or SafetyLevel.DANGER in [safety_level] or violations:
            # Emergency stop if critical danger or violations
            if safety_level == SafetyLevel.CRITICAL or any(
                v in [SafetyViolation.COLLISION_IMMINENT, SafetyViolation.SYSTEM_ERROR]
                for v in violations
            ):
                emergency_msg = Bool()
                emergency_msg.data = True
                self.emergency_stop_pub.publish(emergency_msg)
                self.get_logger().error('EMERGENCY STOP ACTIVATED')

        # Publish violations
        if violations:
            violation_msg = String()
            violation_msg.data = json.dumps({
                'violations': [v.value for v in violations],
                'timestamp': time.time()
            })
            self.safety_violation_pub.publish(violation_msg)

    def quaternion_to_euler(self, quat) -> Tuple[float, float, float]:
        """Convert quaternion to euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyMonitoringNode()

    try:
        rclpy.spin(safety_node)
    except KeyboardInterrupt:
        pass
    finally:
        safety_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Safety Audit Node

Now let's create a safety audit node that performs systematic safety checks:

```python
#!/usr/bin/env python3
"""
Safety Audit Node for Physical AI System
Implements systematic safety audits and compliance checking
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu, JointState
import json
import time
from typing import Dict, List
from datetime import datetime
import threading

class SafetyAuditNode(Node):
    def __init__(self):
        super().__init__('safety_audit_node')

        # Create subscribers for system status
        self.system_status_sub = self.create_subscription(
            String,
            '/system/status',
            self.system_status_callback,
            10
        )
        self.safety_status_sub = self.create_subscription(
            String,
            '/safety/status',
            self.safety_status_callback,
            10
        )
        self.safety_violation_sub = self.create_subscription(
            String,
            '/safety/violation',
            self.safety_violation_callback,
            10
        )
        self.emergency_stop_sub = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_stop_callback,
            10
        )

        # Create publishers for audit outputs
        self.audit_report_pub = self.create_publisher(
            String,
            '/safety/audit_report',
            10
        )
        self.compliance_status_pub = self.create_publisher(
            String,
            '/safety/compliance_status',
            10
        )
        self.audit_log_pub = self.create_publisher(
            String,
            '/safety/audit_log',
            10
        )

        # Audit parameters
        self.audit_enabled = True
        self.audit_frequency = 30.0  # seconds
        self.compliance_standards = [
            "ISO 13482 (Service Robots)",
            "ISO 12100 (Safety of Machinery)",
            "IEEE P7000 (Ethical Design in AI Systems)"
        ]

        # Audit data
        self.violation_history = []
        self.emergency_stop_history = []
        self.compliance_score = 100.0
        self.audit_results = {}

        # Create audit timer
        self.audit_timer = self.create_timer(self.audit_frequency, self.perform_safety_audit)

        self.get_logger().info('Safety Audit Node Started')

    def system_status_callback(self, msg):
        """Process system status for audit"""
        try:
            status_data = json.loads(msg.data)
            # Log system status for audit trail
            self.log_audit_event("system_status", status_data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in system status message')

    def safety_status_callback(self, msg):
        """Process safety status for audit"""
        safety_level = msg.data
        self.log_audit_event("safety_status", {"level": safety_level, "timestamp": time.time()})

    def safety_violation_callback(self, msg):
        """Process safety violations for audit"""
        try:
            violation_data = json.loads(msg.data)
            self.violation_history.append(violation_data)

            # Log violation for audit trail
            self.log_audit_event("safety_violation", violation_data)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in safety violation message')

    def emergency_stop_callback(self, msg):
        """Process emergency stops for audit"""
        if msg.data:
            emergency_event = {
                'timestamp': time.time(),
                'reason': 'manual_emergency_stop'
            }
            self.emergency_stop_history.append(emergency_event)

            # Log emergency stop for audit trail
            self.log_audit_event("emergency_stop", emergency_event)

    def perform_safety_audit(self):
        """Perform comprehensive safety audit"""
        if not self.audit_enabled:
            return

        self.get_logger().info('Performing safety audit...')

        # Conduct various safety checks
        audit_results = {
            'timestamp': time.time(),
            'checks_performed': [],
            'compliance_status': {},
            'risk_assessment': {},
            'recommendations': []
        }

        # Check 1: System uptime and stability
        uptime_check = self.check_system_uptime()
        audit_results['checks_performed'].append(uptime_check)

        # Check 2: Safety violation frequency
        violation_check = self.check_violation_frequency()
        audit_results['checks_performed'].append(violation_check)

        # Check 3: Emergency stop usage
        emergency_check = self.check_emergency_stop_usage()
        audit_results['checks_performed'].append(emergency_check)

        # Check 4: Compliance with standards
        compliance_check = self.check_compliance_standards()
        audit_results['checks_performed'].append(compliance_check)

        # Calculate compliance score
        self.compliance_score = self.calculate_compliance_score(audit_results)

        # Generate recommendations
        audit_results['recommendations'] = self.generate_recommendations(audit_results)

        # Update audit results
        self.audit_results = audit_results

        # Publish audit report
        report_msg = String()
        report_msg.data = json.dumps(audit_results, indent=2)
        self.audit_report_pub.publish(report_msg)

        # Publish compliance status
        compliance_msg = String()
        compliance_msg.data = json.dumps({
            'score': self.compliance_score,
            'status': 'compliant' if self.compliance_score >= 80 else 'non_compliant',
            'timestamp': time.time()
        })
        self.compliance_status_pub.publish(compliance_msg)

        self.get_logger().info(f'Safety audit completed with compliance score: {self.compliance_score:.2f}')

    def check_system_uptime(self) -> Dict:
        """Check system uptime and stability"""
        check_result = {
            'check_name': 'system_uptime',
            'status': 'pass',
            'details': 'System has been stable',
            'timestamp': time.time()
        }

        # In a real system, this would check actual uptime metrics
        return check_result

    def check_violation_frequency(self) -> Dict:
        """Check safety violation frequency"""
        recent_violations = [v for v in self.violation_history
                            if time.time() - v.get('timestamp', 0) < 3600]  # Last hour

        if len(recent_violations) > 5:  # More than 5 violations per hour
            status = 'fail'
            details = f'High violation frequency: {len(recent_violations)} violations in last hour'
        elif len(recent_violations) > 0:
            status = 'warning'
            details = f'Moderate violation frequency: {len(recent_violations)} violations in last hour'
        else:
            status = 'pass'
            details = 'No violations in last hour'

        return {
            'check_name': 'violation_frequency',
            'status': status,
            'details': details,
            'timestamp': time.time()
        }

    def check_emergency_stop_usage(self) -> Dict:
        """Check emergency stop usage patterns"""
        recent_stops = [s for s in self.emergency_stop_history
                       if time.time() - s.get('timestamp', 0) < 3600]  # Last hour

        if len(recent_stops) > 3:  # More than 3 emergency stops per hour
            status = 'fail'
            details = f'Frequent emergency stops: {len(recent_stops)} in last hour'
        elif len(recent_stops) > 0:
            status = 'warning'
            details = f'Emergency stops used: {len(recent_stops)} times in last hour'
        else:
            status = 'pass'
            details = 'No emergency stops in last hour'

        return {
            'check_name': 'emergency_stop_usage',
            'status': status,
            'details': details,
            'timestamp': time.time()
        }

    def check_compliance_standards(self) -> Dict:
        """Check compliance with safety standards"""
        # This would implement detailed compliance checking against standards
        return {
            'check_name': 'compliance_standards',
            'status': 'pass',
            'details': f'Following standards: {", ".join(self.compliance_standards)}',
            'timestamp': time.time()
        }

    def calculate_compliance_score(self, audit_results: Dict) -> float:
        """Calculate overall compliance score"""
        total_points = 0
        max_points = 0

        for check in audit_results['checks_performed']:
            max_points += 100  # Max 100 points per check

            if check['status'] == 'pass':
                total_points += 100
            elif check['status'] == 'warning':
                total_points += 70
            else:  # fail
                total_points += 30

        return (total_points / max_points) * 100 if max_points > 0 else 100

    def generate_recommendations(self, audit_results: Dict) -> List[str]:
        """Generate safety recommendations based on audit results"""
        recommendations = []

        for check in audit_results['checks_performed']:
            if check['status'] == 'fail':
                if check['check_name'] == 'violation_frequency':
                    recommendations.append("Investigate causes of frequent safety violations")
                elif check['check_name'] == 'emergency_stop_usage':
                    recommendations.append("Review operational procedures causing emergency stops")
            elif check['status'] == 'warning':
                if check['check_name'] == 'violation_frequency':
                    recommendations.append("Monitor safety violation trends")
                elif check['check_name'] == 'emergency_stop_usage':
                    recommendations.append("Review operational procedures")

        if not recommendations:
            recommendations.append("System is operating safely within parameters")

        return recommendations

    def log_audit_event(self, event_type: str, data: Dict):
        """Log audit events for traceability"""
        log_entry = {
            'event_type': event_type,
            'data': data,
            'timestamp': time.time(),
            'node': self.get_name()
        }

        # Publish to audit log
        log_msg = String()
        log_msg.data = json.dumps(log_entry)
        self.audit_log_pub.publish(log_msg)

def main(args=None):
    rclpy.init(args=args)
    audit_node = SafetyAuditNode()

    try:
        rclpy.spin(audit_node)
    except KeyboardInterrupt:
        pass
    finally:
        audit_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Safety Policy Enforcement Node

Let's create a node that enforces safety policies:

```python
#!/usr/bin/env python3
"""
Safety Policy Enforcement Node for Physical AI System
Enforces safety policies and ethical guidelines
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import json
import time
from typing import Dict, List
import numpy as np

class SafetyPolicyEnforcementNode(Node):
    def __init__(self):
        super().__init__('safety_policy_enforcement_node')

        # Create subscribers
        self.voice_command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        self.system_command_sub = self.create_subscription(
            String,
            '/system/command',
            self.system_command_callback,
            10
        )

        # Create publishers
        self.filtered_cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel_filtered',
            10
        )
        self.policy_violation_pub = self.create_publisher(
            String,
            '/safety/policy_violation',
            10
        )
        self.ethical_decision_pub = self.create_publisher(
            String,
            '/safety/ethical_decision',
            10
        )

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Safety policies
        self.safety_policies = {
            'human_proximity': {
                'enabled': True,
                'min_distance': 0.5,  # meters
                'action': 'stop'
            },
            'command_filtering': {
                'enabled': True,
                'forbidden_commands': ['attack', 'harm', 'destroy', 'damage'],
                'action': 'reject'
            },
            'velocity_limiting': {
                'enabled': True,
                'linear_max': 0.5,  # m/s
                'angular_max': 1.0,  # rad/s
                'action': 'limit'
            },
            'ethical_guidelines': {
                'enabled': True,
                'principles': [
                    'respect_human_dignity',
                    'prioritize_human_safety',
                    'maintain_transparency'
                ],
                'action': 'evaluate'
            }
        }

        # Policy state
        self.last_valid_cmd = Twist()
        self.human_detected = False
        self.human_distance = float('inf')

        self.get_logger().info('Safety Policy Enforcement Node Started')

    def voice_command_callback(self, msg):
        """Filter voice commands against safety policies"""
        command = msg.data.lower().strip()

        # Check command against forbidden commands
        if self.safety_policies['command_filtering']['enabled']:
            forbidden = self.safety_policies['command_filtering']['forbidden_commands']
            if any(word in command for word in forbidden):
                self.log_policy_violation('command_filtering', command)
                return  # Reject the command

        # Forward filtered command
        self.get_logger().debug(f'Command passed safety filter: {command}')

    def cmd_vel_callback(self, msg):
        """Enforce safety policies on velocity commands"""
        filtered_cmd = Twist()

        if self.safety_policies['velocity_limiting']['enabled']:
            # Apply velocity limits
            max_linear = self.safety_policies['velocity_limiting']['linear_max']
            max_angular = self.safety_policies['velocity_limiting']['angular_max']

            filtered_cmd.linear.x = max(-max_linear, min(max_linear, msg.linear.x))
            filtered_cmd.linear.y = max(-max_linear, min(max_linear, msg.linear.y))
            filtered_cmd.linear.z = max(-max_linear, min(max_linear, msg.linear.z))
            filtered_cmd.angular.x = max(-max_angular, min(max_angular, msg.angular.x))
            filtered_cmd.angular.y = max(-max_angular, min(max_angular, msg.angular.y))
            filtered_cmd.angular.z = max(-max_angular, min(max_angular, msg.angular.z))
        else:
            filtered_cmd = msg

        # Check human proximity before allowing movement
        if self.human_detected and self.human_distance < self.safety_policies['human_proximity']['min_distance']:
            # Stop movement if too close to human
            filtered_cmd.linear.x = 0.0
            filtered_cmd.angular.z = 0.0
            self.log_policy_violation('human_proximity', f'Too close to human: {self.human_distance:.2f}m')

        # Publish filtered command
        self.filtered_cmd_pub.publish(filtered_cmd)

    def image_callback(self, msg):
        """Process images for human detection (simplified)"""
        # In a real system, this would use person detection
        # For simulation, we'll assume human detection based on other sensors
        pass

    def laser_callback(self, msg):
        """Process laser scan for human proximity detection"""
        # Find minimum distance to obstacles (assuming humans are obstacles)
        if msg.ranges:
            valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max and not np.isnan(r)]
            if valid_ranges:
                min_distance = min(valid_ranges)
                self.human_distance = min_distance
                self.human_detected = min_distance < 2.0  # Human within 2m

    def system_command_callback(self, msg):
        """Process system commands against ethical guidelines"""
        try:
            command_data = json.loads(msg.data)
            command = command_data.get('command', '').lower()

            # Check ethical compliance
            if self.safety_policies['ethical_guidelines']['enabled']:
                if not self.is_ethically_compliant(command, command_data):
                    self.log_policy_violation('ethical_guidelines', command)
                    return  # Reject unethical command

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in system command')

    def is_ethically_compliant(self, command: str, command_data: Dict) -> bool:
        """Check if command is ethically compliant"""
        # Check against ethical principles
        principles = self.safety_policies['ethical_guidelines']['principles']

        # Example: Check if command involves harming humans
        if any(word in command for word in ['harm', 'injure', 'attack', 'damage']):
            return False

        # Example: Check if command respects human dignity
        if any(word in command for word in ['disrespect', 'ignore', 'disregard']):
            return False

        return True

    def log_policy_violation(self, policy_type: str, details: str):
        """Log policy violations"""
        violation_data = {
            'policy_type': policy_type,
            'details': details,
            'timestamp': time.time()
        }

        violation_msg = String()
        violation_msg.data = json.dumps(violation_data)
        self.policy_violation_pub.publish(violation_msg)

        self.get_logger().warn(f'Policy violation: {policy_type} - {details}')

def main(args=None):
    rclpy.init(args=args)
    policy_node = SafetyPolicyEnforcementNode()

    try:
        rclpy.spin(policy_node)
    except KeyboardInterrupt:
        pass
    finally:
        policy_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Safety Configuration

Create a configuration file for safety systems:

```yaml
# safety_config.yaml
safety_system:
  # Safety monitoring parameters
  monitoring:
    enabled: true
    frequency: 10.0  # Hz
    collision_threshold: 0.5  # meters
    human_proximity_threshold: 1.0  # meters
    velocity_thresholds:
      linear: 0.5  # m/s
      angular: 1.0  # rad/s
    tilt_threshold: 0.3  # radians
    emergency_stop_timeout: 5.0  # seconds

  # Safety policies
  policies:
    human_proximity:
      enabled: true
      min_distance: 0.5  # meters
      action: "stop"
    command_filtering:
      enabled: true
      forbidden_commands:
        - "attack"
        - "harm"
        - "destroy"
        - "damage"
        - "injure"
      action: "reject"
    velocity_limiting:
      enabled: true
      linear_max: 0.5  # m/s
      angular_max: 1.0  # rad/s
      action: "limit"
    ethical_guidelines:
      enabled: true
      principles:
        - "respect_human_dignity"
        - "prioritize_human_safety"
        - "maintain_transparency"
      action: "evaluate"

  # Audit parameters
  audit:
    frequency: 30.0  # seconds
    compliance_standards:
      - "ISO 13482 (Service Robots)"
      - "ISO 12100 (Safety of Machinery)"
      - "IEEE P7000 (Ethical Design in AI Systems)"
    violation_thresholds:
      violations_per_hour: 5
      emergency_stops_per_hour: 3
    success_criteria:
      compliance_score_min: 80.0  # percent

  # Emergency procedures
  emergency:
    enabled: true
    stop_distance: 0.3  # meters
    stop_time: 5.0  # seconds
    recovery_procedures:
      - "full_system_stop"
      - "safety_check_sequence"
      - "manual_override_required"

  # Performance parameters
  performance:
    max_processing_time: 0.1  # seconds per cycle
    memory_limit: 1024  # MB
    safety_priority: 99  # High priority for safety threads

  # Logging parameters
  logging:
    level: "warn"  # Log safety events
    log_file: "/var/log/safety_system.log"
    rotation_size: "10MB"
    backup_count: 10
    detailed_components:
      - "safety_monitoring"
      - "safety_audit"
      - "policy_enforcement"
```

## Safety Launch File

Create a launch file for the safety system:

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
    config_file = LaunchConfiguration('config_file', default='safety_config.yaml')

    # Safety monitoring node
    safety_monitoring_node = Node(
        package='physical_ai_safety',
        executable='safety_monitoring_node',
        name='safety_monitoring_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_safety'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    # Safety audit node
    safety_audit_node = Node(
        package='physical_ai_safety',
        executable='safety_audit_node',
        name='safety_audit_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_safety'),
                'config',
                config_file
            ])
        ],
        output='screen'
    )

    # Safety policy enforcement node
    safety_policy_node = Node(
        package='physical_ai_safety',
        executable='safety_policy_enforcement_node',
        name='safety_policy_enforcement_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('physical_ai_safety'),
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
            default_value='safety_config.yaml',
            description='Configuration file for safety system'
        ),
        safety_monitoring_node,
        safety_audit_node,
        safety_policy_node
    ])
```

## Quality Assurance for Safety Systems

### Performance Metrics
- **Safety Compliance Rate**: Percentage of operations that meet safety requirements
- **Violation Detection Rate**: Percentage of safety violations correctly identified
- **Emergency Response Time**: Time to respond to safety-critical situations
- **Audit Compliance Score**: Overall compliance with safety standards

### Safety Considerations
Following our constitution's "Safety is Intelligence" principle:

1. **Proactive Safety**: Anticipate and prevent safety issues
2. **Fail-Safe Design**: Ensure safe state when systems fail
3. **Human-Centered Design**: Prioritize human safety and dignity
4. **Continuous Monitoring**: Maintain constant safety oversight

### Testing Scenarios
1. **Safety Boundary Testing**: Test system at safety limits
2. **Emergency Procedures**: Test emergency stop and recovery
3. **Policy Enforcement**: Test command filtering and policy compliance
4. **Audit Verification**: Test compliance checking and reporting

## Looking Forward

With our comprehensive safety and audit systems in place, the final chapter will focus on creating a personal roadmap for continued development and learning in Physical AI.

[Continue to Chapter 20: Personal Roadmap](./chapter-20-personal-roadmap.md)