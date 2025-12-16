#!/usr/bin/env python3
"""
Safety Audit System

This module implements comprehensive safety monitoring, auditing,
and policy enforcement for the Physical AI system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
import time
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGER = "DANGER"
    EMERGENCY = "EMERGENCY"


@dataclass
class SafetyViolation:
    """Represents a safety violation with details"""
    timestamp: float
    level: SafetyLevel
    description: str
    severity: int  # 1-10 scale
    action_taken: str


class SafetyAuditNode(Node):
    def __init__(self):
        super().__init__('safety_audit')

        # Publishers
        self.safety_status_pub = self.create_publisher(String, '/safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.safety_violation_pub = self.create_publisher(String, '/safety_violation', 10)
        self.safety_audit_pub = self.create_publisher(String, '/safety_audit_report', 10)

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.command_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.pose_callback, 10)

        # Timer for safety monitoring
        self.safety_timer = self.create_timer(0.1, self.safety_check)  # 10Hz safety checks

        self.get_logger().info('Safety Audit Node Started')

        # Safety state
        self.current_pose = None
        self.last_command = None
        self.last_scan = None
        self.safety_violations = []
        self.safety_level = SafetyLevel.SAFE
        self.emergency_stop_active = False
        self.safety_policies = self.initialize_safety_policies()
        self.robot_speed_limit = 1.0  # m/s
        self.min_obstacle_distance = 0.5  # meters
        self.max_angular_velocity = 1.0  # rad/s

    def initialize_safety_policies(self) -> Dict:
        """Initialize safety policies and constraints"""
        return {
            'speed_limits': {
                'linear_max': 1.0,      # m/s
                'linear_min': -0.5,     # m/s (reverse)
                'angular_max': 1.0,     # rad/s
                'angular_min': -1.0     # rad/s
            },
            'proximity_limits': {
                'collision_threshold': 0.3,  # meters
                'warning_threshold': 0.8,    # meters
                'stop_threshold': 0.5        # meters
            },
            'operational_limits': {
                'max_operation_time': 3600,  # seconds (1 hour)
                'min_battery_level': 0.1,    # 10%
                'max_temperature': 60.0      # degrees Celsius
            },
            'zone_restrictions': {
                'forbidden_zones': [],       # List of forbidden areas
                'speed_limited_zones': []    # Areas requiring speed limits
            }
        }

    def command_callback(self, msg):
        """Monitor command safety"""
        self.last_command = msg

        # Check speed limits
        violations = []

        if abs(msg.linear.x) > self.safety_policies['speed_limits']['linear_max']:
            violations.append(SafetyViolation(
                timestamp=time.time(),
                level=SafetyLevel.WARNING,
                description=f"Linear speed too high: {msg.linear.x:.2f} m/s",
                severity=6,
                action_taken="SPEED_LIMITED"
            ))

        if abs(msg.angular.z) > self.safety_policies['speed_limits']['angular_max']:
            violations.append(SafetyViolation(
                timestamp=time.time(),
                level=SafetyLevel.WARNING,
                description=f"Angular speed too high: {msg.angular.z:.2f} rad/s",
                severity=5,
                action_taken="SPEED_LIMITED"
            ))

        # Log violations
        for violation in violations:
            self.log_safety_violation(violation)

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        self.last_scan = msg

        # Check for nearby obstacles
        if msg.ranges:
            min_distance = min([r for r in msg.ranges if r > 0 and not math.isinf(r)], default=float('inf'))

            if min_distance < self.safety_policies['proximity_limits']['collision_threshold']:
                violation = SafetyViolation(
                    timestamp=time.time(),
                    level=SafetyLevel.EMERGENCY,
                    description=f"Collision imminent: obstacle at {min_distance:.2f}m",
                    severity=10,
                    action_taken="EMERGENCY_STOP"
                )
                self.log_safety_violation(violation)
                self.trigger_emergency_stop()
            elif min_distance < self.safety_policies['proximity_limits']['stop_threshold']:
                violation = SafetyViolation(
                    timestamp=time.time(),
                    level=SafetyLevel.DANGER,
                    description=f"Obstacle too close: {min_distance:.2f}m",
                    severity=8,
                    action_taken="STOP_COMMAND"
                )
                self.log_safety_violation(violation)
            elif min_distance < self.safety_policies['proximity_limits']['warning_threshold']:
                violation = SafetyViolation(
                    timestamp=time.time(),
                    level=SafetyLevel.WARNING,
                    description=f"Obstacle nearby: {min_distance:.2f}m",
                    severity=4,
                    action_taken="SLOW_DOWN"
                )
                self.log_safety_violation(violation)

    def pose_callback(self, msg):
        """Monitor robot pose for safety"""
        self.current_pose = msg

        # Check if in forbidden zones (simplified check)
        # In a real system, this would check against geofences
        if self.is_in_forbidden_zone(msg):
            violation = SafetyViolation(
                timestamp=time.time(),
                level=SafetyLevel.DANGER,
                description="Robot in forbidden zone",
                severity=9,
                action_taken="RETURN_TO_SAFE_ZONE"
            )
            self.log_safety_violation(violation)

    def is_in_forbidden_zone(self, pose) -> bool:
        """Check if pose is in any forbidden zone"""
        # Simplified implementation - in reality, this would check against geofences
        return False

    def safety_check(self):
        """Perform comprehensive safety check"""
        current_time = time.time()

        # Update safety status
        status = {
            'timestamp': current_time,
            'safety_level': self.safety_level.value,
            'violations_count': len(self.safety_violations),
            'emergency_stop': self.emergency_stop_active,
            'last_violation': self.safety_violations[-1].description if self.safety_violations else "None"
        }

        status_msg = String()
        status_msg.data = str(status)
        self.safety_status_pub.publish(status_msg)

        # Periodic safety audit (every 10 seconds)
        if int(current_time) % 10 == 0:
            self.perform_safety_audit()

    def log_safety_violation(self, violation: SafetyViolation):
        """Log a safety violation"""
        self.safety_violations.append(violation)

        # Update safety level based on violation
        if violation.level == SafetyLevel.EMERGENCY:
            self.safety_level = SafetyLevel.EMERGENCY
        elif violation.level == SafetyLevel.DANGER and self.safety_level != SafetyLevel.EMERGENCY:
            self.safety_level = SafetyLevel.DANGER
        elif violation.level == SafetyLevel.WARNING and self.safety_level not in [SafetyLevel.EMERGENCY, SafetyLevel.DANGER]:
            self.safety_level = SafetyLevel.WARNING

        # Publish violation
        violation_msg = String()
        violation_msg.data = f"{violation.level.value}: {violation.description} (Severity: {violation.severity})"
        self.safety_violation_pub.publish(violation_msg)

        self.get_logger().warn(f"Safety Violation: {violation_msg.data}")

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            stop_msg = Bool()
            stop_msg.data = True
            self.emergency_stop_pub.publish(stop_msg)
            self.get_logger().error("EMERGENCY STOP ACTIVATED")

    def perform_safety_audit(self):
        """Perform comprehensive safety audit"""
        audit_report = {
            'timestamp': time.time(),
            'audit_type': 'COMPREHENSIVE_SAFETY_AUDIT',
            'system_status': 'OPERATIONAL' if not self.emergency_stop_active else 'EMERGENCY_STOP',
            'safety_level': self.safety_level.value,
            'total_violations': len(self.safety_violations),
            'recent_violations': [],
            'safety_metrics': {
                'max_violation_severity': max([v.severity for v in self.safety_violations], default=0) if self.safety_violations else 0,
                'violation_frequency': len([v for v in self.safety_violations if time.time() - v.timestamp < 60]) if self.safety_violations else 0,  # Last minute
                'compliance_rate': self.calculate_compliance_rate()
            },
            'recommendations': self.generate_recommendations(),
            'policies_enforced': list(self.safety_policies.keys())
        }

        # Include recent violations
        recent_violations = [v for v in self.safety_violations if time.time() - v.timestamp < 300]  # Last 5 minutes
        for violation in recent_violations[-5:]:  # Last 5 violations
            audit_report['recent_violations'].append({
                'timestamp': violation.timestamp,
                'level': violation.level.value,
                'description': violation.description,
                'severity': violation.severity
            })

        audit_msg = String()
        audit_msg.data = str(audit_report)
        self.safety_audit_pub.publish(audit_msg)

        self.get_logger().info(f'Safety Audit: {audit_report["safety_level"]} - {audit_report["total_violations"]} violations')

    def calculate_compliance_rate(self) -> float:
        """Calculate safety compliance rate"""
        if not self.safety_violations:
            return 1.0  # 100% compliant

        total_violations = len(self.safety_violations)
        severe_violations = len([v for v in self.safety_violations if v.severity >= 7])

        # Compliance decreases with more severe violations
        compliance = max(0.0, 1.0 - (severe_violations * 0.1 + (total_violations - severe_violations) * 0.01))
        return compliance

    def generate_recommendations(self) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []

        if self.safety_level in [SafetyLevel.DANGER, SafetyLevel.EMERGENCY]:
            recommendations.append("IMMEDIATE: Review operational procedures")
            recommendations.append("IMMEDIATE: Check sensor calibration")

        if len(self.safety_violations) > 10:
            recommendations.append("Review and update safety policies")
            recommendations.append("Consider additional safety training")

        if not recommendations:
            recommendations.append("Continue current safety monitoring")

        return recommendations

    def reset_emergency_stop(self):
        """Reset emergency stop condition"""
        self.emergency_stop_active = False
        stop_msg = Bool()
        stop_msg.data = False
        self.emergency_stop_pub.publish(stop_msg)
        self.safety_level = SafetyLevel.SAFE
        self.get_logger().info("Emergency stop reset, safety level restored to SAFE")


def main(args=None):
    rclpy.init(args=args)

    node = SafetyAuditNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Safety Audit Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()