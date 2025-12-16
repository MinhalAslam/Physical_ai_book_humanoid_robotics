"""
Base Data Models for Physical AI System

This module defines the core data models used throughout the system.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import time
from enum import Enum


@dataclass
class RobotState:
    """Represents the current state of the robot"""
    timestamp: float
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    orientation_x: float = 0.0
    orientation_y: float = 0.0
    orientation_z: float = 0.0
    orientation_w: float = 1.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    battery_level: float = 1.0
    temperature: float = 25.0
    safety_status: str = "SAFE"


@dataclass
class VoiceCommand:
    """Represents a voice command with processing information"""
    timestamp: float
    raw_text: str
    processed_text: str
    confidence: float
    intent: str
    entities: Dict[str, Any]
    source: str = "microphone"


@dataclass
class DetectedObject:
    """Represents an object detected by the perception system"""
    id: str
    type: str
    confidence: float
    position_x: float
    position_y: float
    position_z: float
    size_x: float = 0.0
    size_y: float = 0.0
    size_z: float = 0.0
    color: str = "unknown"


@dataclass
class NavigationGoal:
    """Represents a navigation goal for the robot"""
    id: str
    position_x: float
    position_y: float
    position_z: float
    orientation_x: float = 0.0
    orientation_y: float = 0.0
    orientation_z: float = 0.0
    orientation_w: float = 1.0
    frame_id: str = "map"
    tolerance: float = 0.1


@dataclass
class TaskStep:
    """Represents a single step in a task sequence"""
    id: str
    action: str
    parameters: Dict[str, Any]
    priority: int = 1
    requires_confirmation: bool = False
    estimated_duration: float = 0.0


@dataclass
class SensorData:
    """Represents sensor data from various sources"""
    timestamp: float
    sensor_type: str
    data: Dict[str, Any]
    frame_id: str = "base_link"


@dataclass
class SystemStatus:
    """Represents overall system status"""
    timestamp: float
    system_name: str
    status: str  # "OPERATIONAL", "WARNING", "ERROR", "EMERGENCY"
    components: Dict[str, str]  # Component name -> status
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0


@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    timestamp: float
    level: str  # "INFO", "WARNING", "DANGER", "EMERGENCY"
    description: str
    severity: int  # 1-10 scale
    action_taken: str


class RobotMode(Enum):
    """Enumeration of robot operational modes"""
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    MANIPULATING = "MANIPULATING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    EMERGENCY = "EMERGENCY"
    CHARGING = "CHARGING"


# Utility functions for data model operations
def create_robot_state_from_pose(pose, twist=None, battery=1.0):
    """Create a RobotState from pose and optional twist data"""
    state = RobotState(
        timestamp=time.time(),
        position_x=pose.position.x,
        position_y=pose.position.y,
        position_z=pose.position.z,
        orientation_x=pose.orientation.x,
        orientation_y=pose.orientation.y,
        orientation_z=pose.orientation.z,
        orientation_w=pose.orientation.w,
        battery_level=battery
    )

    if twist:
        state.linear_velocity = twist.linear.x  # Simplified
        state.angular_velocity = twist.angular.z  # Simplified

    return state


def validate_navigation_goal(goal: NavigationGoal) -> bool:
    """Validate that a navigation goal is reasonable"""
    # Check for reasonable coordinates
    if abs(goal.position_x) > 1000 or abs(goal.position_y) > 1000:
        return False

    # Check tolerance is positive
    if goal.tolerance <= 0:
        return False

    return True