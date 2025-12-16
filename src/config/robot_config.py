"""
Environment Configuration Management

This module manages configuration parameters for the robot system.
"""

import json
import os
from typing import Dict, Any, Optional


class RobotConfig:
    """Manages robot configuration parameters"""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with optional config file"""
        self.config_file = config_file or self._find_default_config()
        self.config_data = self._load_config()

    def _find_default_config(self) -> str:
        """Find the default configuration file"""
        # Look for config in common locations
        possible_paths = [
            "config/robot.json",
            "src/config/robot.json",
            "robot_config.json",
            "/etc/physical_ai/robot.json"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # If no config file exists, return default path
        return "src/config/robot.json"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_file}: {e}")
                return self._get_default_config()
        else:
            print(f"Config file {self.config_file} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            "robot": {
                "name": "PhysicalAI_Robot",
                "model": "Humanoid_V1",
                "serial_number": "PAI001",
                "max_linear_velocity": 1.0,
                "max_angular_velocity": 1.0,
                "wheel_radius": 0.1,
                "wheel_separation": 0.5
            },
            "sensors": {
                "camera_enabled": True,
                "lidar_enabled": True,
                "imu_enabled": True,
                "microphone_enabled": True
            },
            "navigation": {
                "planner": "teb_local_planner",
                "global_frame": "map",
                "robot_frame": "base_link",
                "goal_tolerance": 0.1,
                "yaw_tolerance": 0.1
            },
            "voice": {
                "language": "en-US",
                "sensitivity": 0.5,
                "response_delay": 1.0
            },
            "safety": {
                "max_speed": 1.0,
                "safety_distance": 0.5,
                "emergency_stop_timeout": 5.0
            },
            "system": {
                "log_level": "INFO",
                "debug_mode": False,
                "simulation_mode": True
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'robot.name')"""
        keys = key.split('.')
        value = self.config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation"""
        keys = key.split('.')
        config = self.config_data

        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        path = file_path or self.config_file
        try:
            with open(path, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            print(f"Configuration saved to {path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from a dictionary"""
        def deep_update(target: Dict, source: Dict):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_update(target[key], value)
                else:
                    target[key] = value

        deep_update(self.config_data, updates)

    def get_robot_name(self) -> str:
        """Get the robot's name"""
        return self.get("robot.name", "Unknown_Robot")

    def get_max_linear_velocity(self) -> float:
        """Get maximum linear velocity"""
        return self.get("robot.max_linear_velocity", 1.0)

    def get_max_angular_velocity(self) -> float:
        """Get maximum angular velocity"""
        return self.get("robot.max_angular_velocity", 1.0)

    def is_simulation_mode(self) -> bool:
        """Check if running in simulation mode"""
        return self.get("system.simulation_mode", True)

    def get_safety_distance(self) -> float:
        """Get safety distance threshold"""
        return self.get("safety.safety_distance", 0.5)


# Global configuration instance
_robot_config = None


def get_config() -> RobotConfig:
    """Get the global configuration instance"""
    global _robot_config
    if _robot_config is None:
        _robot_config = RobotConfig()
    return _robot_config


def init_config(config_file: Optional[str] = None) -> RobotConfig:
    """Initialize the global configuration with a specific file"""
    global _robot_config
    _robot_config = RobotConfig(config_file)
    return _robot_config