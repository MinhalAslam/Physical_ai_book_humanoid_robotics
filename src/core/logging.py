#!/usr/bin/env python3
"""
Error Handling and Logging Infrastructure

This module implements the error handling and logging infrastructure
for the Physical AI system.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional


class LoggingSystem(Node):
    def __init__(self):
        super().__init__('logging_system')

        # Publisher for log messages
        self.log_pub = self.create_publisher(String, '/system_log', 10)

        # Setup Python logging
        self.setup_logging()

        self.get_logger().info('Logging System Node Started')

    def setup_logging(self):
        """Setup the logging configuration"""
        # Configure the logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                # In a real system, you might add file handlers here
            ]
        )

        self.logger = logging.getLogger('PhysicalAI')
        self.logger.setLevel(logging.INFO)

    def log_info(self, message: str, source: str = "system"):
        """Log an info message"""
        log_entry = self.create_log_entry("INFO", message, source)
        self.publish_log(log_entry)

    def log_warning(self, message: str, source: str = "system"):
        """Log a warning message"""
        log_entry = self.create_log_entry("WARNING", message, source)
        self.publish_log(log_entry)

    def log_error(self, message: str, source: str = "system", exception: Optional[Exception] = None):
        """Log an error message"""
        if exception:
            message += f" | Exception: {str(exception)}"
        log_entry = self.create_log_entry("ERROR", message, source)
        self.publish_log(log_entry)

    def log_debug(self, message: str, source: str = "system"):
        """Log a debug message"""
        log_entry = self.create_log_entry("DEBUG", message, source)
        self.publish_log(log_entry)

    def create_log_entry(self, level: str, message: str, source: str) -> Dict[str, Any]:
        """Create a structured log entry"""
        return {
            "timestamp": time.time(),
            "iso_timestamp": datetime.fromtimestamp(time.time()).isoformat(),
            "level": level,
            "message": message,
            "source": source,
            "node_name": self.get_name(),
            "severity": self.get_severity_number(level)
        }

    def get_severity_number(self, level: str) -> int:
        """Convert log level to severity number"""
        severity_map = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        return severity_map.get(level, 1)

    def publish_log(self, log_entry: Dict[str, Any]):
        """Publish log entry to ROS topic"""
        log_msg = String()
        log_msg.data = json.dumps(log_entry, indent=2)
        self.log_pub.publish(log_msg)

        # Also log to standard ROS logging
        level = log_entry["level"]
        message = log_entry["message"]

        if level == "ERROR":
            self.get_logger().error(message)
        elif level == "WARNING":
            self.get_logger().warn(message)
        elif level == "DEBUG":
            self.get_logger().debug(message)
        else:
            self.get_logger().info(message)

    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log a specific system event"""
        message = f"System Event: {event_type}"
        if details:
            message += f" | Details: {details}"

        self.log_info(message, "event_logger")

    def log_error_with_context(self, error_msg: str, context: Dict[str, Any]):
        """Log an error with additional context information"""
        full_message = f"{error_msg} | Context: {context}"
        self.log_error(full_message, "error_context")


def main(args=None):
    rclpy.init(args=args)

    node = LoggingSystem()

    try:
        # Example logging
        node.log_info("Logging system operational")
        node.log_warning("This is a test warning")

        rclpy.spin(node)
    except Exception as e:
        node.log_error(f"Unexpected error in logging system: {e}", exception=e)
    finally:
        node.log_info("Shutting down logging system")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()