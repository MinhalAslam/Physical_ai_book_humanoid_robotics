#!/usr/bin/env python3
"""
ROS 2 Graph Visualization Node

This node visualizes the ROS 2 communication graph and shows
the relationships between different nodes, topics, and services.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter
from std_msgs.msg import String
import json
import time


class ROSGraphVisualizer(Node):
    def __init__(self):
        super().__init__('ros_graph_visualizer')

        # Publisher for graph information
        self.graph_info_pub = self.create_publisher(String, '/ros_graph_info', 10)

        # Timer to periodically update graph information
        self.timer = self.create_timer(2.0, self.update_graph_info)

        self.get_logger().info('ROS Graph Visualizer Node Started')

        # Track node connections
        self.node_connections = {}
        self.topics = set()
        self.services = set()

    def update_graph_info(self):
        """Update and publish graph information"""
        # Get current node graph information
        node_names = self.get_node_names()

        # Create graph representation
        graph_info = {
            'timestamp': time.time(),
            'nodes': node_names,
            'total_nodes': len(node_names),
            'topics': list(self.get_topic_names_and_types()),
            'services': list(self.get_service_names_and_types()),
            'node_count': len(node_names)
        }

        # Publish graph information as JSON
        msg = String()
        msg.data = json.dumps(graph_info, indent=2)
        self.graph_info_pub.publish(msg)

        self.get_logger().info(f'ROS Graph: {len(node_names)} nodes, {len(graph_info["topics"])} topics')


def main(args=None):
    rclpy.init(args=args)

    node = ROSGraphVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down ROS Graph Visualizer')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()