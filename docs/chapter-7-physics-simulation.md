# Chapter 7: Physics Simulation - The Digital Twin Reality

## Simulation as Foundation

In this chapter, we explore the "Digital Twin" concept from our constitution, where simulation serves as the foundation for our robot's learning. As stated in our principles: "Simulation is Truth, Reality is the Test." This chapter covers implementing physics simulation that will allow our robot to learn and practice in a safe, controlled environment.

## Gazebo Simulation Environment

Gazebo provides a realistic physics simulation environment that includes:
- Accurate physics engine (ODE, Bullet, or DART)
- Sensor simulation (cameras, LIDAR, IMU, etc.)
- Environmental modeling
- Realistic lighting and rendering

### Key Tasks from Our Plan:
- T043: Implement physics simulation with Gazebo in src/simulation/physics_check.py

## Setting Up the Simulation Environment

### Creating a Custom World

First, let's create a custom world file for our robot to operate in:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="physical_ai_world">
    <!-- Include the Sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Include Ground Plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics Engine Configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Office Environment -->
    <model name="table">
      <pose>2 0 0.4 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Obstacle -->
    <model name="obstacle">
      <pose>-1 1 0.1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.01</iyy>
            <iyz>0.0</iyz>
            <izz>0.01</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Objects for Manipulation -->
    <model name="red_cube">
      <pose>2.2 0.1 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.0001</iyy>
            <iyz>0.0</iyz>
            <izz>0.0001</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="blue_cube">
      <pose>2.2 -0.1 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.8 1</ambient>
            <diffuse>0.2 0.2 0.8 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>0.1</mass>
          <inertia>
            <ixx>0.0001</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.0001</iyy>
            <iyz>0.0</iyz>
            <izz>0.0001</izz>
          </inertia>
        </inertial>
      </link>
    </model>

  </world>
</sdf>
```

## Physics Simulation Node

Let's create a Python script to test physics simulation:

```python
#!/usr/bin/env python3
"""
Physics Simulation Check for Physical AI System
Tests physics simulation functionality in Gazebo
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import time
import os

class PhysicsSimulationChecker(Node):
    def __init__(self):
        super().__init__('physics_simulation_checker')

        # Create service clients for Gazebo
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')

        # Wait for services to be available
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Spawn service not available, waiting again...')

        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Delete service not available, waiting again...')

        # Publisher for simulation status
        self.status_pub = self.create_publisher(String, 'simulation_status', 10)

        self.get_logger().info('Physics Simulation Checker Node Started')

    def spawn_object(self, name, xml, pose):
        """Spawn an object in Gazebo"""
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.initial_pose = pose

        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully spawned {name}')
                return True
            else:
                self.get_logger().error(f'Failed to spawn {name}: {response.status_message}')
                return False
        else:
            self.get_logger().error(f'Exception while spawning {name}: {future.exception()}')
            return False

    def delete_object(self, name):
        """Delete an object from Gazebo"""
        req = DeleteEntity.Request()
        req.name = name

        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Successfully deleted {name}')
                return True
            else:
                self.get_logger().error(f'Failed to delete {name}: {response.status_message}')
                return False
        else:
            self.get_logger().error(f'Exception while deleting {name}: {future.exception()}')
            return False

    def run_physics_tests(self):
        """Run comprehensive physics simulation tests"""
        self.get_logger().info('Starting physics simulation tests...')

        # Test 1: Object spawning
        self.get_logger().info('Test 1: Object spawning')

        # Simple cube model
        cube_model = '''<?xml version="1.0"?>
        <sdf version="1.7">
          <model name="test_cube">
            <link name="link">
              <collision name="collision">
                <geometry>
                  <box>
                    <size>0.1 0.1 0.1</size>
                  </box>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <box>
                    <size>0.1 0.1 0.1</size>
                  </box>
                </geometry>
                <material>
                  <ambient>1 0 0 1</ambient>
                  <diffuse>1 0 0 1</diffuse>
                </material>
              </visual>
              <inertial>
                <mass>0.1</mass>
                <inertia>
                  <ixx>0.0001</ixx>
                  <ixy>0.0</ixy>
                  <ixz>0.0</ixz>
                  <iyy>0.0001</iyy>
                  <iyz>0.0</iyz>
                  <izz>0.0001</izz>
                </inertia>
              </inertial>
            </link>
          </model>
        </sdf>'''

        # Create pose
        pose = Pose()
        pose.position.x = 1.0
        pose.position.y = 0.0
        pose.position.z = 0.5
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        # Spawn the cube
        if self.spawn_object('test_cube', cube_model, pose):
            self.get_logger().info('✓ Object spawning test passed')
        else:
            self.get_logger().error('✗ Object spawning test failed')
            return False

        # Test 2: Physics interaction (wait and observe)
        self.get_logger().info('Test 2: Physics interaction observation')
        self.get_logger().info('Waiting 5 seconds to observe physics behavior...')
        time.sleep(5)

        # Test 3: Object deletion
        self.get_logger().info('Test 3: Object deletion')
        if self.delete_object('test_cube'):
            self.get_logger().info('✓ Object deletion test passed')
        else:
            self.get_logger().error('✗ Object deletion test failed')
            return False

        # Test 4: Multiple objects interaction
        self.get_logger().info('Test 4: Multiple objects interaction')

        # Spawn multiple objects
        sphere_model = '''<?xml version="1.0"?>
        <sdf version="1.7">
          <model name="test_sphere">
            <link name="link">
              <collision name="collision">
                <geometry>
                  <sphere>
                    <radius>0.05</radius>
                  </sphere>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <sphere>
                    <radius>0.05</radius>
                  </sphere>
                </geometry>
                <material>
                  <ambient>0 1 0 1</ambient>
                  <diffuse>0 1 0 1</diffuse>
                </material>
              </visual>
              <inertial>
                <mass>0.05</mass>
                <inertia>
                  <ixx>0.0000125</ixx>
                  <ixy>0.0</ixy>
                  <ixz>0.0</ixz>
                  <iyy>0.0000125</iyy>
                  <iyz>0.0</iyz>
                  <izz>0.0000125</izz>
                </inertia>
              </inertial>
            </link>
          </model>
        </sdf>'''

        # Spawn spheres at different positions
        for i in range(3):
            pose.position.x = 2.0 + i * 0.2
            pose.position.y = 0.0
            pose.position.z = 1.0
            if not self.spawn_object(f'test_sphere_{i}', sphere_model, pose):
                self.get_logger().error(f'✗ Failed to spawn sphere {i}')
                return False

        self.get_logger().info('Waiting 5 seconds to observe multiple objects...')
        time.sleep(5)

        # Clean up spheres
        for i in range(3):
            self.delete_object(f'test_sphere_{i}')

        self.get_logger().info('✓ Multiple objects interaction test completed')

        # Publish success status
        status_msg = String()
        status_msg.data = 'Physics simulation tests completed successfully'
        self.status_pub.publish(status_msg)

        self.get_logger().info('✓ All physics simulation tests passed!')
        return True

def main(args=None):
    rclpy.init(args=args)
    checker = PhysicsSimulationChecker()

    # Run the tests
    success = checker.run_physics_tests()

    if success:
        checker.get_logger().info('Physics simulation validation successful!')
    else:
        checker.get_logger().error('Physics simulation validation failed!')

    checker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch File for Simulation

Create a launch file to easily start our simulation environment:

```python
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file = LaunchConfiguration('world', default='physical_ai_world.sdf')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('physical_ai_simulation'),
                'worlds',
                world_file
            ]),
            'verbose': 'false'
        }.items()
    )

    # Physics simulation checker node
    physics_checker = Node(
        package='physical_ai_simulation',
        executable='physics_simulation_checker',
        name='physics_simulation_checker',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='physical_ai_world.sdf',
            description='Choose one of the world files from `/physical_ai_simulation/worlds`'
        ),
        gazebo,
        physics_checker
    ])
```

## Physics Properties Configuration

For more advanced physics simulation, we can configure specific physics properties:

```python
#!/usr/bin/env python3
"""
Advanced Physics Configuration for Physical AI System
"""
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetPhysicsProperties, GetPhysicsProperties
from gazebo_msgs.msg import ODEPhysics
from std_msgs.msg import Float64

class PhysicsConfigurator(Node):
    def __init__(self):
        super().__init__('physics_configurator')

        # Create service clients
        self.set_physics_client = self.create_client(
            SetPhysicsProperties, '/set_physics_properties'
        )
        self.get_physics_client = self.create_client(
            GetPhysicsProperties, '/get_physics_properties'
        )

        # Wait for services
        while not self.set_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set physics service not available, waiting...')

        while not self.get_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Get physics service not available, waiting...')

        # Configure physics properties
        self.configure_physics()

    def configure_physics(self):
        """Configure physics properties for optimal simulation"""
        # Get current physics properties
        get_req = GetPhysicsProperties.Request()
        get_future = self.get_physics_client.call_async(get_req)
        rclpy.spin_until_future_complete(self, get_future)

        if get_future.result() is not None:
            current_props = get_future.result()
            self.get_logger().info(f'Current gravity: {current_props.gravity}')
        else:
            self.get_logger().error('Failed to get current physics properties')

        # Set new physics properties optimized for our robot
        req = SetPhysicsProperties.Request()
        req.time_step = 0.001  # 1ms time step for accuracy
        req.max_update_rate = 1000.0  # 1000 Hz update rate
        req.gravity = [0.0, 0.0, -9.8]  # Standard Earth gravity

        # ODE physics parameters
        req.ode_config = ODEPhysics()
        req.ode_config.auto_disable_bodies = False
        req.ode_config.sor_pgs_precon_iters = 2
        req.ode_config.sor_pgs_iters = 50
        req.ode_config.sor_pgs_w = 1.3
        req.ode_config.contact_surface_layer = 0.001
        req.ode_config.contact_max_correcting_vel = 100.0
        req.ode_config.cfm = 0.0
        req.ode_config.erp = 0.2
        req.ode_config.max_contacts = 20

        # Call the service
        future = self.set_physics_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info('Physics properties configured successfully')
            else:
                self.get_logger().error(f'Failed to set physics properties: {response.status_message}')
        else:
            self.get_logger().error(f'Exception setting physics properties: {future.exception()}')

def main(args=None):
    rclpy.init(args=args)
    configurator = PhysicsConfigurator()

    # Keep the node running
    try:
        rclpy.spin(configurator)
    except KeyboardInterrupt:
        pass
    finally:
        configurator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Simulation Best Practices

### Performance Optimization
1. **Time Step**: Use the smallest time step that provides stable simulation
2. **Update Rate**: Balance between accuracy and performance
3. **Collision Geometry**: Use simple shapes when possible
4. **Visual Fidelity**: Reduce visual complexity during heavy simulation

### Safety in Simulation
1. **Bounds Checking**: Ensure objects stay within simulation boundaries
2. **Force Limits**: Apply reasonable force and velocity limits
3. **Emergency Stops**: Implement simulation emergency stops
4. **Recovery Procedures**: Have methods to reset simulation state

### Validation Techniques
1. **Physics Accuracy**: Compare simulation results with real-world physics
2. **Timing Consistency**: Ensure simulation runs at consistent speeds
3. **Collision Detection**: Verify objects interact as expected
4. **Sensor Simulation**: Validate sensor outputs match expectations

## Looking Forward

With our physics simulation foundation established, the next chapter will focus on sensor awareness systems that will allow our robot to perceive and interact with its simulated environment.

[Continue to Chapter 8: Sensor Awareness](./chapter-8-sensor-awareness.md)