# Chapter 6: Robot Skeleton URDF - Giving Form to Function

## The Digital Body

In this chapter, we create the physical representation of our robot using URDF (Unified Robot Description Format). This is where our robot gains its "digital twin" form, aligning with our constitution's principle that "Simulation is Truth, Reality is the Test."

## Understanding URDF

URDF (Unified Robot Description Format) is an XML format that describes robots in terms of their links, joints, and other elements. It's the blueprint that defines our robot's physical structure, from its base to its sensors and actuators.

### Key URDF Elements:
- **Links**: Rigid bodies that make up the robot
- **Joints**: Connections between links with specific degrees of freedom
- **Visual**: How the robot appears in simulation
- **Collision**: How the robot interacts with the environment
- **Inertial**: Physical properties for physics simulation

## Basic Robot URDF Implementation

Based on our project plan, we need to create a robot skeleton URDF model. Let's start with a simple wheeled robot and then expand to a more complex humanoid form.

### Simple Wheeled Robot (base for our system):

```xml
<?xml version="1.0"?>
<robot name="physical_ai_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57075 0 0"/>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57075 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57075 0 0"/>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <origin rpy="1.57075 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Camera -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.03"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.15 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.15 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo Plugin for differential drive -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>/</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <update_rate>30</update_rate>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>

  <!-- Gazebo plugin for camera -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera_sensor">
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>/camera</namespace>
          <remapping>image_raw:=image_raw</remapping>
          <remapping>camera_info:=camera_info</remapping>
        </ros>
        <camera_name>camera</camera_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

## Advanced Humanoid Robot URDF (Xacro Version)

For a more complex humanoid robot, we can use Xacro (XML Macros) to make our URDF more maintainable:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include macros -->
  <xacro:include filename="$(find physical_ai_description)/urdf/materials.xacro" />
  <xacro:include filename="$(find physical_ai_description)/urdf/properties.xacro" />

  <!-- Base Properties -->
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_length" value="0.3" />
  <xacro:property name="base_height" value="0.15" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
      <material name="blue_material"/>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white_material"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Neck Joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 ${base_height/2 + 0.1}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <!-- Camera in Head -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.03"/>
      </geometry>
      <material name="black_material"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.03"/>
      </geometry>
      <material name="gray_material"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.03"/>
      </geometry>
      <material name="gray_material"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Arm Joints -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="${base_width/2 + 0.03} 0 ${base_height/2}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <!-- Gazebo Plugins -->
  <gazebo>
    <plugin name="ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Camera Gazebo Plugin -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera_sensor">
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>/camera</namespace>
          <remapping>image_raw:=image_raw</remapping>
          <remapping>camera_info:=camera_info</remapping>
        </ros>
        <camera_name>camera</camera_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

## Materials Xacro File

Create a materials.xacro file for reusability:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <xacro:macro name="material_color" params="name color">
    <material name="${name}">
      <color rgba="${color}"/>
    </material>
  </xacro:macro>

  <xacro:material_color name="blue_material" color="0 0 0.8 1"/>
  <xacro:material_color name="red_material" color="0.8 0 0 1"/>
  <xacro:material_color name="green_material" color="0 0.8 0 1"/>
  <xacro:material_color name="yellow_material" color="0.8 0.8 0 1"/>
  <xacro:material_color name="black_material" color="0 0 0 1"/>
  <xacro:material_color name="white_material" color="1 1 1 1"/>
  <xacro:material_color name="gray_material" color="0.5 0.5 0.5 1"/>

</robot>
```

## Loading and Testing the URDF

To load and test your URDF in RViz:

```bash
# Launch robot state publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat your_robot.urdf)

# Or using xacro
ros2 run xacro xacro --inorder your_robot.xacro | ros2 run robot_state_publisher robot_state_publisher

# View in RViz
rviz2

# Or launch in Gazebo
ros2 launch gazebo_ros gazebo.launch.py
ros2 run gazebo_ros spawn_entity.py -topic robot_description -entity my_robot
```

## Integration with Perception System

The URDF is crucial for the perception system as it provides:
- Sensor positions and orientations in the robot frame
- Collision geometry for physics simulation
- Visual geometry for rendering in simulation

## Safety and Validation

Following our safety principles, always validate your URDF:

1. **Check for self-collisions**: Ensure links don't intersect in normal poses
2. **Verify inertial properties**: Use realistic mass and inertia values
3. **Validate joint limits**: Ensure safe ranges of motion
4. **Test in simulation**: Always test new URDF changes in simulation before real hardware

## Looking Forward

With our robot's skeleton defined, the next act will focus on creating the digital twin in simulation, where we'll implement physics simulation and sensor awareness systems.

[Continue to Chapter 7: Physics Simulation](./chapter-7-physics-simulation.md)