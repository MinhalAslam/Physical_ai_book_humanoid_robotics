# Physical AI & Humanoid Robotics - Quick Start Guide

## Overview

This guide will help you get the Physical AI system up and running quickly. The system enables voice-controlled navigation and interaction with the environment using ROS 2, computer vision, and language models.

## Prerequisites

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Python 3.10+
- NVIDIA GPU (for Isaac ROS packages)
- Compatible robot hardware or Gazebo simulation

## Installation

### 1. System Dependencies

```bash
# Install ROS 2 Humble
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-build

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2
source /opt/ros/humble/setup.bash
```

### 2. Project Setup

```bash
# Create workspace
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws

# Clone the repository
git clone <repository-url> src/physical_ai

# Install Python dependencies
pip3 install -r src/physical_ai/requirements.txt

# Build the workspace
colcon build --packages-select physical_ai_robot
source install/setup.bash
```

### 3. Install Isaac ROS Dependencies (Optional)

```bash
# Install Isaac ROS packages
sudo apt install ros-humble-isaac-ros-* ros-humble-nvidia-isaac-* ros-humble-gazebo-*
```

## Quick Start Commands

### 1. Launch Simulation Environment

```bash
# Launch the complete system in simulation
ros2 launch physical_ai_bringup simulation.launch.py
```

### 2. Launch Real Robot (Hardware)

```bash
# Launch the complete system on real robot
ros2 launch physical_ai_bringup robot.launch.py
```

### 3. Test Voice Commands

Once the system is running, you can issue voice commands such as:
- "Move forward"
- "Turn left"
- "Find person"
- "Come here"
- "Stop"

## Basic Usage

### Voice Command Pipeline
1. Say a command to the robot
2. Speech is converted to text using ASR
3. Text is processed by the language model
4. Actions are planned and executed
5. Robot provides verbal confirmation

### Navigation Commands
- **Move commands**: "Move forward/backward", "Go left/right"
- **Navigation**: "Go to kitchen", "Navigate to charging station"
- **Following**: "Follow me", "Come with me"

### Safety Features
- Emergency stop: Say "Stop now" or press emergency stop button
- Obstacle avoidance: Robot automatically stops when obstacles are detected
- Safe zones: Predefined areas where robot movement is restricted

## System Validation

### 1. Check System Status

```bash
# Monitor system status
ros2 topic echo /system_status

# Check safety status
ros2 topic echo /safety_status
```

### 2. Test Individual Components

```bash
# Test speech recognition
ros2 run speech chapter_voice

# Test navigation
ros2 run navigation chapter_nav

# Test perception
ros2 run perception sensor_node
```

## Troubleshooting

### Common Issues

1. **No speech recognition**: Check microphone permissions and audio input
2. **Navigation not working**: Verify map and localization are active
3. **Safety system active**: Check for obstacles or safety violations

### System Diagnostics

```bash
# Run system diagnostics
ros2 run core system_diagnostics

# Check all active nodes
ros2 node list

# Monitor all topics
ros2 topic list
```

## Next Steps

1. **Customize commands**: Modify the command parser in `src/speech/chapter_voice.py`
2. **Add new capabilities**: Extend the cognitive planning system
3. **Configure for your robot**: Update URDF in `src/robot_description/`
4. **Performance tuning**: Adjust parameters in `src/config/robot_config.py`

## Support

For additional support, check the complete documentation or contact the development team.