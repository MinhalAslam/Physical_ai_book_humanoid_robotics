# Quickstart Guide: Physical AI Capstone - Autonomous Humanoid Robot

**Feature**: 001-physical-ai-capstone
**Date**: 2025-12-13
**Status**: Draft

## Prerequisites

### System Requirements
- Ubuntu 22.04 LTS
- NVIDIA GPU (RTX 3060 or better) or NVIDIA Jetson AGX Orin
- 16GB RAM minimum, 32GB recommended
- 100GB free disk space

### Software Dependencies
```bash
# Install ROS 2 Humble
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
echo "deb [arch=amd64,arm64] http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update && sudo apt upgrade
sudo apt install ros-humble-desktop ros-humble-ros-base
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Install additional ROS packages
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-xacro ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-rosbridge-suite ros-humble-tf2-tools ros-humble-teleop-tools

# Install Python dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install openai-whisper opencv-python transformers
```

### Hardware Setup
1. Connect RGB-D camera (Intel RealSense or equivalent)
2. Connect microphones for audio input
3. Ensure robot base is connected and calibrated
4. Verify safety systems are operational

## Setup Instructions

### 1. Clone and Initialize Workspace
```bash
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws
git clone https://github.com/your-org/physical-ai-capstone.git src/physical_ai
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select physical_ai_bringup
source install/setup.bash
```

### 2. Configure Environment
```bash
# Add to ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/physical_ai_ws/install/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=1" >> ~/.bashrc  # Adjust as needed
```

### 3. Run Simulation Environment
```bash
# Terminal 1: Launch simulation
ros2 launch physical_ai_gazebo simulation.launch.py

# Terminal 2: Launch robot brain
ros2 launch physical_ai_bringup robot_bringup.launch.py

# Terminal 3: Launch speech interface
ros2 launch physical_ai_speech speech_interface.launch.py
```

## Basic Commands

### Voice Command Interface
1. Say "Robot, listen" to activate listening mode
2. Give a command like "Go to the kitchen" or "Pick up the red cup"
3. The robot will confirm the command and execute

### Manual Control
```bash
# Publish a navigation goal
ros2 topic pub /navigation_goal geometry_msgs/PoseStamped "header:
  frame_id: 'map'
pose:
  position:
    x: 1.0
    y: 2.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0"

# Check robot state
ros2 topic echo /robot_state --field state
```

## Testing the System

### 1. Basic Functionality Test
```bash
# Test speech recognition
ros2 launch physical_ai_speech test_speech.launch.py

# Test navigation in simulation
ros2 launch physical_ai_navigation test_navigation.launch.py

# Test object detection
ros2 launch physical_ai_perception test_detection.launch.py
```

### 2. Integration Test
1. Launch complete system in simulation
2. Give voice command: "Robot, go to the table and find the book"
3. Verify robot navigates to table and identifies book
4. Check all safety systems are operational

## Performance Validation

### Speech Recognition Test
```bash
# Run speech accuracy test
ros2 run physical_ai_speech test_accuracy.py --duration 60
# Expected: ≥90% accuracy
```

### Navigation Test
```bash
# Run navigation test
ros2 run physical_ai_navigation test_navigation.py --trials 50
# Expected: ≥95% collision-free navigation
```

### End-to-End Response Test
```bash
# Measure response time
ros2 run physical_ai_core test_response_time.py
# Expected: ≤2 seconds from command to action
```

## Troubleshooting

### Common Issues

**Issue**: Robot doesn't respond to voice commands
- **Solution**: Check microphone permissions and audio input levels
- **Check**: `ros2 topic echo /audio_input` to verify audio is being received

**Issue**: Navigation fails frequently
- **Solution**: Verify map quality and localization accuracy
- **Check**: `ros2 topic echo /amcl_pose` to verify robot knows its position

**Issue**: Object detection is inaccurate
- **Solution**: Recalibrate camera and update object recognition models
- **Check**: `ros2 topic echo /detected_objects` to verify detection output

### Safety Checks
- Always verify emergency stop functionality before testing
- Monitor battery level during extended tests
- Ensure safety barriers are in place for real-robot testing

## Next Steps

1. Complete the 20-chapter book-style implementation following the master plan
2. Implement each act (Awakening Intelligence, Nervous System, Digital Twin, etc.)
3. Validate sim-to-real transfer capabilities
4. Deploy on physical robot platform
5. Conduct final capstone demonstration