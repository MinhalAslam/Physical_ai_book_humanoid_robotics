# Data Model: Physical AI Capstone - Autonomous Humanoid Robot

**Feature**: 001-physical-ai-capstone
**Date**: 2025-12-13
**Status**: Draft

## Entity Models

### VoiceCommand
Represents a spoken instruction from a human user

- **id**: UUID (unique identifier)
- **raw_audio**: AudioData (raw audio sample)
- **transcribed_text**: String (speech-to-text result)
- **parsed_intent**: String (NLU-parsed intent)
- **confidence**: Float (0.0-1.0, confidence in transcription)
- **timestamp**: DateTime (when command was received)
- **source**: String (microphone ID or source)

### NavigationGoal
Represents a target location or waypoint for the robot to navigate to

- **id**: UUID (unique identifier)
- **x**: Float (x-coordinate in map frame)
- **y**: Float (y-coordinate in map frame)
- **theta**: Float (orientation in radians)
- **frame_id**: String (coordinate frame)
- **priority**: Int (navigation priority level)
- **timestamp**: DateTime (when goal was set)

### DetectedObject
Represents recognized objects in the environment

- **id**: UUID (unique identifier)
- **name**: String (object class name)
- **confidence**: Float (0.0-1.0, recognition confidence)
- **position**: Point3D (x, y, z coordinates in robot frame)
- **dimensions**: Vector3D (width, height, depth)
- **manipulation_params**: ManipulationParams (grip points, approach vectors)
- **timestamp**: DateTime (when object was detected)

### TaskSequence
Represents a planned sequence of actions derived from user intent

- **id**: UUID (unique identifier)
- **name**: String (task sequence name/description)
- **steps**: Array[TaskStep] (ordered list of steps)
- **status**: TaskStatus (PENDING, RUNNING, COMPLETED, FAILED)
- **created_at**: DateTime (when sequence was created)
- **completed_at**: DateTime (when sequence was completed)
- **robot_state**: RobotState (required robot state for execution)

### TaskStep
Represents a single step in a task sequence

- **id**: UUID (unique identifier)
- **type**: TaskStepType (NAVIGATION, MANIPULATION, SPEECH, WAIT, CONDITIONAL)
- **parameters**: JSON (step-specific parameters)
- **timeout**: Duration (maximum time to complete step)
- **dependencies**: Array[UUID] (IDs of prerequisite steps)
- **status**: TaskStepStatus (PENDING, RUNNING, COMPLETED, FAILED)

### RobotState
Represents the current operational state of the robot

- **id**: UUID (unique identifier)
- **state**: RobotStateEnum (IDLE, LISTENING, PROCESSING, NAVIGATING, MANIPULATING, ERROR, EMERGENCY_STOP)
- **position**: Pose2D (current x, y, theta position)
- **battery_level**: Float (0.0-1.0, battery charge level)
- **safety_status**: SafetyStatus (SAFE, WARNING, DANGER, EMERGENCY)
- **timestamp**: DateTime (when state was updated)

### SensorData
Represents data from various robot sensors

- **id**: UUID (unique identifier)
- **sensor_type**: SensorType (CAMERA, LIDAR, IMU, DEPTH, MICROPHONE)
- **data**: Binary (sensor-specific data format)
- **timestamp**: DateTime (when data was captured)
- **frame_id**: String (coordinate frame for spatial data)

## State Transitions

### RobotState Transitions
- IDLE → LISTENING (when robot enters listening mode)
- LISTENING → PROCESSING (when voice command received)
- PROCESSING → NAVIGATING (when navigation task starts)
- PROCESSING → MANIPULATING (when manipulation task starts)
- PROCESSING → IDLE (when command is simple/complete)
- NAVIGATING → IDLE (when navigation completes)
- MANIPULATING → IDLE (when manipulation completes)
- ANY → ERROR (when error occurs)
- ANY → EMERGENCY_STOP (when emergency stop triggered)
- ERROR → IDLE (when error resolved)
- EMERGENCY_STOP → IDLE (when emergency resolved)

### TaskStep Status Transitions
- PENDING → RUNNING (when step execution starts)
- RUNNING → COMPLETED (when step completes successfully)
- RUNNING → FAILED (when step fails)
- RUNNING → PENDING (when step is paused/retried)

## Validation Rules

### VoiceCommand
- confidence must be between 0.0 and 1.0
- transcribed_text must not be empty
- timestamp must be recent (within 10 seconds)

### NavigationGoal
- x, y coordinates must be within map bounds
- priority must be between 1 and 10
- frame_id must be valid coordinate frame

### DetectedObject
- confidence must be between 0.0 and 1.0
- position coordinates must be valid
- name must be in recognized object classes

### TaskSequence
- steps array must not be empty
- status must be consistent with individual step statuses
- robot_state must be achievable from current state

### RobotState
- battery_level must be between 0.0 and 1.0
- position coordinates must be valid
- state transitions must follow defined rules