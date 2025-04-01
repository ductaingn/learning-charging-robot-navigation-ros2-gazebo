# Learning Charging Robot Navigation

## Project Overview

This repository provides a simulation environment for charging robot navigation using ROS2 (Jazzy Jalisco) and Gazebo (Harmonic). It integrates several components including robot control, simulation environment setup, and navigation algorithms. The repository is structured to help you build the workspace, launch simulation nodes, and test various navigation strategies.

## Setup and Build

1. **Clone the Repository:**  
Clone the repository and navigate to the workspace directory:

```sh
cd learning_charging_robot_navigation
```

2. **Build the Workspace:**  
Use colcon to build the workspace with symbolic links for faster iteration:

```sh
colcon build --symlink-install
```
3. **Source the Environment Setup::**  
If you are using zsh terminal
```zsh
source install/setup.zsh
```

If you are using bash terminal
```sh
source install/setup.bash
```
## Configuration
Before launching the simulation, choose the desired map and modify the robot's starting position in the map configuration file: 
- *"learning_charging_robot_navigation/src/velocity_pub/scripts/map_info.yaml"*

## Launching the Simulation
- Start Gazebo Simulation and Spawn the Robot: Launch the Gazebo simulation and spawn the Turtlebot3 robot:
```sh
ros2 launch turtlebot3_gazebo ros2_drl.launch.py
```


- Train robot navigation: Begin the training process using the training launch file:
```sh
ros2 launch velocity_pub train_robot_navigation.launch.py
```

- Test robot navigation: Run the test launch file to evaluate the robot navigation:
```sh
ros2 launch velocity_pub test_robot_navigation.launch.py
```

## Optional Navigation Modes
- Use Nav2 MPPI for Local planner and A* for Global planner:
```sh
ros2 launch fws_robot_sim navigation.launch.py
```

- Use SLAM toolbox mapping:
```sh
ros2 launch fws_robot_sim mapping.launch.py
```