# How to run

```
cd fws_robot_harmonic
colcon build
source instal/setup.zsh
ros2 launch velocity_pub four_ws_control.launch.py 
ros2 launch fws_robot_sim fws_robot_spawn.launch.py
```