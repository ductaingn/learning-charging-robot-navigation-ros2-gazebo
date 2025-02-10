#!/usr/bin/env python3
import os
from gazebo_environment import WheeledRobotEnv
import stable_baselines3 as sb3
import rclpy
import threading
import yaml

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs.yaml')
    with open(config_path) as file:
        configs = yaml.safe_load(file)    
    env = WheeledRobotEnv(**configs)

    model = sb3.SAC("MultiInputPolicy", env, verbose=1)

    model.learn(total_timesteps=1,progress_bar=True)
    model.save("model/wheeled_robot_SAC")
    env.close()
