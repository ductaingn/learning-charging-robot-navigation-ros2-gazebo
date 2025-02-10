#!/usr/bin/env python3
import gymnasium as gym
import torch
import stable_baselines3 as sb3
from gazebo_environment import WheeledRobotEnv
import os
import yaml

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs.yaml')
    with open(config_path) as file:
        configs = yaml.safe_load(file)    
    env = WheeledRobotEnv(**configs)
    model = sb3.SAC.load("model/wheeled_robot_SAC")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # if terminated or truncated:
            # break

    env.close()