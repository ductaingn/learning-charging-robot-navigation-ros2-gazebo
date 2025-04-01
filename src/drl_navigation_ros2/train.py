#!/usr/bin/env python3
import os
# os.environ["QT_QPA_PLATFORM"] = "xcb"
from environment import WheeledRobotEnv
from architectures import CustomFeatureExtractor
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import yaml
from datetime import datetime

if __name__ == "__main__":
    # Load configurations
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs.yaml')
    with open(config_path) as file:
        configs = yaml.safe_load(file)
    map_info = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map_info.yaml')
    with open(map_info) as file:
        map_info = yaml.safe_load(file)
    map_name = str(map_info['using_map_name'])
    start_x, start_y, start_z, start_roll, start_pitch, start_yaw = list(map_info[map_name]['start'])
    configs['start_pose'] = [start_x, start_y, start_z, start_roll, start_pitch, start_yaw]
    goal_x, goal_y, goal_z, goal_roll, goal_pitch, goal_yaw = list(map_info[map_name]['goal'])
    configs['goal_pos'] = [goal_x, goal_y, goal_z]

    # Use for saving models and logs
    time_now = datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S")

    # Initialize Environment    
    env = WheeledRobotEnv(**configs)

    # Initialize Model
    policy_kwargs = dict(
        features_extractor_class = CustomFeatureExtractor,
        features_extractor_kwargs = dict(
            history_dim=env.lidar_dim,
            observation_dim=env.lidar_dim,
            kin_input_dim=env.kinetic_dim,
            kin_output_dim=64,
            hidden_dim=256,
            n_history_frame=env.n_history_frame,
            n_waypoints=env.n_waypoints,
        )
    )
    model = sb3.SAC(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        buffer_size=100000,
        verbose=1,
        batch_size=1024
    )

    # model.load("trained_weights/SB3-2025-02-25-00-01-28/SAC_model_20000_steps")
    # model.set_parameters("trained_weights/SB3-2025-02-25-00-01-28/SAC_model_20000_steps")
    # model.load_replay_buffer("trained_weights/SB3-2025-02-25-00-01-28/SAC_model_replay_buffer_20000_steps")

    ## Initialize callback and logger
    checkpoint_callback = CheckpointCallback(
        save_freq=2500,
        save_path=f"trained_weights/{time_now}",
        name_prefix="SAC_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    ) 
    logger = configure(folder=f"training_log/{time_now}", format_strings=["stdout","csv"])
    model.set_logger(logger)

    model.learn(total_timesteps=configs['max_step'], progress_bar=True, callback=checkpoint_callback, log_interval=1)
    model.save(f"trained_weights/{time_now}")
    env.close()
