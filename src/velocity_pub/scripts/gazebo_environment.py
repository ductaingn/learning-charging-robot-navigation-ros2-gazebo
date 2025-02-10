import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from gymnasium import spaces
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan
from typing import Optional
from squaternion import Quaternion
from nav_msgs.msg import Odometry
import torch
import threading

class WheeledRobotEnv(gym.Env):
    def __init__(self, max_step:int, environment_shape:list, lidar_dim:int, lidar_range:int, collision_threshold:float, goal_threshold:float, robot_max_lin_vel:float, reward_coeff:dict):
        super(WheeledRobotEnv, self).__init__()
        self.max_step = max_step
        self.environment_shape = environment_shape
        self.lidar_dim = lidar_dim
        self.lidar_range = lidar_range
        self.collision_threshold = collision_threshold
        self.goal_threshold = goal_threshold
        self.reward_coeff = reward_coeff
        self.robot_max_lin_vel = robot_max_lin_vel

        # Robot dimensions
        self.wheel_seperation = 0.122
        self.wheel_base = 0.156
        self.wheel_radius = 0.026
        self.wheel_steering_y_offset = 0.03
        self.steering_track = self.wheel_seperation - 2*self.wheel_steering_y_offset

        rclpy.init()
        self.node = rclpy.create_node('wheeled_robot_env')
        self.vel_msg = Twist()
        self.pos = np.array([0,0,0,0], float)
        self.vel = np.array([0,0,0,0], float) #left_front, right_front, left_rear, right_rear

        # ROS2 topic for controlling the robot
        self.pub_pos = self.node.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.pub_vel = self.node.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.timer = self.node.create_timer(timer_period_sec=0.03, callback=self.timer_callback)

        # ROS2 topic for lidar data
        self.laser_data = None
        self.laser_sub = self.node.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # ROS2 topic for odometry data
        self.odom_sub = self.node.create_subscription(
            Odometry,
            '/tf',
            self.odom_callback,
            10
        )

        # Variable to store robot's position and kinematic information from ROS2 (position, orientation, velocities)
        self.robot_state = {
            'x': 0.0,           # X position
            'y': 0.0,           # Y position
            'theta': 0.0,       # Orientation angle (radians)
            'linear_vel': 0.0,  # Linear velocity
            'angular_vel': 0.0  # Angular velocity
        }

        # Define RL spaces
        self.action_space = spaces.Box(low=np.array([-self.robot_max_lin_vel, -np.pi/3]), high=np.array([robot_max_lin_vel, np.pi/3]), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                "odometry": gym.spaces.Box(
                    low=np.array([-self.environment_shape[0]/2, -self.environment_shape[1]/2, -np.pi/2, 0, 0]),
                    high=np.array([self.environment_shape[0]/2, self.environment_shape[1]/2, np.pi/2, robot_max_lin_vel, np.pi])
                ),
                "lidar": gym.spaces.Box(0, self.lidar_range, shape=(self.lidar_dim, ))
            }
        )

        self.current_obs:dict[np.ndarray, np.ndarray] = self.get_obs()
        self.prev_obs:dict[np.ndarray, np.ndarray] = self.current_obs
        self.goal_pos:np.ndarray = np.zeros(2)
        self.current_num_step = 0

        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.node)

        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        rate = self.node.create_rate(2)

    def timer_callback(self):
        self.vel_msg.linear.x = (torch.rand(1).item()-.5)*10
        self.vel_msg.linear.y = (torch.rand(1).item()-.5)*10

        V = np.hypot(self.vel_msg.linear.x, self.vel_msg.linear.y)
        sign = np.sign(self.vel_msg.linear.x)
        
        if(self.vel_msg.linear.x != 0):
            ang = self.vel_msg.linear.y / self.vel_msg.linear.x
        else:
            ang = 0
        
        self.pos[0] = np.arctan(ang)
        self.pos[1] = np.arctan(ang)
        self.pos[2] = self.pos[0]
        self.pos[3] = self.pos[1]
        
        self.vel[:] = sign*V
            
        pos_array = Float64MultiArray(data=self.pos) 
        vel_array = Float64MultiArray(data=self.vel)

        self.pub_pos.publish(pos_array)
        self.pub_vel.publish(vel_array)
        self.pos[:] = 0
        self.vel[:] = 0

    def odom_callback(self, msg):
        # Update robot position
        self.robot_state['x'] = msg.pose.pose.position.x
        self.robot_state['y'] = msg.pose.pose.position.y

        # Convert quaternion to yaw (theta)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_state['theta'] = np.arctan2(siny_cosp, cosy_cosp)

        # Update velocities
        self.robot_state['linear_vel'] = msg.twist.twist.linear.x
        self.robot_state['angular_vel'] = msg.twist.twist.angular.z

    def lidar_callback(self, msg):
        # Convert the list of range measurements from the LaserScan message to a NumPy array
        raw_laser_data = np.array(msg.ranges)
        self.laser_data = np.clip(raw_laser_data, 0, self.lidar_range)
        print("min:", self.laser_data.min())

    def get_obs(self):
        # Use default high readings if lidar data isn't available or complete
        if self.laser_data is None or len(self.laser_data) < self.lidar_dim:
            lidar_data = np.full((self.lidar_dim,), self.lidar_range, dtype=np.float32)
        else:
            lidar_data = np.array(self.laser_data[:self.lidar_dim], dtype=np.float32)

        # Pack the latest odometry information into an array
        odometry = np.array([
            self.robot_state['x'],
            self.robot_state['y'],
            self.robot_state['theta'],
            self.robot_state['linear_vel'],
            self.robot_state['angular_vel']
        ], dtype=np.float32)

        # Return both sensor modalities in a dictionary
        self.current_obs = {'odometry': odometry,'lidar': lidar_data}
        return self.current_obs
    
    def step(self, action):
        observation = {}
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        self.current_num_step +=1

        # Calculate reward base on current state
        if self.current_num_step >= self.max_step:  # Time limit exceeded
            truncated = True
        
        if self.collision_check(self.current_obs):
            reward, reward_components = self.get_rewards(collise=True)
            terminated = True
        else:
            reward, reward_components = self.get_rewards(collise=False)

        # Take action
        self.vel_msg.linear.x = float(action[0])
        self.vel_msg.angular.z = float(action[1])

        # Next observation
        observation = self.get_obs()
        
        # Step infomation
        info = reward_components

        return observation, reward, terminated, truncated, info
    
    def collision_check(self, observation)->bool:
        lidar_data:np.ndarray = observation['lidar']
        if lidar_data.min() <= self.collision_threshold:
            return True
        return False
    
    def get_rewards(self, collise):
        reward = 0.0
        reward_components = {}

        # Reward collision
        if collise:
            reward_components['collision'] = self.reward_coeff['collision']['collise']
        else:
            reward_components['collision'] = self.reward_coeff['collision']['coeff']*0
        
        # Reward goal 
        robot_pos = self.current_obs['odometry'][:2] # x,y
        goal_distance = np.linalg.norm(robot_pos - self.goal_pos)

        if goal_distance <= self.collision_threshold:
            reward_components['goal'] = self.reward_coeff['goal']['reach']
        else:
            reward_components['goal'] = self.reward_coeff['goal']['coeff']*0

        # To-do: Adjust
        # Reward velocity
        linear_vel = self.current_obs['odometry'][3] # linear velocity
        if linear_vel < 1:
            reward_components['velocity'] = -1

        reward = sum(reward_components.values())

        return reward, reward_components

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        observation = {}
        info = {}

        observation = self.get_obs()
        for key in self.reward_coeff.keys():
            info[key] = 0 # Log componentg rewards as 0

        return observation, info

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
