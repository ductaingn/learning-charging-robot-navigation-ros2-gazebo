import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from gymnasium import spaces
from geometry_msgs.msg import Twist, PoseArray
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan
from typing import Optional
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import torch
import threading
from ros_gz_interfaces.srv import SpawnEntity, SetEntityPose, ControlWorld
from geometry_msgs.msg import Pose
import subprocess
import wandb
from rclpy.qos import qos_profile_system_default
import time

class WheeledRobotEnv(gym.Env):
    def __init__(self, max_step:int, environment_shape:list, lidar_dim:int, lidar_max_range:float, lidar_min_range:float, n_history_frame:int, collision_threshold:float, n_waypoints:int, goal_threshold:float, robot_max_lin_vel:float, reward_coeff:dict, real_time_factor:float, delta_t:float):
        """
        Initialize the WheeledRobotEnv environment.

        This constructor initializes the environment by setting up ROS2 communications, publishers, subscribers,
        the robot's kinematic properties, and reinforcement learning (RL) spaces (both action and observation).

        Parameters:
            max_step (int): Maximum number of steps allowed in an episode. 
            environment_shape (list): Dimensions of the environment [width, height].
            lidar_dim (int): Number of dimensions (measurements) for the LiDAR sensor.
            lidar_max_range (float): Maximum range the LiDAR sensor can measure.
            lidar_min_range (float): Minimum range the LiDAR sensor can measure.
            n_history_frame (int): Number of previous LiDAR frames to maintain for history.
            collision_threshold (float): Threshold distance to detect a collision.
            n_waypoints (int): Number of waypoints for navigation planning.
            goal_threshold (float): Distance threshold to determine if the goal is reached.
            robot_max_lin_vel (float): Maximum linear velocity allowed for the robot.
            reward_coeff (dict): Dictionary of coefficients for computing rewards.

        Attributes:
            wheel_seperation (float): Distance between wheels of the robot.
            wheel_base (float): Distance between the front and rear axles.
            wheel_radius (float): Radius of each wheel.
            wheel_steering_y_offset (float): Lateral offset for steering wheels.
            steering_track (float): Effective distance between steering wheels (wheel_seperation minus offsets).
            node (rclpy.Node): ROS2 node responsible for communication.
            clock (Clock): ROS2 clock for time control within the node.
            loop_rate (Rate): Loop rate object for managing the control loop frequency.
            vel_msg (Twist): Message object to command robot velocities.
            pos (numpy.ndarray): Array representing the robot's position state [x, y, theta, additional_state].
            vel (numpy.ndarray): Array representing wheel velocities [left_front, right_front, left_rear, right_rear].
            pub_pos (Publisher): ROS2 publisher for sending position commands to the robot.
            pub_vel (Publisher): ROS2 publisher for sending velocity commands to the robot.
            current_lidar_data (numpy.ndarray): Array holding the latest LiDAR measurements.
            lidar_data (numpy.ndarray): Array containing current and historical LiDAR data.
            lidar_sub (Subscription): ROS2 subscription for receiving LiDAR sensor data.
            robot_pose_sub (Subscription): ROS2 subscription for receiving odometry data.
            pause_pub (Publisher): ROS2 publisher to pause or unpause the simulation environment.
            robot_state (dict): Dictionary maintaining the robot's current state including position, orientation, and velocities.
            action_space (gym.spaces.Box): RL action space defining allowed linear velocities in x and y.
            observation_space (gym.spaces.Box): RL observation space containing robot state, waypoints, and LiDAR data.
            kinetic_dim (int): Dimension of the robot's kinematic state vector.
            lidar_start_indx (int): Index offset in the observation vector where LiDAR data begins.
            goal_pos (numpy.ndarray): Target goal position in the environment.
            current_num_step (int): Counter tracking the current step number of an episode.
            prev_obs (dict): Holds the previous observation of the environment.
            current_obs (dict): Holds the current observation, initialized via get_obs().

        Side Effects:
            - Initializes ROS2 communications and creates a ROS2 node.
            - Sets up publishers and subscribers for various topics (position, velocity, LiDAR, odometry, world control).
            - Initiates a separate executor thread to handle ROS2 callbacks concurrently.
            - Initializes Weights & Biases (wandb) for experiment tracking.

        """
        super(WheeledRobotEnv, self).__init__()
        self.max_step = max_step
        self.environment_shape = environment_shape
        self.lidar_dim = int(lidar_dim)
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        self.n_history_frame = int(n_history_frame)
        self.collision_threshold = collision_threshold
        self.n_waypoints = n_waypoints
        self.goal_threshold = goal_threshold
        self.reward_coeff = reward_coeff
        self.robot_max_lin_vel = robot_max_lin_vel
        self.real_time_factor = real_time_factor
        self.delta_t = delta_t

        # Robot dimensions
        self.wheel_seperation = 0.122
        self.wheel_base = 0.156
        self.wheel_radius = 0.026
        self.wheel_steering_y_offset = 0.03
        self.steering_track = self.wheel_seperation - 2*self.wheel_steering_y_offset

        rclpy.init()
        self.node = rclpy.create_node('wheeled_robot_env')
        self.clock = self.node.get_clock()
        self.loop_rate = self.node.create_rate(20, self.clock)
        self.vel_msg = Twist()
        self.pos = np.array([0,0,0,0], float)
        self.vel = np.array([0,0,0,0], float) #left_front, right_front, left_rear, right_rear

        # ROS2 topic for controlling the robot
        self.pub_pos = self.node.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.pub_vel = self.node.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)

        # ROS2 topic for lidar data
        self.current_lidar_data = np.full(self.lidar_dim, self.lidar_max_range, float)
        self.lidar_data = np.zeros((1 + self.n_history_frame)*self.lidar_dim) # Contains current lidar data and history lidar data
        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # ROS2 topic for odometry data
        self.robot_pose_sub = self.node.create_subscription(
            PoseArray,
            '/model/fws_robot/pose',
            self.robot_pose_callback,
            10
        )

        # ROS2 client for pause/unpause environment 
        self.control_world_client = self.node.create_client(ControlWorld, '/world/empty/control', qos_profile=qos_profile_system_default)
        while not self.control_world_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().info('Waiting for WorldControl service...')

        # Define RL spaces
        self.action_space = spaces.Box(low=np.array([-self.robot_max_lin_vel, -np.pi]), high=np.array([robot_max_lin_vel, np.pi]), dtype=np.float32) # x linear velocity, angular velocity

        # Observation space: 
        ## x, y, theta, x-linear velocity, y-linear velocity, angular velocity, distance to goal
        ## goal positions, waypoints positions
        ## current lidar data, history lidar data, goal position

        self.observation_space = gym.spaces.Box(
            low=np.array(
                [-self.environment_shape[0]/2, -self.environment_shape[1]/2, -np.pi/2, -self.robot_max_lin_vel, -self.robot_max_lin_vel, 0, 0] +
                [-self.environment_shape[0]/2, -self.environment_shape[1]/2]*(1 + self.n_waypoints) +
                [self.lidar_min_range]*self.lidar_dim*(1+self.n_history_frame)
            ),
            high=np.array(
                [self.environment_shape[0]/2, self.environment_shape[1]/2, np.pi/2, robot_max_lin_vel, robot_max_lin_vel, np.pi, np.sqrt(self.environment_shape[0]**2 + self.environment_shape[1]**2)] +
                [self.environment_shape[0]/2, self.environment_shape[1]/2]*(1 + self.n_waypoints) +
                [self.lidar_max_range]*self.lidar_dim*(1+self.n_history_frame)
            ),
        )
        self.kinetic_dim = 7
        self.lidar_start_indx = self.kinetic_dim + (self.n_waypoints+1)*2

        self.robot_init_pos = np.array([0.0, 0.0, 0.07])
        self.goal_pos:np.ndarray = np.array([-2.0, 4.0])
        self.current_num_step = 0
        self.prev_obs:dict[np.ndarray, np.ndarray] = None
        self.current_obs:dict[np.ndarray, np.ndarray] = self.get_obs(initialize=True)

        # Variable to store robot's position and kinematic information from ROS2
        self.robot_state = {
            'x': self.robot_init_pos[0],           # X position
            'y': self.robot_init_pos[1],           # Y position
            'theta': 0.0,       # Orientation angle (radians)
            'linear_vel_x': 0.0,  # Linear velocity
            'linear_vel_y': 0.0,  # Linear velocity
            'angular_vel': 0.0  # Angular velocity
        }

        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.node)

        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        rate = self.node.create_rate(2)

        wandb.init(
            project="learning robot navigation",
        )

        # self.spawn_cylinder('cylinder1',self.goal_pos[0],self.goal_pos[1], 0)

    def take_action(self, action):
        self.vel_msg.linear.x = float(action[0])
        self.vel_msg.angular.z = float(action[1])

        vel_steerring_offset = self.vel_msg.angular.z * self.wheel_steering_y_offset
        sign = np.sign(self.vel_msg.linear.x)

        self.vel[0] = sign*np.hypot(self.vel_msg.linear.x - self.vel_msg.angular.z*self.steering_track/2, self.vel_msg.angular.z*self.wheel_base/2) - vel_steerring_offset
        self.vel[1] = sign*np.hypot(self.vel_msg.linear.x + self.vel_msg.angular.z*self.steering_track/2, self.vel_msg.angular.z*self.wheel_base/2) + vel_steerring_offset
        self.vel[2] = sign*np.hypot(self.vel_msg.linear.x - self.vel_msg.angular.z*self.steering_track/2, self.vel_msg.angular.z*self.wheel_base/2) - vel_steerring_offset
        self.vel[3] = sign*np.hypot(self.vel_msg.linear.x + self.vel_msg.angular.z*self.steering_track/2, self.vel_msg.angular.z*self.wheel_base/2) + vel_steerring_offset

        a0 = 2*self.vel_msg.linear.x + self.vel_msg.angular.z*self.steering_track
        a1 = 2*self.vel_msg.linear.x - self.vel_msg.angular.z*self.steering_track

        if a0 != 0:
            self.pos[0] = np.arctan(self.vel_msg.angular.z*self.wheel_base/(a0))
        else:
            self.pos[0] = 0
            
        if a1 != 0:
            self.pos[1] = np.arctan(self.vel_msg.angular.z*self.wheel_base/(a1))
        else:
            self.pos[1] = 0

        self.pos[2] = -self.pos[0]
        self.pos[3] = -self.pos[1]
            
        pos_array = Float64MultiArray(data=self.pos) 
        vel_array = Float64MultiArray(data=self.vel)

        try:
            while rclpy.ok():
                self.control_world(pause=False)
                
                self.pub_pos.publish(pos_array)
                self.pub_vel.publish(vel_array)
                
                time.sleep(self.delta_t/self.real_time_factor)
                self.control_world(pause=True)

                break

        except KeyboardInterrupt:
            pass

    def robot_pose_callback(self, msg:PoseArray):
        # Update robot position
        if msg.poses:
            pose:Pose = msg.poses[-1]
            self.robot_state['x'] = pose.position.x
            self.robot_state['y'] = pose.position.y

            # Convert quaternion (x, y, z, w) to yaw angle
            q = pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_state['theta'] = np.arctan2(siny_cosp, cosy_cosp)
        else:
            self.node.get_logger().warn("Received empty pose array in robot_pose_callback")
            
        self.robot_state['linear_vel_x'] = self.vel_msg.linear.x
        self.robot_state['linear_vel_y'] = self.vel_msg.linear.y

        self.robot_state['angular_vel'] = self.vel_msg.angular.z

    def lidar_callback(self, msg):
        # Convert the list of range measurements from the LaserScan message to a NumPy array
        raw_laser_data = np.array(msg.ranges)
        self.current_lidar_data = np.clip(a=raw_laser_data, a_min=self.lidar_min_range, a_max=self.lidar_max_range)

    def control_world(self, pause:bool):
        request = ControlWorld.Request()
        request.world_control.pause = pause
        
        future = self.control_world_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        
        if not future.result():
            self.node.get_logger().error('WorldControl Service call failed')
        else:
            self.node.get_logger().info(f'Pause: {pause}')

    def get_obs(self, initialize:bool=False):
        # Use default high readings if lidar data isn't available or complete
        if initialize:
            self.current_lidar_data = np.full(self.lidar_dim, self.lidar_max_range, float)
            self.lidar_data = np.zeros((1 + self.n_history_frame)*self.lidar_dim)
            self.robot_state = {
                'x': self.robot_init_pos[0],
                'y': self.robot_init_pos[1],
                'theta': 0.0,
                'linear_vel_x': 0.0,
                'linear_vel_y': 0.0,
                'angular_vel': 0.0
            }

        else:
            self.lidar_data[self.lidar_dim:] = self.lidar_data[:-self.lidar_dim]
            self.lidar_data[:self.lidar_dim] = self.current_lidar_data
        
        # Pack the latest odometry information into an array
        d_goal = np.linalg.norm(np.array([self.robot_state['x'], self.robot_state['y']]) - self.goal_pos)
        odometry = np.array([
            self.robot_state['x'],
            self.robot_state['y'],
            self.robot_state['theta'],
            self.robot_state['linear_vel_x'],
            self.robot_state['linear_vel_y'],
            self.robot_state['angular_vel'],
            d_goal
        ], dtype=np.float32)

        # To-do: Setup way points
        if initialize:
            self.prev_obs = np.concatenate([odometry, self.goal_pos, self.goal_pos, self.lidar_data])
        
        else:
            self.prev_obs = self.current_obs.copy()
        
        self.current_obs = np.concatenate([odometry, self.goal_pos, self.goal_pos, self.lidar_data])
        return self.current_obs
       
    def step(self, action):
        observation = {}
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        wandb.log({
            "State / X-Coordinate": self.current_obs[0],
            "State / Y-Coordinate": self.current_obs[1],
            "State / Theta": self.current_obs[2],
            "State / Linear Velocity X": self.current_obs[3],
            "State / Linear Velocity Y": self.current_obs[4],
            "State / Angular Velocity": self.current_obs[5],
            "State / Distance to Goal": self.current_obs[6],
        })

        self.current_num_step +=1

        # Take action
        self.take_action(action)

        # Next observation
        observation = self.get_obs()

        # Calculate reward base on current state
        if self.collision_check():
            reward, reward_components = self.get_rewards(collise=True)
            terminated = True
        else:
            reward, reward_components = self.get_rewards(collise=False)
            
        if reward_components['goal'] == self.reward_coeff['goal']['reach']: # Reached goal
            terminated = True

        if self.current_num_step >= self.max_step:  # Time limit exceeded
            truncated = True

        # Step infomation
        info = {
            'state': self.robot_state,
            'action': action,
            'rewards': reward_components
        }

        wandb.log({
            "Actions / Linear-X Velocity": float(action[0]),
            "Actions / Linear-Y Velocity": float(action[1]),

            "Reward Sum": reward,
            "Reward Components / Collision ": reward_components.get("collision", 0),
            "Reward Components / Goal": reward_components.get("goal", 0),
            "Reward Components / Velocity": reward_components.get("velocity", 0),
            "Reward Components / Angular": reward_components.get("angular", 0)
        })

        return observation, reward, terminated, truncated, info
    
    def collision_check(self)->bool:
        lidar_data:np.ndarray = self.current_lidar_data
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
        robot_pos = self.current_obs[:2] # x,y
        goal_distance = np.linalg.norm(robot_pos - self.goal_pos)

        if goal_distance <= self.goal_threshold:
            reward_components['goal'] = self.reward_coeff['goal']['reach']
        else:
            robot_prev_pos = self.prev_obs[:2]
            prev_goal_distance = np.linalg.norm(robot_prev_pos - self.goal_pos)
            reward_components['goal'] = self.reward_coeff['goal']['coeff']*(prev_goal_distance-goal_distance)

        # To-do: Adjust
        # Reward velocity
        linear_vel = np.sqrt(self.current_obs[3]**2 + self.current_obs[4]**2) # linear velocity
        if linear_vel < 1:
            reward_components['velocity'] = -0.1

        reward_components['angular'] = -np.abs(self.current_obs[5])

        reward = sum(reward_components.values())

        return reward, reward_components

    def set_robot_pose(self, x:float, y:float, z:float):
        command = [
            "gz", "service", "-s", "/world/empty/set_pose",
            "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", 'name: "fws_robot", position: {x:0.0, y:0.0, z:0.07}'
        ]
        command[-1] = f'name: "fws_robot", position: {{x:{x}, y:{y}, z:{z}}}'

        # Run the command with updated coordinates
        result = subprocess.run(command, capture_output=True, text=True)

        if result.stdout=='data: true\n\n':
            print('Set robot position succesfully!')

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.set_robot_pose(self.robot_init_pos[0], self.robot_init_pos[1], self.robot_init_pos[2])
        self.control_world(pause=True)

        observation = self.get_obs(initialize=True)
        info = {}

        for key in self.reward_coeff.keys():
            info[key] = 0 # Log component rewards as 0

        return observation, info

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def spawn_cylinder(self, name: str, x: float, y: float, z: float, radius: float = 0.2, height: float = 0.5):
        """Spawns a cylinder in Gazebo at a given position."""
        client = self.node.create_client(SpawnEntity, "/world/default/create")
        
        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn("Waiting for SpawnEntity service...")
            client = self.node.create_client(SpawnEntity, "/world/default/create")


        request = SpawnEntity.Request()
        request.name = name
        request.xml = f"""
        <sdf version="1.7">
            <model name="{name}">
                <pose>{x} {y} {z} 0 0 0</pose>
                <static>false</static>
                <link name="link">
                    <visual name="visual">
                        <geometry>
                            <cylinder>
                                <radius>{radius}</radius>
                                <length>{height}</length>
                            </cylinder>
                        </geometry>
                        <material>
                            <ambient>0.8 0.3 0.3 1.0</ambient>
                        </material>
                    </visual>
                    <collision name="collision">
                        <geometry>
                            <cylinder>
                                <radius>{radius}</radius>
                                <length>{height}</length>
                            </cylinder>
                        </geometry>
                    </collision>
                </link>
            </model>
        </sdf>
        """
        request.robot_namespace = ""
        request.initial_pose.position.x = x
        request.initial_pose.position.y = y
        request.initial_pose.position.z = z
        request.initial_pose.orientation.w = 1.0

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is not None:
            self.node.get_logger().info(f"Successfully spawned {name}")
        else:
            self.node.get_logger().error(f"Failed to spawn {name}")
