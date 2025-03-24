import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from gymnasium import spaces
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import LaserScan
from typing import Optional
from nav_msgs.msg import Path, Odometry
import threading
from ros_gz_interfaces.srv import SpawnEntity, SetEntityPose, ControlWorld
from geometry_msgs.msg import Pose, PoseStamped
import subprocess
import wandb
from rclpy.qos import qos_profile_system_default
import time
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from nav2_simple_commander.robot_navigator import BasicNavigator
import warnings
from scipy.interpolate import RBFInterpolator
import pickle
import matplotlib.pyplot as plt

class WheeledRobotEnv(gym.Env):
    def __init__(self, max_step:int, environment_shape:list, lidar_dim:int, lidar_max_range:float, lidar_min_range:float, n_history_frame:int, collision_threshold:float, n_waypoints:int, goal_threshold:float, robot_max_lin_vel:float, robot_max_ang_vel:float, reward_coeff:dict, real_time_factor:float, delta_t:float, start_pos:list[float, float, float], goal_pos:list[float, float, float]):
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
        self.robot_max_ang_vel = robot_max_ang_vel
        self.real_time_factor = real_time_factor
        self.delta_t = delta_t

        rclpy.init()
        self.node = rclpy.create_node('wheeled_robot_env')
        self.vel_msg = Twist()

        # ROS2 topic for controlling the robot
        self.pub_twist = self.node.create_publisher(Twist, '/cmd_vel', 10)

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
        self.robot_odom_sub = self.node.create_subscription(
            Odometry,
            '/saye/odometry_with_covariance',
            self.robot_pose_callback,
            10
        )

        # ROS2 client for pause/unpause environment 
        self.control_world_client = self.node.create_client(ControlWorld, '/world/empty/control', qos_profile=qos_profile_system_default)
        while not self.control_world_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().info('Waiting for WorldControl service...')

        # Define RL spaces
        self.action_space = spaces.Box(low=np.array([-self.robot_max_lin_vel, -self.robot_max_ang_vel]), high=np.array([robot_max_lin_vel, self.robot_max_ang_vel]), dtype=np.float32) # x linear velocity, angular velocity

        # Observation space: 
        ## x, y, theta, x-linear velocity, y-linear velocity, angular velocity, distance to goal
        ## current lidar data, history lidar data, goal position
        ## goal positions, waypoints positions

        self.observation_space = gym.spaces.Box(
            low=np.array(
                [-self.environment_shape[0]/2, -self.environment_shape[1]/2, -np.pi/2, -self.robot_max_lin_vel, -self.robot_max_lin_vel, -self.robot_max_ang_vel, 0] +
                [self.lidar_min_range]*self.lidar_dim*(1+self.n_history_frame) +
                [-self.environment_shape[0]/2, -self.environment_shape[1]/2]*(1 + self.n_waypoints)
            ),
            high=np.array(
                [self.environment_shape[0]/2, self.environment_shape[1]/2, np.pi/2, robot_max_lin_vel, robot_max_lin_vel, self.robot_max_ang_vel, np.sqrt(self.environment_shape[0]**2 + self.environment_shape[1]**2)] +
                [self.lidar_max_range]*self.lidar_dim*(1+self.n_history_frame) +
                [self.environment_shape[0]/2, self.environment_shape[1]/2]*(1 + self.n_waypoints)
            ),
        )
        self.kinetic_dim = 7
        self.lidar_start_indx = self.kinetic_dim + (self.n_waypoints+1)*2

        self.robot_init_pos = np.array(start_pos, dtype=float)
        self.goal_pos:np.ndarray = np.array([goal_pos[0], goal_pos[1]], dtype=float)

        # Nav2 Simple Navigator for finding waypoints
        self.nav = BasicNavigator()
        if not hasattr(self, 'waypoints'):
            self.waypoints = self.get_waypoints()
        
        # Create an Arficial Potential Field to guild the robot to move along waypoints
        if not hasattr(self, 'apf'):
            self.apf = self.get_apf()
        
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

        wandb.init(
            project="learning robot navigation",
        )


    def take_action(self, action):
        self.vel_msg.linear.x = float(action[0])
        self.vel_msg.angular.z = float(action[1])

        try:
            while rclpy.ok():
                self.control_world(pause=False)
                
                self.pub_twist.publish(self.vel_msg)
                
                time.sleep(self.delta_t/self.real_time_factor)
                self.control_world(pause=True)

                break

        except KeyboardInterrupt:
            pass

    def robot_pose_callback(self, msg:Odometry):
        # Update robot position
        if msg.pose.pose:
            pose:Pose = msg.pose.pose
            self.robot_state['x'] = pose.position.x
            self.robot_state['y'] = pose.position.y

            # Convert quaternion (x, y, z, w) to yaw angle
            q = pose.orientation
            roll, pitch, yaw = euler_from_quaternion([
                q.x,
                q.y,
                q.z,
                q.w
            ])
            self.robot_state['theta'] = yaw
        else:
            self.node.get_logger().warn("Received empty pose array in robot_pose_callback")
            
        self.robot_state['linear_vel_x'] = msg.twist.twist.linear.x
        self.robot_state['linear_vel_y'] = msg.twist.twist.linear.y

        self.robot_state['angular_vel'] = msg.twist.twist.angular.z
        
    def lidar_callback(self, msg):
        # Convert the list of range measurements from the LaserScan message to a NumPy array
        raw_laser_data = np.array(msg.ranges)
        self.current_lidar_data = np.clip(a=raw_laser_data, a_min=self.lidar_min_range, a_max=self.lidar_max_range)

    def get_waypoints(self):
        self.node.get_logger().info('Waiting for Nav2 to activate...')
        self.nav.waitUntilNav2Active()

        init_pose = PoseStamped()
        init_pose.header.frame_id = 'map'
        init_pose.pose.position.x = self.robot_init_pos[0]
        init_pose.pose.position.y = self.robot_init_pos[1]
        q = quaternion_from_euler(0,0,0)
        init_pose.pose.orientation.x = q[0]
        init_pose.pose.orientation.y = q[1]
        init_pose.pose.orientation.z = q[2]
        init_pose.pose.orientation.w = q[3]

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = self.goal_pos[0]
        goal_pose.pose.position.y = self.goal_pos[1]
        q = quaternion_from_euler(0,0,0)
        goal_pose.pose.orientation.x = q[0]
        goal_pose.pose.orientation.y = q[1]
        goal_pose.pose.orientation.z = q[2]
        goal_pose.pose.orientation.w = q[3]
        
        self.nav.setInitialPose(init_pose)
        self.nav._waitForInitialPose()
        time.sleep(0.5)
        path = self.nav.getPath(init_pose, goal_pose)

        waypoints = []
        for w in path.poses:
            waypoints.append([w.pose.position.x, w.pose.position.y])
        waypoints = np.array(waypoints)
        self.raw_waypoints = waypoints

        if len(waypoints) >= self.n_waypoints:
            indices = np.linspace(0, len(waypoints) - 1, self.n_waypoints, dtype=int)
            waypoints = np.array(waypoints)[indices]
        else:
            self.node.get_logger().warn("Waypoints shape is not stable. Expected at least {} waypoints, but got {}.".format(self.n_waypoints, len(waypoints)))
            warnings.warn("Waypoints shape is not stable. Expected at least {} waypoints, but got {}.".format(self.n_waypoints, len(waypoints)), UserWarning)
            waypoints = np.array(waypoints)

        return waypoints
    
    def calculate_control_points(self, waypoints, goal_point):
        coordinate = []
        max_z = 0
        for wp in waypoints:
            # wp_scaled = [3 * (coord - min_val) / (max_val - min_val) for coord in wp]
            wp_z = np.linalg.norm(np.array(wp)-np.array(goal_point))
            max_z = max(max_z, wp_z)
            coordinate.append([wp[0], wp[1], wp_z])
            
        num_seg = 100 # Segment walls (map boundaries) into num_seg segments
        for i in range(num_seg):
            for j in range(num_seg):
                if i == 1 and j == 1:  # skip the center point
                    continue
                x = i * (self.environment_shape[0]/(num_seg-1))
                y = j * (self.environment_shape[1]/(num_seg-1))
                if x==0 or x==self.environment_shape[0] or y==0 or y==self.environment_shape[1]:
                    coordinate.append([
                        i * (self.environment_shape[0]/(num_seg-1)) - self.environment_shape[0]/2, 
                        j * (self.environment_shape[1]/(num_seg-1)) - self.environment_shape[1]/2, 
                        max_z
                    ])

        coordinate = np.array(coordinate)

        return coordinate
    
    def get_apf(self):
        control_points = self.calculate_control_points(self.waypoints, self.goal_pos)

        xy = control_points[:, :2]  # (x, y) coordinates
        z = control_points[:, 2]    # z values

        # Fit RBF interpolator
        rbf = RBFInterpolator(xy, z, kernel='inverse_multiquadric', epsilon=1)

        # Generate a grid to evaluate the surface
        # x_vals = np.linspace(0, 20, 100)
        # y_vals = np.linspace(0, 20, 100)
        # X, Y = np.meshgrid(x_vals, y_vals)
        # Z = rbf(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)

        # # Plot the RBF-interpolated surface
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

        # # Plot the control points
        # ax.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2], color='red', marker='o', label="Control Points")

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_title("Artificial Potential Field")
        # ax.legend()
        # fig.savefig('apf.png')
        # plt.show()
        with open("control_points.pkl", "wb") as f:
            pickle.dump(control_points, f)
        print('saved waypoints')

        return rbf
        
    def control_world(self, pause:bool):
        request = ControlWorld.Request()
        request.world_control.pause = pause
        
        future = self.control_world_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        
        while not future.result():
            self.node.get_logger().error('WorldControl Service call failed')

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

        if initialize:
            self.prev_obs = np.concatenate([odometry, self.lidar_data, self.goal_pos.flatten(), self.waypoints.flatten()])
        
        else:
            self.prev_obs = self.current_obs.copy()

        self.current_obs = np.concatenate([odometry, self.lidar_data, self.goal_pos.flatten(), self.waypoints.flatten()])
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
            "Actions / Angular-Z Velocity": float(action[1]),
            "Reward Sum": reward,
            "Reward Components / Collision ": reward_components.get("collision", 0),
            "Reward Components / Goal": reward_components.get("goal", 0),
            "Reward Components / Velocity": reward_components.get("velocity", 0),
            "Reward Components / Angular": reward_components.get("angular", 0),
            "Reward Components / Waypoint": reward_components.get("waypoint", 0)
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
        robot_prev_pos = self.prev_obs[:2]
        distance_to_goal = np.linalg.norm(robot_pos - self.goal_pos)

        if distance_to_goal <= self.goal_threshold:
            reward_components['goal'] = self.reward_coeff['goal']['reach']
        else:
            prev_distance_to_goal = np.linalg.norm(robot_prev_pos - self.goal_pos)
            reward_components['goal'] = self.reward_coeff['goal']['coeff']*(prev_distance_to_goal - distance_to_goal)
            
        prev_waypoints_score = self.apf([robot_prev_pos])
        waypoints_score = self.apf([robot_pos])
        reward_components['waypoint'] = self.reward_coeff['waypoint']['coeff'] * ((prev_waypoints_score - waypoints_score)).item()

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
            "--req", 'name: "saye", position: {x:0.0, y:0.0, z:0.07}'
        ]
        command[-1] = f'name: "saye", position: {{x:{x}, y:{y}, z:{z}}}'

        # Run the command with updated coordinates
        result = subprocess.run(command, capture_output=True, text=True)

        if result.stdout=='data: true\n\n':
            print('Set robot position succesfully!')

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.set_robot_pose(self.robot_init_pos[0], self.robot_init_pos[1], self.robot_init_pos[2])
        # self.control_world(pause=False)
        # self.get_waypoints()
        self.control_world(pause=True)

        observation = self.get_obs(initialize=True)
        info = {}

        for key in self.reward_coeff.keys():
            info[key] = 0 # Log component rewards as 0

        return observation, info

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()