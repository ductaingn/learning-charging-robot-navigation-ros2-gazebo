import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from gymnasium import spaces
from geometry_msgs.msg import Twist, PoseArray
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan
from typing import Optional
from squaternion import Quaternion
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import torch
import threading
from ros_gz_interfaces.srv import SpawnEntity, SetEntityPose
from ros_gz_interfaces.msg import Entity, WorldControl
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
import subprocess
import wandb

class WheeledRobotEnv(gym.Env):
    def __init__(self, max_step:int, environment_shape:list, lidar_dim:int, lidar_max_range:float, lidar_min_range:float, collision_threshold:float, goal_threshold:float, robot_max_lin_vel:float, reward_coeff:dict):
        super(WheeledRobotEnv, self).__init__()
        self.max_step = max_step
        self.environment_shape = environment_shape
        self.lidar_dim = lidar_dim
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
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
        self.clock = self.node.get_clock()
        self.loop_rate = self.node.create_rate(20, self.clock)
        self.vel_msg = Twist()
        self.pos = np.array([0,0,0,0], float)
        self.vel = np.array([0,0,0,0], float) #left_front, right_front, left_rear, right_rear

        # ROS2 topic for controlling the robot
        self.pub_pos = self.node.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.pub_vel = self.node.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        # self.timer = self.node.create_timer(timer_period_sec=1, callback=self.timer_callback)

        # ROS2 topic for lidar data
        self.laser_data = None
        self.laser_sub = self.node.create_subscription(
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

        # ROS2 topic for pause/unpause environment 
        self.pause_pub = self.node.create_publisher(WorldControl, '/world/empty/control', 10)

        # Variable to store robot's position and kinematic information from ROS2 (position, orientation, velocities)
        self.robot_state = {
            'x': 0.0,           # X position
            'y': 0.0,           # Y position
            'theta': 0.0,       # Orientation angle (radians)
            'linear_vel_x': 0.0,  # Linear velocity
            'linear_vel_y': 0.0,  # Linear velocity
            'angular_vel': 0.0  # Angular velocity
        }

        # Define RL spaces
        self.action_space = spaces.Box(low=np.array([-self.robot_max_lin_vel, -self.robot_max_lin_vel]), high=np.array([robot_max_lin_vel, self.robot_max_lin_vel]), dtype=np.float32) # linear velocity: x,y

        self.observation_space = gym.spaces.Dict(
            {
                "odometry": gym.spaces.Box(
                    low=np.array([-self.environment_shape[0]/2, -self.environment_shape[1]/2, -np.pi/2, 0, 0, 0,0]),
                    high=np.array([self.environment_shape[0]/2, self.environment_shape[1]/2, np.pi/2, robot_max_lin_vel, robot_max_lin_vel, np.pi, 100])
                ),
                "lidar": gym.spaces.Box(self.lidar_min_range, self.lidar_max_range, shape=(self.lidar_dim, ))
            }
        )

        self.goal_pos:np.ndarray = np.array([15.0, -2.0])
        self.current_num_step = 0
        self.prev_obs:dict[np.ndarray, np.ndarray] = None
        self.current_obs:dict[np.ndarray, np.ndarray] = self.get_obs(initialize=True)

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
        self.vel_msg.linear.y = float(action[1])
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

        try:
            while rclpy.ok():
                self.pub_pos.publish(pos_array)
                self.pub_vel.publish(vel_array)
                self.loop_rate.sleep()  
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

        if(self.vel_msg.linear.x != 0):
            ang = self.vel_msg.linear.y / self.vel_msg.linear.x
        else:
            ang = 0
        self.robot_state['angular_vel'] = np.arctan(ang)

    def lidar_callback(self, msg):
        # Convert the list of range measurements from the LaserScan message to a NumPy array
        raw_laser_data = np.array(msg.ranges)
        self.laser_data = np.clip(a=raw_laser_data, a_min=self.lidar_min_range, a_max=self.lidar_max_range)

    def pause_sim(self):
        msg = WorldControl()
        msg.pause = True
        self.pause_pub.publish(msg)

    def unpause_sim(self):
        msg = WorldControl()
        msg.pause = False
        self.pause_pub.publish(msg)

    def get_obs(self, initialize:bool=False):
        # Use default high readings if lidar data isn't available or complete
        if self.laser_data is None or len(self.laser_data) < self.lidar_dim:
            lidar_data = np.full((self.lidar_dim,), self.lidar_max_range, dtype=np.float32)
        else:
            lidar_data = np.array(self.laser_data[:self.lidar_dim], dtype=np.float32)
                
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
            self.prev_obs = {
                'odometry': odometry, 
                'lidar': lidar_data
            }
        
        else:
            self.prev_obs = {
                'odometry': self.current_obs['odometry'].copy(), 
                'lidar': self.current_obs['lidar'].copy()
            }
        
        self.current_obs = {
            'odometry': odometry, 
            'lidar': lidar_data
        }

        return self.current_obs
       
    def step(self, action):
        observation = {}
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        wandb.log({
            "State / X-Coordinate": self.current_obs["odometry"][0],
            "State / Y-Coordinate": self.current_obs["odometry"][1],
            "State / Theta": self.current_obs["odometry"][2],
            "State / Linear Velocity X": self.current_obs["odometry"][3],
            "State / Linear Velocity Y": self.current_obs["odometry"][4],
            "State / Angular Velocity": self.current_obs["odometry"][5],
            "State / Distance to Goal": self.current_obs["odometry"][6],
        })

        self.current_num_step +=1

        # Take action
        self.unpause_sim()
        self.take_action(action)
        self.pause_sim()

        # Next observation
        observation = self.get_obs()

        # Calculate reward base on current state
        if self.collision_check(self.current_obs):
            reward, reward_components = self.get_rewards(collise=True)
            terminated = True
        else:
            reward, reward_components = self.get_rewards(collise=False)
            
        # if self.current_num_step >= self.max_step:  # Time limit exceeded
        #     truncated = True

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
            robot_prev_pos = self.prev_obs['odometry'][:2]
            prev_goal_distance = np.linalg.norm(robot_prev_pos - self.goal_pos)
            reward_components['goal'] = self.reward_coeff['goal']['coeff']*(prev_goal_distance-goal_distance)

        # To-do: Adjust
        # Reward velocity
        linear_vel = np.sqrt(self.current_obs['odometry'][3]**2 + self.current_obs['odometry'][4]**2) # linear velocity
        if linear_vel < 1:
            reward_components['velocity'] = -0.1

        reward_components['angular'] = -np.abs(self.current_obs['odometry'][5])

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
        self.set_robot_pose(0.0, 0.0, 0.07)

        observation = self.get_obs()
        info = {}

        observation = self.get_obs()
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
