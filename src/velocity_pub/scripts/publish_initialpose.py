#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import quaternion_from_euler
import sys
from example_interfaces.srv import Trigger
import time

class InitialPosePublisher(Node):
    def __init__(self, x, y, z, roll, pitch, yaw):
        super().__init__('initial_pose_publisher')
        self.publisher = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.timer = self.create_timer(2.0, self.publish_initial_pose)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.confirm_pose_received, 10)

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    def confirm_pose_received(self, msg):
        self.get_logger().info("AMCL has received the initial pose!")
        self.pose_published = True

    def publish_initial_pose(self):
        while self.publisher.get_subscription_count() == 0:
            self.get_logger().warn("Waiting for subscribers on /initialpose...")
            time.sleep(1)  # Small delay to prevent spamming

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = self.z
        q = quaternion_from_euler(self.roll, self.pitch, self.yaw, 'sxyz')
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        self.publisher.publish(msg)
        self.get_logger().info("Published new initial pose")
        self.timer.cancel()
        self.destroy_node()

def parse_arg(args, expected_keys):
    kwargs = {}

    for i in range(0, len(args) - 1, 2):
        key = args[i]
        if key in expected_keys:
            try:
                kwargs[key] = float(args[i + 1])
            except ValueError:
                raise ValueError(f"Invalid value for {key}: {args[i + 1]}")
                
    missing = [k for k in expected_keys if k not in kwargs]
    if missing:
        raise ValueError("Missing required parameters: " + ", ".join(missing))

    return kwargs

def main(args=None):
    rclpy.init()

    # Process args in key-value pairs
    expected_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    kwargs = parse_arg(args, expected_keys)

    node = InitialPosePublisher(**kwargs)
    rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main(sys.argv[1:])
