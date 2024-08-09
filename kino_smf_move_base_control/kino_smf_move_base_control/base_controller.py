import rclpy
import rclpy.node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from math import atan2, pi
import math
from numpy import nanmin
from path_follower.node import Node
import time
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformListener, Buffer

move_tolerance = 0.2
scan_tolerance_front = 0.4
scan_tolerance_side = 0.3
rotate_tolerance = 0.004


class PathFollower:
    """Follows the path obtained to one of its topics"""

    def __init__(self):

        self.node_ = rclpy.node.Node("path_follower_node")

        self.robot_position = None  # Set from planner
        self.path_deviation = 0.0  # Used to avoid obstacles

        # self.planner = planner
        self.goal = PoseStamped()

        self.following_path = []

        self.is_shutdown_initiated = False
        self.is_moving = False
        self.is_obstacle_ahead = False

        self.goal_tolerance = self.node_.declare_parameter("goal_tolerance", 0.1).value
        self.max_trans_vel = self.node_.declare_parameter("max_trans_vel", 0.25).value
        self.max_rot_vel = self.node_.declare_parameter("max_rot_vel", 1.5).value

        self.cmd_vel_topic = self.node_.declare_parameter(
            "control_output_topic", "/cmd_vel"
        ).value

        self.path_topic = self.node_.declare_parameter(
            "control_path_topic", "/path"
        ).value

        self.goal_topic = self.node_.declare_parameter(
            "query_goal_topic", "/goal"
        ).value

        self.sweep_motion_topic = self.node_.declare_parameter(
            "sweep_motion_topic", "/sweep_motion"
        ).value

        self.robot_frame = self.node_.declare_parameter(
            "robot_frame", "base_footprint"
        ).value

        self.new_path_arrived = False

        self.tf_buffer = Buffer()
        self.transform_listener = TransformListener(self.tf_buffer, self.node_)

        # ! Subscribers
        self.mover_subscriber = self.node_.create_subscription(
            Path, self.path_topic, self.path_callback, 1
        )
        self.scan_subscriber = self.node_.create_subscription(
            LaserScan, "/scan", self.scan_callback, 1
        )
        self.motion_stop_subscriber = self.node_.create_subscription(
            Bool, "/stop_motion", self.stop_motion_callback, 1
        )
        self.start_subscriber = self.node_.create_subscription(
            TFMessage, "/tf", self.robot_pose_callback, 1
        )
        self.goal_subscriber = self.node_.create_subscription(
            PoseStamped, self.goal_topic, self.goal_callback, 1
        )
        self.sweep_motion_subscriber = self.node_.create_subscription(
            Bool, self.sweep_motion_topic, self.sweep_motion, 1
        )

        # ! Publishers
        self.velocity_publisher = self.node_.create_publisher(
            Twist, self.cmd_vel_topic, 10
        )
        self.motion_publisher = self.node_.create_publisher(
            Bool, "/robot_is_moving", 10
        )
        self.goal_reached_publisher = self.node_.create_publisher(
            Bool, "/goal_reached", 10
        )

    def goal_callback(self, data: PoseStamped):
        """Obtain goal from query"""
        self.goal = Node.from_pose(data.pose)

    def robot_pose_callback(self, data: TFMessage):
        """Listens to robot current pose"""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            position = transform.transform.translation
            quaternion = transform.transform.rotation
        except:
            return

        position = [
            position.x,
            position.y,
            position.z,
        ]
        orientation = [
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        ]

        self.robot_position = Node.from_tf(position, orientation)

    def stop_motion_callback(self, stop_motion: Bool):
        """Listens to a flag if the motions has to be stopped"""
        if stop_motion.data:
            self.stop_moving()
            self.initialize_stop()

    def path_callback(self, path: Path):
        """Listens to obtained solution path from path planner"""

        self.new_path_arrived = True
        node_path = []

        nearest_node = 0
        distance = float("inf")

        if len(path.poses) > 0:
            self.goal = Node.from_pose(path.poses[-1].pose)

        for i in range(0, len(path.poses)):
            node = Node.from_pose(path.poses[i].pose)
            node_path.append(node)
            node_distance = node.calculate_distance(self.robot_position)
            if node_distance < distance:
                distance = node_distance
                nearest_node = i

        node_path = node_path[nearest_node + 1 :]
        self.following_path = node_path
        self.is_shutdown_initiated = False

    def scan_callback(self, scan_data: LaserScan):
        """Listens to the scan obtained from the lidar"""
        if (
            nanmin(scan_data.ranges[0:10] + scan_data.ranges[350:360])
            < scan_tolerance_front
        ):
            self.is_obstacle_ahead = False
        elif nanmin(scan_data.ranges[11:165]) < scan_tolerance_side:
            self.path_deviation = 0
        elif nanmin(scan_data.ranges[195:349]) < scan_tolerance_side:
            self.path_deviation = 0
        else:
            self.path_deviation = 0

    def initialize_stop(self):
        self.is_shutdown_initiated = True

    def stop_moving(self):
        self.following_path = []
        self.is_shutdown_initiated = False
        self.is_moving = False
        self.velocity_publisher.publish(Twist())

    def follow_path(self):
        self.is_moving = True

        while len(self.following_path) > 0:
            if self.is_shutdown_initiated:
                self.stop_moving()
                return
            node_to_follow = self.following_path.pop(0)
            self.move_to_point(node_to_follow)

        if (
            math.sqrt(
                math.pow(self.goal.x - self.robot_position.x, 2)
                + math.pow(self.goal.y - self.robot_position.y, 2)
            )
            < self.goal_tolerance + 0.1
        ):
            self.velocity_publisher.publish(Twist())  # Stopping robot
            self.rotate_to_goal(self.goal)
            self.velocity_publisher.publish(Twist())

            # self.goal_reached_publisher.publish(Bool(data=True))
            self.node_.get_logger().info("Goal has been reached")
            self.is_moving = False

    def go_back(self):
        current_distance = 0.0
        vel_msg = Twist()
        vel_msg.linear.x = -float(self.max_trans_vel)
        t0 = self.node_.get_clock().now().seconds_nanoseconds()[0]

        while current_distance < 0.4:
            if self.is_shutdown_initiated:
                self.is_obstacle_ahead = False
                return

            self.velocity_publisher.publish(vel_msg)
            t1 = self.node_.get_clock().now().seconds_nanoseconds()[0]
            current_distance = self.max_trans_vel * (t1 - t0)
            # time.sleep(0.001)
            rclpy.spin_once(self.node_)

        self.is_obstacle_ahead = False

    def move_to_point(self, point: Node):
        vel_msg = Twist()
        vel_msg.linear.x = float(self.max_trans_vel)

        while self.robot_position.calculate_distance(point) > move_tolerance:
            if (
                self.is_shutdown_initiated
                or self.is_obstacle_ahead
                or self.new_path_arrived
            ):
                self.new_path_arrived = False
                return

            ang_error = self.angular_difference(point)

            speed = self.max_rot_vel * ang_error
            vel_msg.angular.z = float(
                min(self.max_rot_vel, speed) + self.path_deviation
            )

            if abs(ang_error) > 1.6:
                vel_msg.linear.x = 0.0
            else:
                speed = self.max_trans_vel * (1 - (abs(ang_error) / (math.pi)))
                vel_msg.linear.x = float(speed)

            self.velocity_publisher.publish(vel_msg)
            # time.sleep(0.001)
            rclpy.spin_once(self.node_)

    def rotate_to_goal(self, goal: Node):
        if isinstance(goal, PoseStamped):
            goal = Node.from_pose(goal.pose)

        vel_msg = Twist()

        while abs(goal.theta - self.robot_position.theta) > rotate_tolerance:
            if self.is_shutdown_initiated:
                self.stop_moving()
                return

            speed = self.max_rot_vel * (goal.theta - self.robot_position.theta)
            vel_msg.angular.z = float(min(self.max_rot_vel, speed))
            self.velocity_publisher.publish(vel_msg)
            # time.sleep(0.001)
            rclpy.spin_once(self.node_)

        self.velocity_publisher.publish(Twist())

    def sweep_motion(self, msg):
        time_sweep = 2 * math.pi / (self.max_rot_vel / 2)

        vel_msg = Twist()
        vel_msg.angular.z = float(self.max_rot_vel / 2)

        init_time = time.time()

        while time.time() - init_time < time_sweep:
            self.velocity_publisher.publish(vel_msg)
            # time.sleep(0.001)
            rclpy.spin_once(self.node_)

        vel_msg.angular.z = 0.0

        self.velocity_publisher.publish(vel_msg)

    def angular_difference(self, point: Node) -> float:
        angle = (
            atan2(point.y - self.robot_position.y, point.x - self.robot_position.x)
            - self.robot_position.theta
        )

        if angle <= -pi:  # Normalizing angle
            angle += 2 * pi
        elif angle > pi:
            angle -= 2 * pi

        return angle


def main(args=None):
    rclpy.init(args=args)

    controller = PathFollower()

    try:
        while rclpy.ok():
            rclpy.spin_once(controller.node_)
            if len(controller.following_path) > 0:
                controller.follow_path()

    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
