import math
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _yaw_from_quaternion(q) -> float:
    """Compute yaw (around Z) from quaternion message parts."""
    w = q.w
    x = q.x
    y = q.y
    z = q.z
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class LimoController(Node):
    def __init__(self):
        super().__init__('limo_controller')
        if not rclpy.ok():
            rclpy.init()

        # Declare parameters with defaults matching original script
        # This parameters can be changed dynamically using e.g. :
        #   ros2 param set /limo_controller target_x 2.5
        self.declare_parameter('target_x', 1.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_theta', 0.0)
        self.declare_parameter('krho', 0.05)
        self.declare_parameter('kalpha', 2.0)
        self.declare_parameter('kbeta', -2.0)
        self.declare_parameter('period', 0.1)
        self.declare_parameter('max_linear_vel', 0.5)

        self.target_x = float(self.get_parameter('target_x').value)
        self.target_y = float(self.get_parameter('target_y').value)
        self.target_theta = float(self.get_parameter('target_theta').value)
        self.krho = float(self.get_parameter('krho').value)
        self.kalpha = float(self.get_parameter('kalpha').value)
        self.kbeta = float(self.get_parameter('kbeta').value)
        self.period = float(self.get_parameter('period').value)
        self.max_linear_vel = float(self.get_parameter('max_linear_vel').value)
        self.max_angular_vel = 2 * self.max_linear_vel

        qos_profile = rclpy.qos.QoSProfile(depth=10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10) 

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.odom_received = False

        self.timer = self.create_timer(self.period, self.control_loop)

        self.get_logger().info('LimoController started. Target: (%.3f, %.3f, %.3f)' % (self.target_x, self.target_y, self.target_theta))

    def odom_callback(self, msg: Odometry) -> None:
        # Update pose from odometry
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.theta = float(_yaw_from_quaternion(msg.pose.pose.orientation))
        self.odom_received = True

    def control_loop(self) -> None:
        if not self.odom_received:
            # Wait until we have odometry
            return

        # 1. Compute relative position to target (Cartesian)
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist_to_goal = math.hypot(dx, dy)
        
        # 2. Compute the absolute angle of the line connecting robot to goal
        # This must remain independent of target_theta
        angle_to_goal = math.atan2(dy, dx)

        # 3. Compute control errors in polar coordinates (Astolfi/Siegwart formulation)
        # Alpha is the error between robot heading and the line of sight
        Alpha = _normalize_angle(angle_to_goal - self.theta)
        
        # Beta is the error between the line of sight and the final target heading
        Beta = _normalize_angle(self.target_theta - angle_to_goal)

        # 4. Apply control law
        # v_lin drives the robot forward, Gamma handles the rotation
        v_lin = self.krho * dist_to_goal
        Gamma = self.kalpha * Alpha + self.kbeta * Beta

        # Special condition: if the robot is very close to the target, 
        # Alpha and Beta become unstable (atan2 of very small numbers).
        # We switch to pure rotation to fix the final heading.
        if dist_to_goal < 0.05:
            v_lin = 0.0
            Gamma = self.kalpha * _normalize_angle(self.target_theta - self.theta)

        # Limit velocities
        v_lin = max(-self.max_linear_vel, min(self.max_linear_vel, v_lin))
        Gamma = max(-self.max_angular_vel, min(self.max_angular_vel, Gamma))

        # Construct Twist message
        cmd = Twist()
        cmd.linear.x = float(v_lin)
        cmd.angular.z = float(Gamma)

        self.cmd_pub.publish(cmd)
        self.odom_received = False

        self.get_logger().info(f"\tRobot pose: x={self.x:.4f}, y={self.y:.4f}, theta={self.theta:.4f} \tDistance to goal: {dist_to_goal:.4f}")
        if dist_to_goal < 0.01 and abs(_normalize_angle(self.target_theta - self.theta)) < 0.05:
            self.get_logger().info('Target pose fully reached!')


def main(args=None):
    rclpy.init(args=args)
    node = LimoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
