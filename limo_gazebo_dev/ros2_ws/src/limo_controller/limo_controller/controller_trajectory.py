#!/usr/bin/env python3

import math
import time
from scipy.spatial import KDTree

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException

from limo_interfaces.action import FollowSequencePlan
from . utils.pursuit_pt import PursuitPoint

# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def angle_diff(a: float, b: float) -> float:
    d = a - b
    while d >  math.pi: d -= 2.0 * math.pi
    while d < -math.pi: d += 2.0 * math.pi
    return d

def pose_from_tf(tf):
    x   = tf.transform.translation.x
    y   = tf.transform.translation.y
    q   = tf.transform.rotation
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )
    return x, y, yaw

# ─────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────

class FollowSequencePlanServer(Node):

    def __init__(self):
        super().__init__('follow_sequence_plan_server')

        self.declare_parameter('cmd_vel_topic',          '/cmd_vel')
        self.declare_parameter('robot_frame',            'base_footprint')
        self.declare_parameter('world_frame',            'odom')

        self.declare_parameter('pursuit_speed',           0.15)   # m/s del punto mobile
        self.declare_parameter('d_star',                  0.0)    # distanza di inseguimento [m]
        self.declare_parameter('kh_linear',               1.50)   # K_h per il PI lineare
        self.declare_parameter('ki_linear',               0.10)   # K_i per il PI lineare
        self.declare_parameter('kh_angular',              1.50)   # K_h per il P angolare

        self.declare_parameter('goal_radius',             0.15)   # m — robot vicino al goal
        self.declare_parameter('goal_pause_sec',          2.0)
        self.declare_parameter('control_rate',           20.0)    # Hz

        self.declare_parameter('max_linear_vel',          0.20)   # m/s
        self.declare_parameter('max_angular_vel',         1.20)   # rad/s
        self.declare_parameter('integral_windup_limit',   1.0)

        self.gain_base = 10

        self.cmd_vel_topic  = self.get_parameter('cmd_vel_topic').value
        self.robot_frame    = self.get_parameter('robot_frame').value
        self.world_frame    = self.get_parameter('world_frame').value

        self.max_v          = self.get_parameter('max_linear_vel').value
        self.max_omega      = self.get_parameter('max_angular_vel').value
        self.windup_limit   = self.get_parameter('integral_windup_limit').value

        self.gain_coeff = self.max_v * self.gain_base

        self.pursuit_speed  = self.gain_coeff * self.get_parameter('pursuit_speed').value
        self.d_star         = self.get_parameter('d_star').value
        self.kh_lin         = self.gain_coeff * self.get_parameter('kh_linear').value
        self.ki_lin         = self.gain_coeff * self.get_parameter('ki_linear').value
        self.kh_ang         = self.get_parameter('kh_angular').value

        self.goal_radius    = self.get_parameter('goal_radius').value
        self.pause_sec      = self.get_parameter('goal_pause_sec').value
        self.control_rate   = self.get_parameter('control_rate').value
        self.dt             = 1.0 / self.control_rate

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.debug_pub  = self.create_publisher(PoseStamped, '/limo/control/moving_ref', 10)
        self.nn_pub = self.create_publisher(PoseStamped, '/limo/control/nearest_point', 10)
        self.cmd_pub    = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self._cb_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            FollowSequencePlan,
            '/follow_sequence_plan',
            execute_callback  = self.execute_callback,
            goal_callback     = self.goal_callback,
            cancel_callback   = self.cancel_callback,
            callback_group    = self._cb_group
        )

        # Switch logic for stopping the robot

        self._enabled = True

        self.enable_sub = self.create_subscription(
            Bool,
            '/mission/enable',
            self._enable_callback,
            10
        )

        self.get_logger().info('FollowSequencePlanServer ready.')

    # ── ACTION CALLBACKS ──────────────────────

    def goal_callback(self, goal_request):
        self.get_logger().info('New goal request received.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel requested.')
        self._stop_robot()
        return CancelResponse.ACCEPT

    # ── PAUSE CALLBACK ────────────────────────

    def _enable_callback(self, msg: Bool):

        self._enabled = msg.data

        if not self._enabled:
            self.get_logger().warn("PAUSE")
            self._stop_robot()
        else:
            self.get_logger().info("RESUME")

    # ── EXECUTE ───────────────────────────────

    def execute_callback(self, goal_handle: ServerGoalHandle):

        ordered_goals = goal_handle.request.ordered_goals
        paths         = goal_handle.request.paths

        all_dist_errors  = []
        all_ang_errors   = []
        all_lin_vels     = []
        all_ang_vels     = []
        goal_reach_times = []

        feedback_msg = FollowSequencePlan.Feedback()
        rate         = self.create_rate(self.control_rate)
        start_time   = self.get_clock().now().nanoseconds * 1e-9

        # ══════════════════════════════════════
        #  GOAL LOOP
        # ══════════════════════════════════════
        for goal_idx, (goal_pose, ros_path) in enumerate(zip(ordered_goals, paths)):

            # attesa resume con cancel check
            while not self._enabled and rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self._stop_robot()
                    goal_handle.canceled()
                    return FollowSequencePlan.Result()
                self._stop_robot()
                rate.sleep()

            self.get_logger().info(f'→ Goal {goal_idx + 1}/{len(ordered_goals)}')

            goal_x = goal_pose.pose.position.x
            goal_y = goal_pose.pose.position.y

            waypoints = [(ps.pose.position.x, ps.pose.position.y)
                         for ps in ros_path.poses]

            if len(waypoints) < 2:
                self.get_logger().warn(f'Goal {goal_idx+1}: path troppo corto, skip.')
                continue

            # ── crea il pursuit point per questo segmento ──
            kd_tree = KDTree(waypoints)
            pp = PursuitPoint(waypoints, self.pursuit_speed)
            integral_err = 0.0

            # ── CONTROL LOOP ──
            while rclpy.ok():
                
                self.get_logger().info(f"[LOOP] enabled={self._enabled} cancel={goal_handle.is_cancel_requested}") 

                # ── CANCEL: sempre prioritario ─────────────────────
                if goal_handle.is_cancel_requested:
                    self.get_logger().warn("[LOOP] CANCEL DETECTED → uscita")
                    self._stop_robot()
                    goal_handle.canceled()
                    return FollowSequencePlan.Result()

                # ── PAUSE ──────────────────────────────────────────
                if not self._enabled:
                    self._stop_robot()
                    rate.sleep()
                    continue

                # robot state
                try:
                    tf = self.tf_buffer.lookup_transform(
                        self.world_frame, self.robot_frame, rclpy.time.Time()
                    )
                except TransformException as e:
                    self.get_logger().warn(f'TF not available: {e}')
                    rate.sleep()
                    continue

                rx, ry, ryaw = pose_from_tf(tf)

                # ── NEAREST POINT (solo per logging) ──────────────────────────
                _, nn_idx = kd_tree.query([rx, ry])
                nn_x, nn_y = waypoints[nn_idx]
                dx_nn   = nn_x - rx
                dy_nn   = nn_y - ry
                e_lin_log = math.hypot(dx_nn, dy_nn)

                # gradiente locale del path nel punto più vicino
                window = 5
                i0 = max(0, nn_idx - window)
                i1 = min(len(waypoints) - 1, nn_idx + window)

                dx = waypoints[i1][0] - waypoints[i0][0]
                dy = waypoints[i1][1] - waypoints[i0][1]

                theta_path = math.atan2(dy, dx)
                e_ang_log  = angle_diff(theta_path, ryaw)

                nn_msg = PoseStamped()
                nn_msg.header.stamp = self.get_clock().now().to_msg()
                nn_msg.header.frame_id = self.world_frame
                nn_msg.pose.position.x = nn_x
                nn_msg.pose.position.y = nn_y
                nn_msg.pose.position.z = 0.0
                # quaternion da theta_path
                nn_msg.pose.orientation.z = math.sin(theta_path / 2.0)
                nn_msg.pose.orientation.w = math.cos(theta_path / 2.0)
                self.nn_pub.publish(nn_msg)

                # avanza il pursuit point solo se non è già arrivato
                if not pp.done:
                    pp.advance(self.dt)

                ref_msg = PoseStamped()

                ref_msg.header.stamp = self.get_clock().now().to_msg()
                ref_msg.header.frame_id = self.world_frame

                ref_msg.pose.position.x = pp.x
                ref_msg.pose.position.y = pp.y
                ref_msg.pose.position.z = 0.0

                ref_msg.pose.orientation.w = 1.0

                self.debug_pub.publish(ref_msg)

                # ── ERRORI ────────────────────────────────────────────────
                # errore lineare: distanza dal pursuit point meno d*
                dx = pp.x - rx
                dy = pp.y - ry
                dist_pp = math.hypot(dx, dy)
                e_lin   = dist_pp - self.d_star          # errore per il PI

                # direzione verso il pursuit point
                theta_star = math.atan2(dy, dx)
                e_ang      = angle_diff(theta_star, ryaw)

                # distanza dal goal finale (per la condizione di stop)
                dist_to_goal = math.hypot(goal_x - rx, goal_y - ry)

                # ── LEGGI DI CONTROLLO (slide) ─────────────────────────────
                # v* = K_h * e + K_i * ∫e dt   (PI lineare)
                integral_err = max(-self.windup_limit,
                                   min(self.windup_limit,
                                       integral_err + e_lin * self.dt))
                v_star = self.kh_lin * e_lin + self.ki_lin * integral_err
                v_star = max(0.0, min(self.max_v, v_star))   # no retromarcia

                # γ = K_h * (θ* ⊖ θ)   (P angolare)
                gamma = self.kh_ang * e_ang
                gamma = max(-self.max_omega, min(self.max_omega, gamma))

                self._send_cmd(v_star, gamma)

                all_dist_errors.append(e_lin_log)
                all_ang_errors.append(e_ang_log)
                all_lin_vels.append(v_star)
                all_ang_vels.append(gamma)

                next_goal_msg = PoseStamped()
                next_goal_msg.header.stamp = self.get_clock().now().to_msg()
                next_goal_msg.header.frame_id = self.world_frame
                next_goal_msg.pose = ordered_goals[goal_idx].pose

                feedback_msg.current_goal_index    = goal_idx
                feedback_msg.distance_error        = e_lin_log
                feedback_msg.angular_error         = e_ang_log
                feedback_msg.linear_velocity       = v_star
                feedback_msg.angular_velocity      = gamma
                feedback_msg.distance_to_next_goal = dist_to_goal
                feedback_msg.next_goal_pose = next_goal_msg
                
                goal_handle.publish_feedback(feedback_msg)

                # ── CONDIZIONE DI STOP: pursuit point fermo + robot vicino al goal ──
                if pp.done and dist_to_goal < self.goal_radius:
                    self.get_logger().info(
                        f'  Goal {goal_idx+1} reached (dist={dist_to_goal:.3f} m)'
                    )
                    break

                rate.sleep()

            # ── stop + pausa ──────────────────────────────────────────────
            self._stop_robot()
            t_reached = self.get_clock().now().nanoseconds * 1e-9 - start_time
            goal_reach_times.append(t_reached)
            self.get_logger().info(f'  Pausing {self.pause_sec}s...')
            time.sleep(self.pause_sec)

        # ══════════════════════════════════════
        #  RESULT
        # ══════════════════════════════════════
        result = FollowSequencePlan.Result()
        result.distance_errors    = all_dist_errors
        result.angular_errors     = all_ang_errors
        result.linear_velocities  = all_lin_vels
        result.angular_velocities = all_ang_vels
        result.goal_reach_times   = goal_reach_times

        goal_handle.succeed()
        self._stop_robot()
        self.get_logger().info('Sequential plan completed.')
        return result

    # ── HELPERS ───────────────────────────────

    def _send_cmd(self, v: float, omega: float):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(omega)
        self.cmd_pub.publish(msg)

    def _stop_robot(self):
        self._send_cmd(0.0, 0.0)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = FollowSequencePlanServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()