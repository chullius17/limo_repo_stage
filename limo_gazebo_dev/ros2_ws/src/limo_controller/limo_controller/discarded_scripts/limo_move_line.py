import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from std_msgs.msg import Float64MultiArray

def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

class PixelLineFollowerController(Node):
    def __init__(self):
        super().__init__('pixel_line_follower_controller')

        self.declare_parameter('kd', 0.002)
        self.declare_parameter('kh', 0.1)
        self.declare_parameter('period', 0.1)

        # Parameters for velocity scaling
        self.declare_parameter('max_linear_vel', 0.3)
        self.declare_parameter('min_linear_vel', 0.1)
        self.declare_parameter('max_sat_score', 0.52)   # pi/6 and 0  px
        self.declare_parameter('min_sat_score', 6.03)    # pi/2 and 40 px
        self.declare_parameter('score_distance_factor', 160.0) 

        self.declare_parameter('enable_damping', True)
        self.declare_parameter('alpha_translation', 0.2)
        self.declare_parameter('alpha_rotation', 0.15)

        self.declare_parameter('enable_control', True)

        self.max_linear_vel      = float(self.get_parameter('max_linear_vel').value)
        self.min_linear_vel      = float(self.get_parameter('min_linear_vel').value)
        self.max_angular_vel       = 2 * self.min_linear_vel
        self.kd                  = float(self.get_parameter('kd').value)
        self.kh                  = float(self.get_parameter('kh').value)
        self.period              = float(self.get_parameter('period').value)
        self.enable_damping      = bool(self.get_parameter('enable_damping').value)
        self.alpha_trans         = float(self.get_parameter('alpha_translation').value)
        self.alpha_rot           = float(self.get_parameter('alpha_rotation').value)
        self.enable_control      = bool(self.get_parameter('enable_control').value)
        self.max_sat_score        = float(self.get_parameter('max_sat_score').value)
        self.min_sat_score        = float(self.get_parameter('min_sat_score').value)
        self.score_distance_factor = float(self.get_parameter('score_distance_factor').value)

        self.bottom_margin_px = int()
        self.roi_height_frac = float()
        self.roi_width_frac = float()
        self.center_width_frac = float()
        self.trap_height_frac = float()
        self.roi_bottom_width_frac = float()
        self.roi_received = False

        # Raw / filtered goal state
        self.raw_goal_x      = 0.0
        self.raw_goal_y      = 0.0
        self.raw_theta_star  = 0.0
        self.goal_x          = 0.0
        self.goal_y          = 0.0
        self.theta_star      = 0.0
        self.goal_received   = False
        self.first_goal      = True

        self.latest_d             = 0.0
        self.latest_heading_error = 0.0
        self.gain_constant        = 0.05
        self.max_accel            = 0.1
        self.max_dv_per_cycle = self.max_accel * self.period / 10  # m/s consentiti ogni 0.1s

        self.linear_vel = 0.0

        self.bridge = CvBridge()

        self.roi_sub = self.create_subscription( 
            Float64MultiArray, '/limo/reference/roi_parameters', self.roi_callback, 10)
        self.goals_sub = self.create_subscription(
            PoseArray, '/limo/reference/reference_goals', self.goals_callback, 10)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/limo/costmap/costmap_grid', self.costmap_callback, 10)
        self.cmd_pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Image, '/limo/reference/line_tracking_debug', 10)

        self.timer = self.create_timer(self.period, self.control_loop)
        self.get_logger().info('PixelLineFollowerController avviato con terna traslata.')

    # ──────────────────────────────────────────────────────────────────────────
    def roi_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 6:
            self.bottom_margin_px = int(msg.data[0])
            self.roi_height_frac = float(msg.data[1])
            self.roi_width_frac = float(msg.data[2])
            self.center_width_frac = float(msg.data[3])
            self.trap_height_frac = float(msg.data[4])
            self.roi_bottom_width_frac = float(msg.data[5])
            self.roi_received = True
        else:
            self.get_logger().warn('ROI params received with insufficient length.')
    
    # ──────────────────────────────────────────────────────────────────────────
    def goals_callback(self, msg: PoseArray) -> None:
        if len(msg.poses) < 2:
            self.get_logger().warn('Fewer than 2 goals received.')
            return
# ──────────────────────────────────────────────────────────────────────────
        # Target 1 matches the designated secondary lookahead goal
        target_goal = msg.poses[1]
        self.raw_goal_x     = float(target_goal.position.x)
        self.raw_goal_y     = float(target_goal.position.y)
        qz = target_goal.orientation.z
        qw = target_goal.orientation.w
        self.raw_theta_star = float(2.0 * math.atan2(qz, qw))

        # Reload dynamic params
        self.enable_damping = bool(self.get_parameter('enable_damping').value)
        self.alpha_trans    = float(self.get_parameter('alpha_translation').value)
        self.alpha_rot      = float(self.get_parameter('alpha_rotation').value)

        if not self.enable_damping or self.first_goal:
            self.goal_x     = self.raw_goal_x
            self.goal_y     = self.raw_goal_y
            self.theta_star = self.raw_theta_star
            self.first_goal = False
        else:
            self.goal_x = self.alpha_trans * self.raw_goal_x + (1.0 - self.alpha_trans) * self.goal_x
            self.goal_y = self.alpha_trans * self.raw_goal_y + (1.0 - self.alpha_trans) * self.goal_y

            raw_cos   = math.cos(self.raw_theta_star)
            raw_sin   = math.sin(self.raw_theta_star)
            curr_cos  = math.cos(self.theta_star)
            curr_sin  = math.sin(self.theta_star)
            smooth_cos = self.alpha_rot * raw_cos + (1.0 - self.alpha_rot) * curr_cos
            smooth_sin = self.alpha_rot * raw_sin + (1.0 - self.alpha_rot) * curr_sin
            self.theta_star = math.atan2(smooth_sin, smooth_cos)

        self.goal_received = True

    # ──────────────────────────────────────────────────────────────────────────
    def costmap_callback(self, msg: OccupancyGrid) -> None:
        if not self.roi_received:
            self.get_logger().warn('Waiting for ROI parameters from topic...', throttle_duration_sec=2.0)
            return
        if self.goal_received:
            self._publish_debug_image(msg)

    # ──────────────────────────────────────────────────────────────────────────
    def control_loop(self) -> None:
        if not self.goal_received:
            return

        # 1. PRIMA DI TUTTO: Calcola gli errori geometrici con i dati più freschi
        self.latest_d = (self.goal_y * math.cos(self.theta_star)
                         - self.goal_x * math.sin(self.theta_star))
        self.latest_heading_error = _normalize_angle(self.theta_star)
        
        # # 2. ORA calcoli lo score adattivo (Usa i valori appena aggiornati qui sopra!)
        # score = 6.0 * np.abs(self.latest_heading_error) - np.abs(self.latest_d) / self.score_distance_factor
        
        # # 3. Scegli la velocità lineare in base allo score corrente
        # if score > self.min_sat_score:
        #     target_linear_vel = self.min_linear_vel
        # elif score < self.max_sat_score:
        #     target_linear_vel = self.max_linear_vel
        # else:
        #     ratio = (score - self.min_sat_score) / (self.max_sat_score - self.min_sat_score)
        #     target_linear_vel = self.min_linear_vel + ratio * (self.max_linear_vel - self.min_linear_vel)


        # if target_linear_vel > self.linear_vel:
        #     # Stiamo accelerando: limitiamo l'incremento massimo
        #     self.linear_vel = min(target_linear_vel, self.linear_vel + self.max_dv_per_cycle)
        # else:
        #     # Stiamo decelerando: nessuna restrizione, frenata istantanea
        #     self.linear_vel = target_linear_vel

        self.linear_vel = self.min_linear_vel

        # 4. Scala i guadagni del controller
        gain_scale = self.linear_vel / self.gain_constant
        kd = self.kd * gain_scale
        kh = self.kh * gain_scale

        # 5. Calcola le azioni di controllo (alpha_d, alpha_h, raw_gamma)
        alpha_d = kd * self.latest_d
        alpha_h = kh * self.latest_heading_error
        raw_gamma = alpha_d + alpha_h
        
        # Apply saturation
        gamma = max(min(raw_gamma, self.max_angular_vel), -self.max_angular_vel)
        is_saturated = "YES" if abs(raw_gamma) > self.max_angular_vel else "NO"

        # ── TABULAR LOGGER FOR CONTROL METRICS ─────────────────────────────────
        # Building a clean, scannable fixed-width table format for terminals
        log_msg = (
            "\n"
            "+─────────────────────────────────────────────────────────────────────────+\n"
            "|                      LINE TRACKING CONTROL METRICS                      |\n"
            "+────────────────────────┬────────────────────────┬───────────────────────+\n"
            f"| Scaling_score:   {float('nan'):6.1f}  | K_d (Scaled): {kd:8.5f} | Alpha_d (Cross): {alpha_d:6.3f} |\n"
            f"| Error_d:  {self.latest_d:6.1f} px  | K_h (Scaled): {kh:8.5f} | Alpha_h (Head):  {alpha_h:6.3f} |\n"
            "+────────────────────────┴────────────────────────┼───────────────────────+\n"
            f"| Head_Err: {math.degrees(self.latest_heading_error):6.1f} deg | V_linear:     {self.linear_vel:5.2f} m/s  | Raw Gamma:       {raw_gamma:6.3f} |\n"
            f"| Saturation Active: {is_saturated:3s} | Max W_ang:    {self.max_angular_vel:5.2f} rad/s| CMD W_angular:   {gamma:6.3f} |\n"
            "+─────────────────────────────────────────────────┴───────────────────────+"
        )
        self.get_logger().info(log_msg)

        cmd = Twist()
        cmd.linear.x  = float(self.linear_vel)
        cmd.angular.z = float(gamma)

        self.enable_control = bool(self.get_parameter('enable_control').value)
        if self.enable_control:
            self.cmd_pub.publish(cmd)

    # ──────────────────────────────────────────────────────────────────────────
    def _build_center_roi_overlay(self, h: int, w: int):
        """
        Returns a boolean mask (h x w) for the CENTER region of the ROI,
        replicating exactly the same geometry as reference_path_builder.
        """
        roi_h      = int(h * self.roi_height_frac)
        roi_top    = max(0, h - self.bottom_margin_px - roi_h)
        roi_bottom = max(roi_top + 3, h - self.bottom_margin_px - h // 2)
        global_cx  = w // 2
        roi_w      = int(w * self.roi_width_frac)
        roi_left   = max(0, global_cx - roi_w // 2)
        roi_right  = min(w, global_cx + roi_w // 2)

        rH = roi_bottom - roi_top
        rW = roi_right  - roi_left

        # Trapezoid mask (local)
        trap_mask = np.ones((rH, rW), dtype=np.uint8)
        trap_start_row = int(rH * (1.0 - self.trap_height_frac))
        trap_start_row = np.clip(trap_start_row, 0, rH - 1)
        trap_h = rH - trap_start_row
        if trap_h > 1:
            cx_l = rW // 2
            for idx, r in enumerate(range(trap_start_row, rH)):
                t = float(idx) / max(trap_h - 1, 1)
                hw = (rW / 2.0) * (1.0 - t) + (rW * self.roi_bottom_width_frac / 2.0) * t
                trap_mask[r, :max(0, int(cx_l - hw))]  = 0
                trap_mask[r, min(rW, int(cx_l + hw)):] = 0
        trap_bool = trap_mask.astype(bool)

        # Center region mask (local, tapered)
        cx  = rW // 2
        cht = max(2, int(rW * self.center_width_frac / 2))
        chb = 2
        t_arr = np.arange(rH, dtype=np.float32) / max(rH - 1, 1)
        half  = (cht * (1.0 - t_arr) + chb * t_arr).astype(np.int32)
        col_left  = np.clip(cx - half, 0, rW - 1)[:, np.newaxis]
        col_right = np.clip(cx + half, 0, rW - 1)[:, np.newaxis]
        cols      = np.arange(rW, dtype=np.int32)[np.newaxis, :]
        center_local = ((cols >= col_left) & (cols <= col_right)) & trap_bool

        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[roi_top:roi_bottom, roi_left:roi_right] = center_local

        return full_mask, roi_top, roi_bottom, roi_left, roi_right

    # ──────────────────────────────────────────────────────────────────────────
    def _publish_debug_image(self, msg: OccupancyGrid) -> None:
        h, w = msg.info.height, msg.info.width

        raw  = np.array(msg.data, dtype=np.float32).reshape((h, w))
        disp = np.clip(raw, 0, 255).astype(np.uint8)
        bgr  = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        # ── 1. Draw center-ROI overlay (light blue, 50% alpha) ────────────────
        center_mask, roi_top, roi_bottom, roi_left, roi_right = \
            self._build_center_roi_overlay(h, w)

        overlay = bgr.copy()
        overlay[center_mask] = (200, 130, 30)   # BGR center color overlay
        cv2.addWeighted(overlay, 0.50, bgr, 0.50, 0, bgr)

        # Outer ROI rectangle border
        cv2.rectangle(bgr,
                      (roi_left, roi_top),
                      (roi_right - 1, roi_bottom - 1),
                      (200, 200, 200), 1)

        # ── 2. TRANSLATED Robot anchor — aligned to roi_bottom ────────────────
        # Shifting start_r up to roi_bottom to match the frame position of image_e60618.png
        start_r = roi_bottom
        start_c = w // 2

        # ── 3. Goal position in image space ───────────────────────────────────
        goal_r = int(round(start_r - self.goal_x))
        goal_c = int(round(start_c - self.goal_y))

        # ── 4. Reference line through goal ────────────────────────────────────
        line_len = max(h, w)
        ldc = -math.sin(self.theta_star)
        ldr = -math.cos(self.theta_star)
        p1 = (int(goal_c + line_len * ldc), int(goal_r + line_len * ldr))
        p2 = (int(goal_c - line_len * ldc), int(goal_r - line_len * ldr))
        cv2.line(bgr, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)   # Cyan tracking line

        # ── 5. Raw goal marker (magenta cross) ────────────────────────────────
        if self.enable_damping:
            raw_r = int(round(start_r - self.raw_goal_x))
            raw_c = int(round(start_c - self.raw_goal_y))
            if 0 <= raw_r < h and 0 <= raw_c < w:
                cv2.drawMarker(bgr, (raw_c, raw_r),
                               (255, 0, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

        # ── 6. Filtered goal circle + orientation arrow ────────────────────────
        if 0 <= goal_r < h and 0 <= goal_c < w:
            cv2.circle(bgr, (goal_c, goal_r), 5, (0, 255, 255), -1, cv2.LINE_AA)
            arrow_len = 20
            arr_c = int(goal_c + arrow_len * ldc)
            arr_r = int(goal_r + arrow_len * ldr)
            cv2.arrowedLine(bgr, (goal_c, goal_r), (arr_c, arr_r),
                            (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.3)

        # ── 7. Robot dot + reference frame axes at the translated anchor ──────
        cv2.circle(bgr, (start_c, start_r), 5, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.circle(bgr, (start_c, start_r), 2, (255, 255, 255), -1, cv2.LINE_AA)

        axis_len = 35
        # X (forward) → UP in image
        cv2.arrowedLine(bgr,
                        (start_c, start_r),
                        (start_c, start_r - axis_len),
                        (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(bgr, "X (fwd)",
                    (start_c - 18, start_r - axis_len - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        # Y (left) → LEFT in image
        cv2.arrowedLine(bgr,
                        (start_c, start_r),
                        (start_c - axis_len, start_r),
                        (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(bgr, "Y (left)",
                    (start_c - axis_len - 45, start_r + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        # ── 8. Text overlay ────────────────────────────────────────────────────
        cv2.putText(bgr, f"Dist d: {self.latest_d:.1f} px", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"Head Err: {math.degrees(self.latest_heading_error):.1f} deg", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        damp_color = (200, 255, 200) if self.enable_damping else (150, 150, 255)
        cv2.putText(bgr, f"Damping: {'ON' if self.enable_damping else 'OFF'}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, damp_color, 1, cv2.LINE_AA)

        img_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        img_msg.header.stamp    = self.get_clock().now().to_msg()
        img_msg.header.frame_id = ""
        self.debug_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    controller = PixelLineFollowerController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down: Sending stop command to robot...')
        
        # Create a zero-velocity Twist message
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        
        # Publish the stop command immediately
        controller.cmd_pub.publish(stop_cmd)
        
        # Small sleep ensures the network layer actually flushes the packet 
        # to the /cmd_vel topic before the context is destroyed
        import time
        time.sleep(0.1)
        
    finally:
        # Clean up ROS 2 resources properly
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()