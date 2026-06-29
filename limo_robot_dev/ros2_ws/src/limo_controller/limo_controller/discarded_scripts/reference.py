import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from std_msgs.msg import Float64MultiArray

_COLOR_LEFT   = (220, 80,  30)
_COLOR_CENTER = (30,  160, 220)
_COLOR_RIGHT  = (30,  160,  30)
_COLOR_ROBOT  = (0,   0,   220)
_COLOR_GOAL   = (0,   220, 220)


class ReferencePathBuilder(Node):

    def __init__(self):
        super().__init__('reference_path_builder')

        # Pure pixel and image-space parameters
        self.declare_parameter('publish_debug',            True)
        self.declare_parameter('cost_min',                 10.0)
        self.declare_parameter('cost_max',                 100.0)
        self.declare_parameter('max_cost_threshold',       5000.0)
        self.declare_parameter('bottom_margin_px',         50)
        self.declare_parameter('roi_height_frac',          2.0 / 3.0)
        self.declare_parameter('roi_width_frac',           0.30)        
        self.declare_parameter('center_width_frac',        0.60)        
        self.declare_parameter('erode_kernel_size',        5)
        self.declare_parameter('internal_step_px',         1)
        self.declare_parameter('distance_lambda',          1.5)
        self.declare_parameter('flood_fill_threshold',     100.0)
        self.declare_parameter('right_bias_weight',        1.5) 
        
        self.declare_parameter('trapezoid_height_frac',    0.40)  
        self.declare_parameter('roi_bottom_width_frac',    0.30) 

        self.publish_debug         = self.get_parameter('publish_debug').value
        self.cost_min              = self.get_parameter('cost_min').value
        self.cost_max              = self.get_parameter('cost_max').value
        raw_thr                    = self.get_parameter('max_cost_threshold').value
        self.max_cost_threshold    = np.inf if raw_thr < 0 else raw_thr
        self.bottom_margin_px      = self.get_parameter('bottom_margin_px').value
        self.roi_height_frac       = self.get_parameter('roi_height_frac').value
        self.roi_width_frac        = self.get_parameter('roi_width_frac').value
        self.center_width_frac     = self.get_parameter('center_width_frac').value
        self.erode_kernel_size     = self.get_parameter('erode_kernel_size').value
        self.internal_step_px      = self.get_parameter('internal_step_px').value
        self.dist_lambda           = self.get_parameter('distance_lambda').value
        self.flood_fill_threshold  = self.get_parameter('flood_fill_threshold').value
        self.right_bias_weight     = self.get_parameter('right_bias_weight').value
        
        self.trap_height_frac      = self.get_parameter('trapezoid_height_frac').value
        self.roi_bottom_width_frac = self.get_parameter('roi_bottom_width_frac').value

        self.bridge = CvBridge()
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/limo/costmap/costmap_grid', self.costmap_callback, 10)
        
        self.roi_pub = self.create_publisher(Float64MultiArray, '/limo/reference/roi_parameters', 10)
        
        self.goals_pub = self.create_publisher(PoseArray, '/limo/reference/reference_goals', 10)
        self.debug_pub = self.create_publisher(Image, '/limo/reference/roi_and_goals', 10)
        self.flooding_pub = self.create_publisher(Image, '/limo/reference/flooding_debug', 10)

        self.get_logger().info('ReferencePathBuilder avviato — Output in PIXEL relativi allo Start.')

    # ──────────────────────────────────────────────────────────────────────────
    def costmap_callback(self, msg: OccupancyGrid):
        t_start = time.perf_counter()

        roi_msg = Float64MultiArray()
        roi_msg.data = [
            float(self.bottom_margin_px),
            self.roi_height_frac,
            self.roi_width_frac,
            self.center_width_frac,
            self.trap_height_frac,
            self.roi_bottom_width_frac
        ]
        self.roi_pub.publish(roi_msg)

        h, w = msg.info.height, msg.info.width
        raw      = np.array(msg.data, dtype=np.float32).reshape((h, w))
        cost_map = np.where(raw < 0, 0.0, raw)

        roi_h      = int(h * self.roi_height_frac)
        roi_top    = max(0, h - self.bottom_margin_px - roi_h)
        roi_bottom = max(roi_top + 3, h - self.bottom_margin_px - h // 2)

        global_cx  = w // 2
        roi_w      = int(w * self.roi_width_frac)
        roi_left   = max(0, global_cx - roi_w // 2)
        roi_right  = min(w, global_cx + roi_w // 2)

        if roi_bottom <= roi_top + 2 or roi_right <= roi_left + 2:
            self.get_logger().warn('ROI troppo piccola, frame saltato.')
            return

        roi_cost = cost_map[roi_top:roi_bottom, roi_left:roi_right].copy()
        rH, rW   = roi_cost.shape
        start_local = (rH - 2, rW // 2)

        t_roi = time.perf_counter()

        trapezoid_mask = self._build_hybrid_mask(rH, rW)
        connected_mask = self._flood_fill_connected(roi_cost, start_local, trapezoid_mask)

        t_flood = time.perf_counter()

        stamp = self.get_clock().now().to_msg()

        if self.publish_debug:
            self._publish_flooding_image(cost_map, roi_cost, connected_mask,
                                         roi_top, roi_bottom, roi_left, roi_right, stamp)

        cx  = rW // 2
        cht = max(2, int(rW * self.center_width_frac / 2))
        chb = 2
        regions = self._build_region_masks(rH, rW, cx, cht, chb)

        for name in regions.keys():
            regions[name] = regions[name] & trapezoid_mask

        t_regions = time.perf_counter()

        goals_local = []
        for name, mask in regions.items():
            g = self._best_goal_in_region(roi_cost, mask, connected_mask, start_local, name)
            if g is not None:
                r, c = g
                yaw = self._goal_orientation_extended(cost_map, connected_mask, r, c, roi_top, roi_left)
                goals_local.append((r, c, yaw))

        t_goals = time.perf_counter()

        goals_global = [(r + roi_top, c + roi_left, yaw) for r, c, yaw in goals_local]
        start_global = (start_local[0] + roi_top, start_local[1] + roi_left)

        # ── Pure Pixel-Space PoseArray Publication (FIXED COORDINATES) ────────
        stamp = self.get_clock().now().to_msg()
        pose_array = PoseArray()
        pose_array.header.stamp    = stamp
        pose_array.header.frame_id = "" 

        r_start, c_start = start_global

        goals_ref = []

        self.get_logger().info(f"Start (pixel): row={r_start}, col={c_start} | Goals (pixel): {[(r, c) for r, c, _ in goals_global]}")

        for (r, c, yaw) in goals_global:
            pose = Pose()
            
            # X-forward: positive when the target row is above the robot row (r < r_start)
            pose.position.x = float(r_start - r)  
            
            # Y-left: positive when the target column is to the left of the robot column (c < c_start)
            pose.position.y = float(c_start - c)  
            pose.position.z = 0.0

            goals_ref.append((pose.position.x, pose.position.y, yaw))
            
            # Rotation representation (0 rad is aligned with the X-forward axis)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.orientation.w = float(np.cos(yaw / 2.0))
            pose_array.poses.append(pose)

        self.goals_pub.publish(pose_array)

        if self.publish_debug:
            self._publish_debug_image(
                cost_map, roi_top, roi_bottom, roi_left, roi_right,
                regions, cx, cht, chb, goals_global, start_global, stamp, trapezoid_mask)

        t_end = time.perf_counter()
        self.get_logger().info(
            f"[PROFILE] Total: {(t_end-t_start)*1000:.1f}ms | Goals Array: {goals_ref}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _build_hybrid_mask(self, H: int, W: int) -> np.ndarray:
        mask = np.ones((H, W), dtype=np.uint8)
        trap_start_row = int(H * (1.0 - self.trap_height_frac))
        trap_start_row = np.clip(trap_start_row, 0, H - 1)
        
        trap_h = H - trap_start_row
        if trap_h <= 1:
            return mask.astype(bool)

        cx = W // 2
        for idx, r in enumerate(range(trap_start_row, H)):
            t = float(idx) / max(trap_h - 1, 1)
            current_half_w = (W / 2.0) * (1.0 - t) + ((W * self.roi_bottom_width_frac) / 2.0) * t
            
            c_left = max(0, int(cx - current_half_w))
            c_right = min(W, int(cx + current_half_w))
            
            mask[r, :c_left] = 0
            mask[r, c_right:] = 0
            
        return mask.astype(bool)

    # ──────────────────────────────────────────────────────────────────────────
    def _flood_fill_connected(self, roi_cost: np.ndarray, start_local: tuple, trapezoid_mask: np.ndarray) -> np.ndarray:
        rH, rW = roi_cost.shape
        passable_mask = (roi_cost < self.flood_fill_threshold) & trapezoid_mask
        passable = np.where(passable_mask, np.uint8(255), np.uint8(0))
        
        flood_mask = np.zeros((rH + 2, rW + 2), dtype=np.uint8)
        seed = (int(start_local[1]), int(start_local[0]))

        if passable[start_local[0], start_local[1]] == 0:
            sr, sc = start_local
            found  = False
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < rH and 0 <= nc < rW and passable[nr, nc] == 255:
                        seed  = (nc, nr)
                        found = True
                        break
                if found:
                    break
            if not found:
                return np.zeros((rH, rW), dtype=bool)

        cv2.floodFill(passable, flood_mask, seed, newVal=128, flags=8 | cv2.FLOODFILL_FIXED_RANGE)
        return passable == 128

    # ──────────────────────────────────────────────────────────────────────────
    def _goal_orientation_extended(self, cost_map: np.ndarray,
                                   connected_mask_local: np.ndarray,
                                   r_local: int, c_local: int,
                                   roi_top: int, roi_left: int) -> float:
        gH, gW = cost_map.shape
        rH, rW = connected_mask_local.shape
        
        r_global = r_local + roi_top
        c_global = c_local + roi_left
        R = 12  

        r0 = max(0, r_global - R);  r1 = min(gH, r_global + R + 1)
        c0 = max(0, c_global - R);  c1 = min(gW, c_global + R + 1)

        patch_cost = cost_map[r0:r1, c0:c1]
        r_grid, c_grid = np.mgrid[r0:r1, c0:c1]
        
        under_threshold = patch_cost < self.flood_fill_threshold
        
        in_roi_r = (r_grid >= roi_top) & (r_grid < roi_top + rH)
        in_roi_c = (c_grid >= roi_left) & (c_grid < roi_left + rW)
        in_roi = in_roi_r & in_roi_c
        
        r_loc_mapped = r_grid - roi_top
        c_loc_mapped = c_grid - roi_left
        
        patch_conn = np.zeros_like(patch_cost, dtype=bool)
        
        if np.any(in_roi):
            patch_conn[in_roi] = connected_mask_local[r_loc_mapped[in_roi], c_loc_mapped[in_roi]]
            
        patch_conn[~in_roi] = True

        local_valid = patch_conn & under_threshold
        rs_loc, cs_loc = np.where(local_valid)

        if len(rs_loc) < 3:
            sr_global = (rH - 2) + roi_top
            sc_global = (rW // 2) + roi_left
            dy = float(sr_global - r_global)
            dx = float(sc_global - c_global)
            return float(np.arctan2(-dy, dx))

        pts = np.stack([cs_loc.astype(np.float32), rs_loc.astype(np.float32)], axis=1)
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = (centered.T @ centered) / max(len(pts) - 1, 1)

        eigvals, eigvecs = np.linalg.eigh(cov)
        main_axis = eigvecs[:, np.argmax(eigvals)]

        dx_img = float(main_axis[0]) # OpenCV columns axis (+right)
        dy_img = float(main_axis[1]) # OpenCV rows axis (+down)

        # Convert the PCA vector to our Robot Frame (X: forward, Y: left)
        # Moving up in image (decreasing rows) -> +X forward
        # Moving left in image (decreasing columns) -> +Y left
        robot_dx = -dy_img
        robot_dy = -dx_img
        yaw = np.arctan2(robot_dy, robot_dx)

        # Base Reference: Robot Start Position
        sr_global = (rH - 2) + roi_top
        sc_global = (rW // 2) + roi_left
        
        # Calculate the direction from the robot START to the GOAL in the robot frame
        # (This is the direction the robot must look to move towards the goal)
        goal_direction_x = float(sr_global - r_global) # positive if goal is ahead
        goal_direction_y = float(sc_global - c_global) # positive if goal is to the left
        
        # Dot product to check if the PCA axis is aligned with the path progression
        dot = robot_dx * goal_direction_x + robot_dy * goal_direction_y
        
        # If the dot product is negative, the PCA vector is pointing backwards (towards the robot)
        # We flip it by 180 degrees to force it to point forward along the corridor
        if dot < 0:
            yaw = yaw + np.pi if yaw < 0 else yaw - np.pi

        return yaw

    # ──────────────────────────────────────────────────────────────────────────
    def _build_region_masks(self, H, W, cx, cht, chb):
        t    = np.arange(H, dtype=np.float32) / max(H - 1, 1)
        half = (cht * (1.0 - t) + chb * t).astype(np.int32)

        col_left  = np.clip(cx - half, 0, W - 1)[:, np.newaxis]
        col_right = np.clip(cx + half, 0, W - 1)[:, np.newaxis]
        cols      = np.arange(W, dtype=np.int32)[np.newaxis, :]

        mask_center = (cols >= col_left) & (cols <= col_right)
        mask_left   = cols < col_left
        mask_right  = cols > col_right

        return {'left': mask_left, 'center': mask_center, 'right': mask_right}

    # ──────────────────────────────────────────────────────────────────────────
    def _best_goal_in_region(self, roi_cost, region_mask, connected_mask,
                             start_local, region_name):
        cmin, cmax = self.cost_min, self.cost_max
        sr, sc     = start_local
        step       = self.internal_step_px

        valid_full = region_mask & connected_mask & (roi_cost >= cmin) & (roi_cost <= cmax)

        valid_downsampled = np.zeros_like(valid_full)
        valid_downsampled[::step, ::step] = valid_full[::step, ::step]

        rs, cs = np.where(valid_downsampled)

        if len(rs) == 0:
            cols_top = np.where(region_mask[0, :])[0]
            if len(cols_top) == 0:
                return None
            return (0, int(cols_top[len(cols_top) // 2]))

        score = (rs - sr) ** 2 + (cs - sc) ** 2

        if region_name == 'center':
            right_distance = cs - sc
            score = score + (self.right_bias_weight * (right_distance ** 2) * np.sign(right_distance))

        best = np.argmax(score)
        return (int(rs[best]), int(cs[best]))

    # ──────────────────────────────────────────────────────────────────────────
    def _publish_debug_image(self, cost_map,
                             roi_top, roi_bottom, roi_left, roi_right,
                             regions_local, cx, cht, chb, goals, start, stamp, trapezoid_mask=None):
        disp    = np.clip(cost_map, 0, 255).astype(np.uint8)
        bgr     = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        overlay = bgr.copy()

        colors_region = {'left': _COLOR_LEFT, 'center': _COLOR_CENTER, 'right': _COLOR_RIGHT}
        for name, mask in regions_local.items():
            full = np.zeros(cost_map.shape, dtype=bool)
            full[roi_top:roi_bottom, roi_left:roi_right] = mask
            overlay[full] = colors_region[name]
        cv2.addWeighted(overlay, 0.30, bgr, 0.70, 0, bgr)

        if trapezoid_mask is not None:
            res = cv2.findContours(trapezoid_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = res[0] if len(res) == 2 else res[1]
            for c in contours:
                c[:, :, 0] += roi_left
                c[:, :, 1] += roi_top
                cv2.drawContours(bgr, [c], -1, (100, 255, 100), 1, cv2.LINE_AA)

        cv2.line(bgr, (cx - cht + roi_left, roi_top),
                      (cx - chb + roi_left, roi_bottom - 1), (255,255,255), 1, cv2.LINE_AA)
        cv2.line(bgr, (cx + cht + roi_left, roi_top),
                      (cx + chb + roi_left, roi_bottom - 1), (255,255,255), 1, cv2.LINE_AA)
        cv2.rectangle(bgr, (roi_left, roi_top),
                           (roi_right - 1, roi_bottom - 1), (200,200,200), 1)

        mid_row = roi_top + (roi_bottom - roi_top) // 2
        cv2.putText(bgr, 'L', (roi_left + 6,       mid_row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_LEFT,   1, cv2.LINE_AA)
        cv2.putText(bgr, 'C', (cx + roi_left - 6,  roi_top + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_CENTER, 1, cv2.LINE_AA)
        cv2.putText(bgr, 'R', (roi_right - 18,     mid_row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_RIGHT,  1, cv2.LINE_AA)

        # ── DRAW CUSTOM START REFERENCE FRAME (DEBUG) ─────────────────────────
        # X Axis = Forward (Upwards in image space -> decreasing rows) -> RED
        # Y Axis = Left (Leftwards in image space -> decreasing columns) -> GREEN
        axis_len = 30
        start_c, start_r = start[1], start[0]
        
        # X-Axis (Forward) points UP
        cv2.arrowedLine(bgr, (start_c, start_r), (start_c, start_r - axis_len), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(bgr, 'X(fwd)', (start_c - 15, start_r - axis_len - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Y-Axis (Left) points LEFT
        cv2.arrowedLine(bgr, (start_c, start_r), (start_c - axis_len, start_r), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(bgr, 'Y(left)', (start_c - axis_len - 35, start_r + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        # ──────────────────────────────────────────────────────────────────────

        # Draw Goal Markers and Orientations (correctly handling the image space transformation)
        arrow_len = 18
        for (r, c, yaw) in goals:
            cv2.circle(bgr, (c, r), 5, _COLOR_GOAL, 2)
            cv2.drawMarker(bgr, (c, r), _COLOR_GOAL, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
            
            # Since yaw is 0 forward (upward), we shift by pi/2 to draw correctly in standard image cos/sin space
            draw_angle = yaw + np.pi / 2.0
            img_dx = int(round( np.cos(draw_angle) * arrow_len))
            img_dy = int(round(-np.sin(draw_angle) * arrow_len))
            cv2.arrowedLine(bgr, (c, r), (c + img_dx, r + img_dy), _COLOR_GOAL, 2, cv2.LINE_AA, tipLength=0.35)

        cv2.circle(bgr, (start_c, start_r), 6, _COLOR_ROBOT, -1)
        msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        msg.header.stamp = stamp
        self.debug_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────────────
    def _publish_flooding_image(self, full_cost_map: np.ndarray, roi_cost: np.ndarray,
                                connected_mask: np.ndarray,
                                roi_top: int, roi_bottom: int, roi_left: int, roi_right: int,
                                stamp):
        flood_full = np.full(full_cost_map.shape, 255, dtype=np.uint8)
        accepted = (connected_mask & (roi_cost < self.flood_fill_threshold))

        rH = roi_bottom - roi_top
        rW = roi_right - roi_left
        if accepted.shape != (rH, rW):
            return

        flood_full[roi_top:roi_bottom, roi_left:roi_right][accepted] = 0

        bgr = cv2.cvtColor(flood_full, cv2.COLOR_GRAY2BGR)
        msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = "" 
        self.flooding_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReferencePathBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()