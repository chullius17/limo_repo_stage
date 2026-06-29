import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from .fmm import fmm_multi_goal


class ReferencePathBuilder(Node):
    """
    ROS2 Node:
      1. Receives OccupancyGrid (/limo/costmap)
      2. Maintains OpenCV coordinates consistent with the costmap
      3. ROI: lower 2/3 of the map (minus bottom_margin_px), central 1/3 in width
      4. Local minima on the 3 borders of the ROI + internal points weighted by distance
      5. Selects a maximum number of 5 goals (the most convenient by total cost)
      6. Start: second-to-last row of the ROI, central column
      7. FMM on the ROI → backtrack → publishes Path + debug images
    """

    def __init__(self):
        super().__init__('reference_path_builder')

        self.declare_parameter('map_frame',          'map')
        self.declare_parameter('resolution',         0.05)
        self.declare_parameter('publish_debug',      True)
        self.declare_parameter('cost_min',           5.0)
        self.declare_parameter('cost_max',           50.0)
        self.declare_parameter('max_cost_threshold', 5000.0)
        self.declare_parameter('bottom_margin_px',   50)
        self.declare_parameter('roi_height_frac',    2.0 / 3.0)
        self.declare_parameter('roi_width_frac',     1.0 / 5.0)
        self.declare_parameter('erode_kernel_size',  5)
        self.declare_parameter('min_goal_distance',  12)
        self.declare_parameter('max_goals_count',    5)
        self.declare_parameter('distance_lambda',    1.5)
        
        # --- New parameters for internal goals weights ------------------------
        # Total cost = Map cost + (distance_weight * (1.0 / distance_from_start))
        self.declare_parameter('internal_distance_weight', 500.0) 
        # Downsampling factor to avoid evaluating every single internal pixel (speed-up)
        self.declare_parameter('internal_step_px', 4) 

        self.map_frame               = self.get_parameter('map_frame').value
        self.resolution              = self.get_parameter('resolution').value
        self.publish_debug           = self.get_parameter('publish_debug').value
        self.cost_min                = self.get_parameter('cost_min').value
        self.cost_max                = self.get_parameter('cost_max').value
        raw_thr                      = self.get_parameter('max_cost_threshold').value
        self.max_cost_threshold      = np.inf if raw_thr < 0 else raw_thr
        self.bottom_margin_px        = self.get_parameter('bottom_margin_px').value
        self.roi_height_frac         = self.get_parameter('roi_height_frac').value
        self.roi_width_frac          = self.get_parameter('roi_width_frac').value
        self.erode_kernel_size       = self.get_parameter('erode_kernel_size').value
        self.min_goal_distance       = self.get_parameter('min_goal_distance').value
        self.max_goals_count         = self.get_parameter('max_goals_count').value
        self.internal_dist_weight    = self.get_parameter('internal_distance_weight').value
        self.internal_step_px        = self.get_parameter('internal_step_px').value
        self.dist_lambda             = self.get_parameter('distance_lambda').value

        self.bridge = CvBridge()

        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/limo/costmap', self.costmap_callback, 10)

        self.path_pub  = self.create_publisher(Path,  '/limo/reference_path',       10)
        self.final_pub = self.create_publisher(Image, '/limo/reference_path_final', 10)
        self.debug_pub = self.create_publisher(Image, '/limo/admissible_roi_debug', 10)

        self.get_logger().info('ReferencePathBuilder avviato con supporto goal interni e fallback.')

    def costmap_callback(self, msg: OccupancyGrid):
        h = msg.info.height
        w = msg.info.width

        raw = np.array(msg.data, dtype=np.float32).reshape((h, w))
        cost_map = np.where(raw < 0, 0.0, raw)

        # --- ROI bounds ------------------------------------------------------
        roi_h      = int(h * self.roi_height_frac)
        roi_top    = max(0, h - self.bottom_margin_px - roi_h)
        roi_bottom = min(h, h - self.bottom_margin_px - h // 2)
        half_w     = int(w * self.roi_width_frac // 2)
        roi_left   = max(0, w // 2 - half_w)
        roi_right  = min(w, w // 2 + half_w)

        if roi_bottom <= roi_top + 2 or roi_right <= roi_left + 2:
            self.get_logger().warn('ROI troppo piccola, frame saltato.')
            return

        roi_cost     = cost_map[roi_top:roi_bottom, roi_left:roi_right].copy()
        roi_h_actual = roi_cost.shape[0]
        roi_w_actual = roi_cost.shape[1]

        # --- Start locale: penultima riga, colonna centrale ----------------───
        start_local = (roi_h_actual - 2, roi_w_actual // 2)

        # --- Goals Extraction (Borders + Internals + Fallback) ----------------
        goals_local = self._extract_all_goals(roi_cost, start_local)

        self.get_logger().info(f'Goal selezionati (max {self.max_goals_count}): {len(goals_local)}')

        # --- FMM ─────────────────────────────────────────────────────────────
        valid_paths_local = fmm_multi_goal(
            roi_cost, start_local, goals_local,
            max_cost_threshold=self.max_cost_threshold)

        # Remap local coordinates → global coordinates
        valid_paths = {
            (r + roi_top, c + roi_left): [(pr + roi_top, pc + roi_left)
                                           for pr, pc in pts]
            for (r, c), pts in valid_paths_local.items()
        }

        start_global = (start_local[0] + roi_top, start_local[1] + roi_left)

        self.get_logger().info(f'Percorsi validi: {len(valid_paths)} / {len(goals_local)}')

        # --- Publish ROS Path ------------------------------------------------
        stamp = self.get_clock().now().to_msg()

        for goal, path_points in valid_paths.items():
            ros_path = Path()
            ros_path.header.stamp    = stamp
            ros_path.header.frame_id = self.map_frame

            for (r, c) in path_points:
                ps = PoseStamped()
                ps.header.stamp    = stamp
                ps.header.frame_id = self.map_frame
                ps.pose.position.x = c * self.resolution + msg.info.origin.position.x
                r_ros = h - 1 - r
                ps.pose.position.y = r_ros * self.resolution + msg.info.origin.position.y
                ps.pose.orientation.w = 1.0
                ros_path.poses.append(ps)

            self.path_pub.publish(ros_path)

        # --- Visualizations ------------------------------------------------──
        goals_global = [(r + roi_top, c + roi_left) for r, c in goals_local]
        self._publish_final_image(cost_map, valid_paths, start_global, stamp)
        if self.publish_debug:
            self._publish_debug_image(
                cost_map, roi_top, roi_bottom, roi_left, roi_right,
                goals_global, valid_paths, start_global, stamp)

    def _extract_all_goals(self, roi_cost: np.ndarray, start_local: tuple) -> list:
        """
        Extracts border minima and valid internal points, sorts them by penalized cost,
        and applies a greedy distance clustering. Falls back to top-center if empty.
        """
        ks = self.erode_kernel_size
        cmin = self.cost_min
        cmax = self.cost_max
        candidates = []  # List of tuples: (row, col, evaluated_cost)

        # 1. BORDER MINIMA (As original code)
        # Top border
        border_top = roi_cost[0, :]
        u8_top = np.clip(border_top, 0, 255).astype(np.uint8)
        ker_h = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, 1))
        eroded_top = cv2.erode(u8_top.reshape(1, -1), ker_h).flatten()
        mask_top = (eroded_top == u8_top) & (border_top >= cmin) & (border_top <= cmax)
        for col in np.where(mask_top)[0]:
            candidates.append((0, int(col), float(border_top[col])))

        # Left border
        border_left = roi_cost[:, 0]
        u8_left = np.clip(border_left, 0, 255).astype(np.uint8)
        ker_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ks))
        eroded_left = cv2.erode(u8_left.reshape(-1, 1), ker_v).flatten()
        mask_left = (eroded_left == u8_left) & (border_left >= cmin) & (border_left <= cmax)
        for row in np.where(mask_left)[0]:
            candidates.append((int(row), 0, float(border_left[row])))

        # Right border
        last_col = roi_cost.shape[1] - 1
        border_right = roi_cost[:, -1]
        u8_right = np.clip(border_right, 0, 255).astype(np.uint8)
        eroded_right = cv2.erode(u8_right.reshape(-1, 1), ker_v).flatten()
        mask_right = (eroded_right == u8_right) & (border_right >= cmin) & (border_right <= cmax)
        for row in np.where(mask_right)[0]:
            candidates.append((int(row), last_col, float(border_right[row])))

        # 2. INTERNAL GOALS SECTION WITH DISTANCE PENALTY
        # Filter matching elements inside the core ROI matrix using slicing steps to save CPU
        internal_mask = (roi_cost >= cmin) & (roi_cost <= cmax)
        
        # Exclude borders from the internal search
        internal_mask[0, :]  = False
        internal_mask[:, 0]  = False
        internal_mask[:, -1] = False

        internal_rows, internal_cols = np.where(internal_mask)
        
        # Process every Nth pixel (internal_step_px) to maintain real-time performance
        for i in range(0, len(internal_rows), self.internal_step_px):
            r = int(internal_rows[i])
            c = int(internal_cols[i])
            
            # Euclidean distance calculation from start_local (origin)
            dist = np.hypot(r - start_local[0], c - start_local[start_local.__len__()-1])
            
            if dist > 2.0:
                # Penalty inversely proportional to distance: closer points get a massive cost penalty
                distance_penalty = self.dist_lambda * self.internal_dist_weight / dist
                total_cost = float(roi_cost[r, c]) + distance_penalty
                candidates.append((r, c, total_cost))

        # 3. FALLBACK: "Go Straight" policy if absolutely no goals are found
        if not candidates:
            self.get_logger().warn('Nessun goal rilevato. Attivazione Fallback: Centro Superiore ROI (Vai dritto).')
            # Target is at row 0, middle column of the ROI
            return [(0, roi_cost.shape[1] // 2)]

        # 4. GREEDY CLUSTERING & LIMIT CUT
        # Sort all combined candidates by cost ascending (lower penalized cost = higher priority)
        candidates.sort(key=lambda x: x[2])
        d_min = self.min_goal_distance
        chosen = []

        for r, c, _ in candidates:
            if not any(abs(r - gr) + abs(c - gc) < d_min for gr, gc in chosen):
                chosen.append((r, c))
                if len(chosen) >= self.max_goals_count:
                    break

        return chosen

    def _publish_final_image(self, cost_map, valid_paths, start, stamp):
        disp = np.clip(cost_map, 0, 255).astype(np.uint8)
        bgr  = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0),
                  (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for idx, (goal, pts) in enumerate(valid_paths.items()):
            col = colors[idx % len(colors)]
            for (r, c) in pts:
                if 0 <= r < bgr.shape[0] and 0 <= c < bgr.shape[1]:
                    bgr[r, c] = col
            cv2.circle(bgr, (goal[1], goal[0]), 4, col, -1)

        cv2.circle(bgr, (start[1], start[0]), 6, (0, 0, 255), -1)

        img_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        img_msg.header.stamp = stamp
        self.final_pub.publish(img_msg)

    def _publish_debug_image(self, cost_map,
                             roi_top, roi_bottom, roi_left, roi_right,
                             goals, valid_paths, start, stamp):
        disp = np.clip(cost_map, 0, 255).astype(np.uint8)
        bgr  = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(bgr, (roi_left, roi_top), (roi_right, roi_bottom),
                      (255, 100, 0), 1)

        for (r, c) in goals:
            cv2.circle(bgr, (c, r), 3, (0, 255, 255), -1)

        for goal in valid_paths.keys():
            cv2.circle(bgr, (goal[1], goal[0]), 5, (0, 255, 0), 2)

        cv2.circle(bgr, (start[1], start[0]), 6, (0, 0, 255), -1)

        img_msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        img_msg.header.stamp = stamp
        self.debug_pub.publish(img_msg)


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