import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Header
from typing import Tuple
from geometry_msgs.msg import TransformStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as TimeMsg

class Costmap(Node):
    """
    Converts a specific color channel of a BEV image into a dedicated ROS2 OccupancyGrid costmap.
    Supported parameters: 'MAGENTA', 'RED', 'GREEN'.
    """

    # --- exact BGR colours ---------------------------------------------------
    COLOR_MAP = {
        'MAGENTA': np.array([255,   0, 255], dtype=np.uint8),
        'RED':     np.array([  0,   0, 255], dtype=np.uint8),
        'GREEN':   np.array([  0, 255,   0], dtype=np.uint8)
    }

    TOLERANCE = 30          # pixel-value tolerance for colour matching

    # --- Configuration profiles per color ------------------------------------
    CONFIG_MAP = {
        'MAGENTA': {'peak_cost': 100.0, 'radius': 2},
        'RED':     {'peak_cost': 60.0,  'radius': 5},
        'GREEN':   {'peak_cost': 30.0,  'radius': 5}
    }

    # --- decay steepness (higher = faster fall-off) --------------------------
    DECAY = 6.0

    # --- ROI: fraction of image height to KEEP (measured from the bottom) ----
    ROI_FRACTION = 0.8
    RULER_FRACTION = 0.58

    def __init__(self):
        super().__init__('costmap_node')
        self.bridge = CvBridge()

        # Dynamic color configuration parameter
        self.declare_parameter('color', 'MAGENTA')
        self.declare_parameter('fixed_frame', 'base_footprint')
        self.declare_parameter('resolution', 0.0092)
        self.declare_parameter('publish_debug', True)

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', 
                             rclpy.Parameter.Type.BOOL, True)])

        self.fixed_frame     = self.get_parameter('fixed_frame').value
        self.resolution    = self.get_parameter('resolution').value
        self.publish_debug = self.get_parameter('publish_debug').value
        
        # Sanitize and read input color string
        self.color_flag = self.get_parameter('color').value.upper()

        if self.color_flag not in self.COLOR_MAP:
            self.get_logger().error(f"Color '{self.color_flag}' not supported! Defaulting to MAGENTA.")
            self.color_flag = 'MAGENTA'

        # TF2 Listener setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # TF2 Broadcaster to publish the new frame visualizer
        self.tf_broadcaster = TransformBroadcaster(self)

        self.tf_timer = self.create_timer(0.1, self.tf_callback)

        self.bev_sub = self.create_subscription(
            Image,
            '/limo/color/image_raw_bird_perspective',
            self.bev_callback,
            10
        )
        
        # Dynamic topic output naming based on selection
        self.topic_suffix = self.color_flag.lower()
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, 
            f'/limo/costmap/costmap_grid_{self.topic_suffix}', 
            10
        )

        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, f'/limo/costmap/costmap_debug_{self.topic_suffix}', 10)
            self.roi_debug_pub = self.create_publisher(Image, f'/limo/costmap/roi_debug_{self.topic_suffix}', 10)

        self.get_logger().info(
            f'Costmap node initialized for target channel [{self.color_flag}].'
        )

    # -------------------------------------------------------------------------

    def tf_callback(self):
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.fixed_frame   # base_footprint
        t.child_frame_id = f'cv_origin_{self.topic_suffix}'
        t.transform.translation.x = 0.6
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def _crop_to_roi(self, bgr: np.ndarray) -> Tuple[np.ndarray, int]:
        h = bgr.shape[0]
        roi_bottom = int(round(h * (1.0 - self.ROI_FRACTION)))   
        roi_top = int(round(h * (1.0 - self.RULER_FRACTION)))     
        return bgr[roi_bottom:roi_top, :], roi_bottom, roi_top

    def _make_mask(self, bgr: np.ndarray, exact_color: np.ndarray) -> np.ndarray:
        lo = np.clip(exact_color.astype(np.int16) - self.TOLERANCE, 0, 255).astype(np.uint8)
        hi = np.clip(exact_color.astype(np.int16) + self.TOLERANCE, 0, 255).astype(np.uint8)
        return cv2.inRange(bgr, lo, hi)

    def _inflate_layer(self, mask: np.ndarray, peak_cost: float, radius_px: int) -> np.ndarray:
        obstacle = cv2.bitwise_not(mask)
        dist = cv2.distanceTransform(obstacle, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        norm_dist = dist / max(radius_px, 1)
        cost_layer = peak_cost * np.exp(-self.DECAY * norm_dist)
        cost_layer[dist > radius_px] = 0.0
        return cost_layer.astype(np.float32)

    def _image_to_costmap(self, bgr: np.ndarray) -> np.ndarray:
        """Processes ONLY the configured color channel to save CPU overhead."""
        target_color = self.COLOR_MAP[self.color_flag]
        config = self.CONFIG_MAP[self.color_flag]

        # Extract the specific mask
        mask = self._make_mask(bgr, target_color)

        # Cross-channel disambiguation to separate Red from Magenta
        if self.color_flag == 'RED':
            mask_magenta = self._make_mask(bgr, self.COLOR_MAP['MAGENTA'])
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_magenta))

        # Build single layer cost representation
        combined = self._inflate_layer(mask, config['peak_cost'], config['radius'])

        return np.clip(combined, 0, 100).astype(np.uint8)

    def _costmap_to_occupancy_grid(self, cost_img: np.ndarray, header: Header) -> OccupancyGrid:
        rotated_cost = cv2.rotate(cost_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_cost = cv2.flip(rotated_cost, 1)

        h, w = rotated_cost.shape
        grid = OccupancyGrid()
        
        grid.header.stamp = TimeMsg()        
        grid.header.frame_id = f'cv_origin_{self.topic_suffix}'  

        grid.info.resolution = self.resolution
        grid.info.width = w
        grid.info.height = h

        grid.info.origin.position.x = 0.0   
        grid.info.origin.position.y = -(h * self.resolution) / 2.0
        grid.info.origin.orientation.w = 1.0

        grid.data = rotated_cost.flatten().tolist()
        return grid

    def _render_debug(self, cost_img: np.ndarray) -> np.ndarray:
        normalized = (cost_img.astype(np.float32) / 100.0 * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        heatmap[cost_img == 0] = (0, 0, 0)
        return heatmap

    def _render_roi_debug(self, full_bgr: np.ndarray, roi_bottom: int, roi_top: int) -> np.ndarray:
        debug = full_bgr.copy()
        h, w = full_bgr.shape[0], full_bgr.shape[1]

        overlay = debug[:roi_bottom, :].copy()
        cv2.addWeighted(overlay, 0.3, np.zeros_like(overlay), 0.7, 0, debug[:roi_bottom, :])

        cv2.line(debug, (0, roi_bottom), (w - 1, roi_bottom), color=(0, 255, 0), thickness=2)

        cv2.putText(debug, f'ROI: bottom {self.ROI_FRACTION:.0%}', (8, roi_bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        ruler_y = roi_top

        cv2.line(debug, (0, ruler_y), (w - 1, ruler_y), color=(255, 255, 0), thickness=1)

        for x in range(0, w, 10):
            if x % 50 == 0:
                cv2.line(debug, (x, ruler_y - 8), (x, ruler_y + 8), color=(255, 255, 0), thickness=2)
                if x > 0 and x < w - 20:
                    cv2.putText(debug, str(x), (x - 10, ruler_y - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.line(debug, (x, ruler_y - 4), (x, ruler_y + 4), color=(255, 255, 0), thickness=1)

        return debug

    # -------------------------------------------------------------------------

    def bev_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        roi_bgr, roi_bottom, roi_top = self._crop_to_roi(bgr)

        if self.publish_debug:
            roi_debug_bgr = self._render_roi_debug(bgr, roi_bottom, roi_top)
            roi_debug_msg = self.bridge.cv2_to_imgmsg(roi_debug_bgr, encoding='bgr8')
            roi_debug_msg.header = msg.header
            self.roi_debug_pub.publish(roi_debug_msg)

        cost_img_cropped = self._image_to_costmap(roi_bgr)

        self.costmap_pub.publish(self._costmap_to_occupancy_grid(cost_img_cropped, msg.header))

        if self.publish_debug:
            debug_bgr = self._render_debug(cost_img_cropped)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_bgr, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Costmap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()