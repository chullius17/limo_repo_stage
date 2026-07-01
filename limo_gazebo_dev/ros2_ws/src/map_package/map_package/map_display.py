import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import time

class MapDisplay(Node):
    """
    Listens to processed OccupancyGrids, combines them into a high-resolution 
    global costmap canvas, generates lightweight downsampled images for visualization,
    and publishes the combined local ego canvas as a ROS2 OccupancyGrid.
    """

    def __init__(self):
        super().__init__('map_display_node')

        self.declare_parameter('global_frame',        'odom')
        self.declare_parameter('robot_frame',         'base_footprint')
        self.declare_parameter('costmap_resolution',  0.005)    # High definition for navigation (5mm/px)
        self.declare_parameter('view_resolution',     0.02)     # Lightweight definition for images (2cm/px)
        self.declare_parameter('view_range_m',        3.0)      # Semi-side of ego canvas
        self.declare_parameter('magenta_factor',      1.0)      
        self.declare_parameter('red_factor',          0.6)      
        self.declare_parameter('green_factor',        0.3)      

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                             rclpy.Parameter.Type.BOOL, True)])

        self.global_frame       = self.get_parameter('global_frame').value
        self.robot_frame        = self.get_parameter('robot_frame').value
        self.costmap_resolution = self.get_parameter('costmap_resolution').value
        self.view_resolution    = self.get_parameter('view_resolution').value
        self.view_range_m       = self.get_parameter('view_range_m').value
        self.magenta_factor     = self.get_parameter('magenta_factor').value
        self.red_factor         = self.get_parameter('red_factor').value
        self.green_factor       = self.get_parameter('green_factor').value

        # Dynamic canvas dimensions based on chosen resolutions
        self.canvas_px = int(2 * self.view_range_m / self.view_resolution)

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge      = CvBridge()

        # Storage for incoming occupancy grid messages
        self.map_data_magenta = None
        self.map_data_red     = None
        self.map_data_green   = None

        # --- SUBSCRIPTIONS WITH ROI PROJECTORS ---
        self.create_subscription(OccupancyGrid, '/limo/map_paper_magenta', self.magenta_map_callback, 10)
        self.create_subscription(OccupancyGrid, '/limo/map_paper_red',     self.red_map_callback,     10)
        self.create_subscription(OccupancyGrid, '/limo/map_paper_green',   self.green_map_callback,   10)

        # --- COMBINED IMAGE PUBLISHERS ---
        self.firstp_img_pub         = self.create_publisher(Image, '/limo/map_firstp_combined', 10)
        
        # --- NEW PUBLISHER: COMBINED OCCUPANCY GRID ---
        self.grid_pub               = self.create_publisher(OccupancyGrid, '/limo/global_map_combined', 10)

        # Pre-allocate the OccupancyGrid message to optimize CPU usage
        self.combined_grid_msg      = self.get_default_occupancy_grid()

        self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('MapDisplay (Visualization & OccupancyGrid) node started')

    def get_default_occupancy_grid(self) -> OccupancyGrid:
        """Initialize metadata for the combined occupancy grid centered on the robot."""
        msg = OccupancyGrid()
        msg.header.frame_id = self.robot_frame  # Since it is robot-centric (ego), the frame of reference is the robot itself
        msg.info.resolution = self.view_resolution
        msg.info.width = self.canvas_px
        msg.info.height = self.canvas_px
        
        # The origin must shift the grid backward and left by half of its total size
        # so that the (0,0) point relative to the robot is exactly at the center of the canvas
        msg.info.origin.position.x = -self.view_range_m
        msg.info.origin.position.y = -self.view_range_m
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        return msg

    # ------------------------------------------------------------------

    def magenta_map_callback(self, msg: OccupancyGrid):
        self.map_data_magenta = msg

    def red_map_callback(self, msg: OccupancyGrid):
        self.map_data_red = msg

    def green_map_callback(self, msg: OccupancyGrid):
        self.map_data_green = msg

    # ------------------------------------------------------------------

    def _project_grid_layer_optimized(self, m: OccupancyGrid, robot_x: float, robot_y: float,
                                        cos_y: float, sin_y: float, canvas: np.ndarray, 
                                        seen_canvas: np.ndarray, factor: float = 1.0) -> None:
        grid = np.array(m.data, dtype=np.int8).reshape((m.info.height, m.info.width)).astype(np.float32)
        unknown_mask = (grid < 0)
        grid = np.clip(grid, 0, 100) * factor
        grid[unknown_mask] = -1.0 

        res_src = m.info.resolution
        x0_src  = m.info.origin.position.x
        y0_src  = m.info.origin.position.y

        M_map_to_odom = np.array([[res_src, 0.0,     x0_src],
                                [0.0,     res_src, y0_src],
                                [0.0,     0.0,     1.0   ]], dtype=np.float32)

        M_odom_to_robot = np.array([[cos_y,  sin_y, -cos_y * robot_x - sin_y * robot_y],
                                    [-sin_y, cos_y,  sin_y * robot_x - cos_y * robot_y],
                                    [0.0,    0.0,    1.0                              ]], dtype=np.float32)

        cx = self.canvas_px / 2.0
        cy = self.canvas_px / 2.0
        M_robot_to_canvas = np.array([[0.0,                    -1.0 / self.view_resolution, cx],
                                    [-1.0 / self.view_resolution, 0.0,                     cy],
                                    [0.0,                    0.0,                     1.0]], dtype=np.float32)

        M_warp = (M_robot_to_canvas @ M_odom_to_robot @ M_map_to_odom)[0:2, :]

        local_layer = cv2.warpAffine(grid, M_warp, (self.canvas_px, self.canvas_px),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)

        observed = (local_layer >= 0)
        np.maximum(canvas, np.where(observed, local_layer, 0), out=canvas)
        np.logical_or(seen_canvas, observed, out=seen_canvas)

    # ------------------------------------------------------------------

    def timer_callback(self):
        t_timer_start = time.perf_counter()

        if self.map_data_magenta is None and self.map_data_red is None and self.map_data_green is None:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_frame, rclpy.time.Time()
            )
        except TransformException as e:
            self.get_logger().warn(f'TF not available: {e}', throttle_duration_sec=2.0)
            return

        robot_x = tf.transform.translation.x
        robot_y = tf.transform.translation.y
        q = tf.transform.rotation
        robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)

        # 1. Local Ego Canvas Projection (Fast because it uses view_resolution)
        ego_canvas = np.zeros((self.canvas_px, self.canvas_px), dtype=np.float32)
        seen_canvas = np.zeros((self.canvas_px, self.canvas_px), dtype=bool)

        if self.map_data_magenta is not None:
            self._project_grid_layer_optimized(self.map_data_magenta, robot_x, robot_y, cos_y, sin_y, ego_canvas, seen_canvas, self.magenta_factor)
        if self.map_data_red is not None:
            self._project_grid_layer_optimized(self.map_data_red, robot_x, robot_y, cos_y, sin_y, ego_canvas, seen_canvas, self.red_factor)
        if self.map_data_green is not None:
            self._project_grid_layer_optimized(self.map_data_green, robot_x, robot_y, cos_y, sin_y, ego_canvas, seen_canvas, self.green_factor)

        t_ego_done = time.perf_counter()

        # ------------------------------------------------------------------
        # NEW LOGIC: COMBINED OCCUPANCY GRID PUBLICATION
        # ------------------------------------------------------------------
        # Convert ego_canvas matrix (0-100 float) to int8 required by ROS2.
        # Optional: you can use np.rot90 or flipping if you notice axis inversions in RViz.
        grid_data = np.where(seen_canvas, np.clip(ego_canvas, 0, 100), -1).astype(np.int8)
        
        # Update timestamp and insert data into the pre-allocated grid message
        current_time = self.get_clock().now().to_msg()
        self.combined_grid_msg.header.stamp = current_time
        self.combined_grid_msg.data = grid_data.flatten().tolist()
        self.grid_pub.publish(self.combined_grid_msg)
        # ------------------------------------------------------------------

        # 2. Rendering firstp from ego canvas
        norm_e   = (np.clip(ego_canvas, 0, 100) / 100.0 * 255.0).astype(np.uint8)
        firstp   = cv2.applyColorMap(norm_e, cv2.COLORMAP_JET)
        firstp[ego_canvas == 0] = (30, 30, 30)
        half = self.canvas_px // 2
        cv2.arrowedLine(firstp, (half, half), (half, half - 25), (0, 0, 255), 2, tipLength=0.3)
        cv2.arrowedLine(firstp, (half, half), (half - 25, half), (0, 255, 0), 2, tipLength=0.3)
        cv2.circle(firstp, (half, half), 2, (255, 255, 255), -1)

        firstp_msg = self.bridge.cv2_to_imgmsg(firstp, encoding='bgr8')
        firstp_msg.header.stamp    = current_time
        firstp_msg.header.frame_id = self.robot_frame
        self.firstp_img_pub.publish(firstp_msg)

        t_end = time.perf_counter()

        self.get_logger().info(
            f"[TIMER] "
            f"ego={(t_ego_done-t_timer_start)*1000:.1f}ms "
            f"grid_pub={(t_end-t_ego_done)*1000:.1f}ms "
            f"total={(t_end-t_timer_start)*1000:.1f}ms"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MapDisplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()