import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
from pathlib import Path

def find_ros2_ws(start: Path):
    for parent in [start] + list(start.parents):
        if parent.name == "ros2_ws":
            return parent
    return None

class MapSaver(Node):
    """
    Listens to processed OccupancyGrids, combines them into a high-resolution 
    global costmap canvas on-demand (event-driven), and saves the final grid 
    as PGM/YAML upon shutdown.
    """

    def __init__(self):
        super().__init__('map_display_node')

        # --- ROS2 PARAMETERS ---
        self.declare_parameter('global_frame',        'odom')
        self.declare_parameter('view_resolution',     0.02)     
        self.declare_parameter('magenta_factor',      1.0)      
        self.declare_parameter('red_factor',          0.6)      
        self.declare_parameter('green_factor',        0.3) 
        self.declare_parameter('canvas_size_meters',  10.0)     
        self.declare_parameter('robot_frame',         'base_footprint')
        
        self.robot_frame = self.get_parameter('robot_frame').value
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                             rclpy.Parameter.Type.BOOL, True)])

        self.global_frame       = self.get_parameter('global_frame').value
        self.view_resolution    = self.get_parameter('view_resolution').value
        self.magenta_factor     = self.get_parameter('magenta_factor').value
        self.red_factor         = self.get_parameter('red_factor').value
        self.green_factor       = self.get_parameter('green_factor').value
        self.canvas_size_meters = self.get_parameter('canvas_size_meters').value

        # Dynamic canvas dimensions based on chosen resolutions
        self.canvas_px = int(self.canvas_size_meters / self.view_resolution)

        self.bridge = CvBridge()
        self.img_pub = self.create_publisher(Image, '/limo/global_map_jet', 10)

        # Storage for incoming occupancy grid messages
        self.map_data_magenta = None
        self.map_data_red     = None
        self.map_data_green   = None

        # --- SUBSCRIPTIONS ---
        self.create_subscription(OccupancyGrid, '/limo/map_paper_magenta', self.magenta_map_callback, 10)
        self.create_subscription(OccupancyGrid, '/limo/map_paper_red',     self.red_map_callback,     10)
        self.create_subscription(OccupancyGrid, '/limo/map_paper_green',   self.green_map_callback,   10)

        # Pre-allocation of the OccupancyGrid message to optimize CPU usage
        self.grid_msg_combined = self.get_default_occupancy_grid()

        ws = find_ros2_ws(Path(__file__).resolve())

        if ws is not None:
            self.save_dir_src = ws / "src" / "ros2_maps"
            self.save_dir_src.mkdir(parents=True, exist_ok=True)

        # Canvas preallocation
        self.global_canvas   = np.zeros((self.canvas_px, self.canvas_px), dtype=np.float32)
        self.seen_canvas     = np.zeros((self.canvas_px, self.canvas_px), dtype=bool)
        
        # Last valid grid stored for final storage/saving
        self.latest_grid_data = np.full((self.canvas_px, self.canvas_px), -1, dtype=np.int8)

        self.get_logger().info('Map Saver node started (Save on Shutdown mode)')

    def get_default_occupancy_grid(self) -> OccupancyGrid:
        """Initializes metadata for the combined occupancy grid centered on the robot."""
        msg = OccupancyGrid()
        msg.header.frame_id = self.global_frame
        msg.info.resolution = self.view_resolution
        msg.info.width = self.canvas_px
        msg.info.height = self.canvas_px
        half = self.canvas_size_meters / 2.0
        msg.info.origin.position.x = -half
        msg.info.origin.position.y = -half
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        return msg

    def magenta_map_callback(self, msg: OccupancyGrid):
        self.map_data_magenta = msg
        self.update_and_publish_map()

    def red_map_callback(self, msg: OccupancyGrid):
        self.map_data_red = msg
        self.update_and_publish_map()

    def green_map_callback(self, msg: OccupancyGrid):
        self.map_data_green = msg
        self.update_and_publish_map()

    def _project_grid_layer_optimized(self, m: OccupancyGrid,
                                    canvas: np.ndarray,
                                    seen_canvas: np.ndarray,
                                    factor: float = 1.0) -> None:
        grid = np.array(m.data, dtype=np.int8).reshape(
            (m.info.height, m.info.width)).astype(np.float32)

        unknown_mask = (grid < 0)
        grid = np.clip(grid, 0, 100) * factor
        grid[unknown_mask] = -1.0

        observed = (grid >= 0)
        np.maximum(canvas, np.where(observed, grid, 0), out=canvas) 
        np.logical_or(seen_canvas, observed, out=seen_canvas)

    def save_map_to_disk(self):
        """
        Saves the map by converting cost levels into proportional grayscale values 
        for RViz/Nav2, avoiding rendering everything as black.
        """
        # Initialize the saving canvas as "Unknown" (Grayscale 205)
        pgm_img = np.full_like(self.latest_grid_data, 205, dtype=np.uint8)
        
        # 1. Free Space (value 0 in grid_data -> 254 in PGM)
        pgm_img[self.latest_grid_data == 0] = 254  
        
        # 2. Cost Levels / Obstacles (values from 1 to 100)
        # Map proportionally: cost 100 -> grayscale 0 (black), cost 1 -> grayscale ~250 (almost white)
        mask_occupied = self.latest_grid_data > 0
        costs = self.latest_grid_data[mask_occupied].astype(np.float32)
        
        # Inverted linear conversion formula to make high costs appear darker
        pgm_img[mask_occupied] = (254 - (costs / 100.0 * 254.0)).astype(np.uint8)

        # Vertical flip to synchronize the OpenCV origin with the ROS display orientation
        pgm_img = np.flipud(pgm_img)

        img_path = os.path.join(self.save_dir_src, "limo_map.pgm")
        yaml_path = os.path.join(self.save_dir_src, "limo_map.yaml")
        
        cv2.imwrite(img_path, pgm_img)

        half = self.canvas_size_meters / 2.0
        yaml_content = (
            f"image: limo_map.pgm\n"
            f"resolution: {self.view_resolution}\n"
            f"origin: [{-half}, {-half}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.196\n"
            f"mode: scale\n"
        )
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        self.get_logger().info(f"Map flipped and saved with cost levels in: {self.save_dir_src}!")

    def update_and_publish_map(self):
        if self.map_data_magenta is None and self.map_data_red is None and self.map_data_green is None:
            return

        # CRITICAL: Clear canvases before re-projecting to avoid incorrect data accumulation
        self.global_canvas.fill(0)
        self.seen_canvas.fill(False)

        if self.map_data_magenta is not None:
            self._project_grid_layer_optimized(self.map_data_magenta, self.global_canvas, self.seen_canvas, self.magenta_factor)
        if self.map_data_red is not None:
            self._project_grid_layer_optimized(self.map_data_red, self.global_canvas, self.seen_canvas, self.red_factor)
        if self.map_data_green is not None:
            self._project_grid_layer_optimized(self.map_data_green, self.global_canvas, self.seen_canvas, self.green_factor)

        grid_data = np.where(self.seen_canvas,
                             np.clip(self.global_canvas, 0, 100),
                             -1).astype(np.int8)

        self.latest_grid_data = grid_data

        # --- JET Visualization (Unchanged, processing in parallel) ---
        norm = (np.clip(self.global_canvas, 0, 100) / 100.0 * 255.0).astype(np.uint8)
        jet_img = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        unseen = ~self.seen_canvas
        jet_img[unseen] = (30, 30, 30)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_frame, rclpy.time.Time()
            )
            robot_x = tf.transform.translation.x
            robot_y = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            half_m = self.canvas_size_meters / 2.0
            rx = int((robot_x + half_m) / self.view_resolution)
            ry = int((robot_y + half_m) / self.view_resolution)

            if 0 <= rx < self.canvas_px and 0 <= ry < self.canvas_px:
                arrow_len = 25
                dx_x = int(arrow_len * math.cos(yaw))
                dy_x = int(arrow_len * math.sin(yaw))
                cv2.arrowedLine(jet_img, (rx, ry), (rx + dx_x, ry - dy_x), (0, 0, 255), 2, tipLength=0.3)

                dx_y = int(-arrow_len * math.sin(yaw))
                dy_y = int(arrow_len * math.cos(yaw))
                cv2.arrowedLine(jet_img, (rx, ry), (rx + dx_y, ry - dy_y), (0, 255, 0), 2, tipLength=0.3)
                cv2.circle(jet_img, (rx, ry), 3, (255, 255, 255), -1)

        except TransformException as e:
            self.get_logger().warn(f'TF not available: {e}', throttle_duration_sec=2.0)

        img_msg = self.bridge.cv2_to_imgmsg(jet_img, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = self.global_frame
        self.img_pub.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MapSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('CTRL+C received, saving map...')
    finally:
        # Saving safely occurs here during controlled node shutdown
        node.save_map_to_disk()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()