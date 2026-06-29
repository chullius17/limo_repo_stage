import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class Mapper(Node):
    def __init__(self):
        super().__init__('mapper_node')

        # --- ROS2 PARAMETERS ---
        self.declare_parameter('color', 'MAGENTA') # Options: MAGENTA, RED, GREEN
        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('resolution', 0.02)
        self.declare_parameter('map_size_meters', 10.0)

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                             rclpy.Parameter.Type.BOOL, True)])

        self.color_flag = self.get_parameter('color').value.upper()
        self.global_frame = self.get_parameter('global_frame').value
        self.resolution = self.get_parameter('resolution').value
        self.map_size_meters = self.get_parameter('map_size_meters').value

        self.map_size_pixels = int(self.map_size_meters / self.resolution)

        # --- TF2 CONFIGURATION ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- DYNAMIC TOPIC CONFIGURATION ---
        self.color_suffix = self.color_flag.lower()
        costmap_topic = f'/limo/costmap/costmap_grid_{self.color_suffix}'
        map_topic = f'/limo/map_paper_{self.color_suffix}'

        self.filtered_publisher = self.create_publisher(OccupancyGrid, map_topic, 10)

        # --- BAYESIAN LOG-ODDS FILTER CONFIGURATION ---
        # The global canvas stores log-odds to accelerate probabilistic calculations.
        self.canvas_logodds = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=np.float32)
        self.seen_canvas = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=bool)
        self.L_OCC = 0.85    # Certainty increment if an obstacle is detected
        self.L_FREE = 0.35    # Decrement if free space is detected (reduced value for smooth clearing)
        self.L_MAX = 5.0     # Maximum saturation point of the memory
        self.L_MIN = -3.0    # Minimum saturation point of the map

        self.costmap = None
        
        # Geometric Caching variables to avoid continuous allocations on the CPU
        self.cached_local_x = None
        self.cached_local_y = None
        self.cached_shape = None
        self.cached_roi_mask = None  # Local mask of the actual field of view cone

        # Subscription and Timer at 10Hz
        self.costmap_sub = self.create_subscription(OccupancyGrid, costmap_topic, self.costmap_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Pre-allocation of the output message to optimize real-time performance
        self.grid_msg_filtered = self.get_default_occupancy_grid()

        self.get_logger().info(f'Probabilistic Mapper initialized for channel [{self.color_flag}] on topic {costmap_topic}')

    def get_default_occupancy_grid(self) -> OccupancyGrid:
        """Initializes standard metadata for the global occupancy grid map."""
        msg = OccupancyGrid()
        msg.header.frame_id = self.global_frame
        msg.info.resolution = self.resolution
        msg.info.width = self.map_size_pixels
        msg.info.height = self.map_size_pixels
        half = self.map_size_meters / 2.0
        msg.info.origin.position.x = -half
        msg.info.origin.position.y = -half
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        return msg

    def costmap_callback(self, msg: OccupancyGrid):
        self.costmap = msg

    def timer_callback(self):
        if self.costmap is None:
            return

        try:
            # Retrieving the transformation between the global world (odom) and the local sensor origin
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, f'cv_origin_{self.color_suffix}', rclpy.time.Time()
            )
            origin_x = tf.transform.translation.x
            origin_y = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            cw = self.costmap.info.width
            ch = self.costmap.info.height
            
            data = np.array(self.costmap.data, dtype=np.uint8).reshape((ch, cw))

            # --- GEOMETRIC CACHING & ROI MASK GENERATION ---
            # Executed only if the dimensions of the incoming frame change
            if self.cached_shape != (ch, cw):
                cres = self.costmap.info.resolution
                cox = self.costmap.info.origin.position.x
                coy = self.costmap.info.origin.position.y
                
                cols = np.arange(cw)
                rows = np.arange(ch)
                cc, rr = np.meshgrid(cols, rows)

                # X and Y coordinates expressed in meters relative to the cv_origin_[color] frame
                self.cached_local_x = cox + cc * cres + cres / 2.0
                self.cached_local_y = coy + rr * cres + cres / 2.0
                self.cached_shape = (ch, cw)

                # --- MANUAL ROI TRAPEZOID CONFIGURATION ---
                # Modify these parameters by looking at the shape of the blue beam in RViz to make it match perfectly!
                min_x = 0  # Rear limit of the field of view relative to cv_origin (in meters)
                max_x = 1.85   # Maximum visible front limit from the camera (in meters) 
                
                width_at_min_x = 0.70  # Total width of the beam at the closest point (min_x)
                width_at_max_x = 2.2  # Total width of the beam at the farthest point (max_x)

                # Linear interpolation of the allowed semi-width along the X axis
                slope = (width_at_max_x - width_at_min_x) / (max_x - min_x)
                max_allowed_y = (width_at_min_x / 2.0) + slope * (self.cached_local_x - min_x)
                # max_allowed_y is a vector

                # Generation of the local boolean mask to isolate the useful field of view
                self.cached_roi_mask = (self.cached_local_x >= min_x) & \
                                       (self.cached_local_x <= max_x) & \
                                       (np.abs(self.cached_local_y) <= max_allowed_y)

            # Vectorial projection of the local map onto global map coordinates (Rotation-translation)
            world_x = origin_x + cos_yaw * self.cached_local_x - sin_yaw * self.cached_local_y
            world_y = origin_y + sin_yaw * self.cached_local_x + cos_yaw * self.cached_local_y

            half = self.map_size_meters / 2.0
            px = ((world_x + half) / self.resolution).astype(np.int32)
            py = ((world_y + half) / self.resolution).astype(np.int32)

            # Filter to exclude pixels accidentally projected outside the global canvas
            inside_canvas = (px >= 0) & (px < self.map_size_pixels) & (py >= 0) & (py < self.map_size_pixels)

            # --- PROBABILISTIC BAYESIAN FILTER WITH ROI COVERAGE ---
            update_matrix = np.zeros_like(self.canvas_logodds)
            
            # 1. OBSTACLE ACCUMULATION: If the sensor detects an obstacle (>0), we add it to the map
            occ_mask = inside_canvas & (data > 0)
            update_matrix[py[occ_mask], px[occ_mask]] = self.L_OCC

            # 2. OBSTACLE REMOVAL (BUG SOLVED): We subtract certainty points ONLY if the cell is 0
            # AND it is physically inside the geometric ROI visible by the robot!
            free_mask = inside_canvas & (data == 0) & self.cached_roi_mask
            update_matrix[py[free_mask], px[free_mask]] = -self.L_FREE

            # Mark all pixels inside the ROI that are within the canvas as "seen"
            seen_mask = inside_canvas & self.cached_roi_mask
            self.seen_canvas[py[seen_mask], px[seen_mask]] = True

            # Applying the update and saturation within stability limits
            self.canvas_logodds = np.clip(self.canvas_logodds + update_matrix, self.L_MIN, self.L_MAX)
            
            # Inverse conversion: Log-Odds -> Probability via Sigmoid function
            prob_matrix = 1.0 / (1.0 + np.exp(-self.canvas_logodds))
            canvas_filtered = (prob_matrix * 100.0).astype(np.int8)

        except TransformException as e:
            self.get_logger().warn(f'TF not available: {e}', throttle_duration_sec=2.0)
            return

        # --- OCCUPANCY GRID MAP PUBLICATION ---
        # np.where(condition, value if true, value if false)
        canvas_out = np.where(self.seen_canvas, canvas_filtered, np.int8(-1))
        self.grid_msg_filtered.header.stamp = self.get_clock().now().to_msg()
        self.grid_msg_filtered.data = canvas_out.flatten().tolist()
        self.filtered_publisher.publish(self.grid_msg_filtered)


def main(args=None):
    rclpy.init(args=args)
    node = Mapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()