import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np

# TF2 imports for tracking the robot position
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class GlobalMapper(Node):
    """
    Listens to the local occupancy grid and projects it onto a permanent,
    fixed global map frame ('odom') using TF2 transforms.
    """

    def __init__(self):
        super().__init__('global_mapper_node')

        self.declare_parameter('local_frame', 'base_footprint')
        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('map_size_m', 20.0)

        self.local_frame = self.get_parameter('local_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.map_size_m = self.get_parameter('map_size_m').value

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Global map variables (initialized on the first received message)
        self.resolution = None
        self.global_w = None
        self.global_h = None
        self.global_cost_data = None

        # Subscribers and Publishers
        self.local_sub = self.create_subscription(
            OccupancyGrid,
            '/limo/costmap/costmap_grid',
            self.local_costmap_callback,
            10
        )
        self.global_pub = self.create_publisher(OccupancyGrid, '/limo/costmap/global_costmap_grid', 10)

        self.get_logger().info(f'Global Mapper started. Integrating local maps into fixed frame: {self.global_frame}')

    def local_costmap_callback(self, msg: OccupancyGrid):
        # Initialize the global map dimensions dynamically based on incoming map resolution
        if self.global_cost_data is None:
            self.resolution = msg.info.resolution
            self.global_w = int(self.map_size_m / self.resolution)
            self.global_h = int(self.map_size_m / self.resolution)
            self.global_cost_data = np.zeros((self.global_h, self.global_w), dtype=np.uint8)
            self.get_logger().info(f'Initialized global map matrix: {self.global_w}x{self.global_h}px')

        # Convert the incoming flattened list back into a 2D local OpenCV image
        local_w = msg.info.width
        local_h = msg.info.height
        local_cost = np.array(msg.data, dtype=np.uint8).reshape((local_h, local_w))

        # Look up the transformation from global frame (odom) to local frame (base_footprint)
        try:
            # FORCE lookup to use the latest available transform in the cache (Time 0)
            # This completely bypasses simulation/hardware clock desynchronization issues
            t = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.local_frame,
                rclpy.time.Time()  # Using Time(0) tells TF2 to give us the latest frame available
            )
            
            robot_x = t.transform.translation.x
            robot_y = t.transform.translation.y
            
            # Convert quaternion to yaw angle
            q = t.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            robot_yaw = np.arctan2(siny_cosp, cosy_cosp)

            # Integrate this local frame into the permanent global sheet
            self._project_into_global(local_cost, msg.info, robot_x, robot_y, robot_yaw)

        except TransformException as ex:
            # We degrade this to a debug/info log or silent pass to avoid terminal flooding
            # during the first few milliseconds of node startup
            return

        # Publish the updated permanent map using the current node time to keep RViz happy
        self._publish_global_map(self.get_clock().now().to_msg())

    def _project_into_global(self, local_cost: np.ndarray, info, robot_x: float, robot_y: float, robot_yaw: float):
        # 1. Map local grid pixels (c, r) to local robot meters (x_robot, y_robot)
        # ROS OccupancyGrid data starts from the bottom-left corner (info.origin)
        T_local_pixel_to_robot = np.array([
            [self.resolution, 0, info.origin.position.x],
            [0, self.resolution, info.origin.position.y],
            [0, 0, 1]
        ], dtype=np.float64)

        # 2. Map local robot meters to global world meters using the robot pose from TF2
        cos_g = np.cos(robot_yaw)
        sin_g = np.sin(robot_yaw)
        M_robot_to_world_meters = np.array([
            [cos_g, -sin_g, robot_x],
            [sin_g,  cos_g, robot_y],
            [0,      0,     1]
        ], dtype=np.float64)

        # 3. Map global world meters to the permanent global grid pixel positions (centered)
        T_world_meters_to_global_grid = np.array([
            [1.0 / self.resolution, 0, self.global_w / 2.0],
            [0, 1.0 / self.resolution, self.global_h / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)

        # Combine all forward transformations (Local Pixels -> Global Pixels)
        M_combined = T_world_meters_to_global_grid @ M_robot_to_world_meters @ T_local_pixel_to_robot

        # cv2.warpAffine requires the inverse mapping (Global Pixels -> Local Pixels) by default.
        # Explicitly inverting the matrix guarantees maximum mathematical accuracy.
        M_inverse = np.linalg.inv(M_combined)
        M_affine = M_inverse[0:2, :]

        # Warp the local map onto the permanent global canvas size
        local_projected = cv2.warpAffine(
            local_cost, M_affine, (self.global_w, self.global_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Apply a 5% decay to old data to clear out past traces over time
        self.global_cost_data = (self.global_cost_data * 0.95).astype(np.uint8)
        
        # Merge the newly projected cost tracking the maximum values
        self.global_cost_data = np.maximum(self.global_cost_data, local_projected)

    def _publish_global_map(self, stamp):
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = self.global_frame
        grid.info.resolution = self.resolution
        grid.info.width = self.global_w
        grid.info.height = self.global_h

        # The origin points to the bottom-left corner of the 10x10m square
        grid.info.origin.position.x = -(self.global_w * self.resolution) / 2.0
        grid.info.origin.position.y = -(self.global_h * self.resolution) / 2.0
        grid.info.origin.orientation.w = 1.0

        grid.data = self.global_cost_data.flatten().tolist()
        self.global_pub.publish(grid)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()