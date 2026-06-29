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
        self.declare_parameter('map_size_m', 10.0)

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