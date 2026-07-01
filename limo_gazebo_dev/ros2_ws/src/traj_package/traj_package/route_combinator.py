#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RouteCombinator(Node):

    def __init__(self):
        super().__init__('routes_coordinator')

        # --- PARAMETER DECLARATIONS (WITH DEFAULTS) ---
        self.declare_parameter('input_map_topic', '/map')
        self.declare_parameter('open_mask_topic', '/limo/routes/selected_mask_grid_open')
        self.declare_parameter('center_road_mask_topic', '/limo/routes/selected_mask_grid_center_road')
        self.declare_parameter('output_coordinated_map_topic', '/limo/routes/coordinated_map')

        # --- FETCH TOPIC CONFIGURATIONS ---
        input_map_topic = self.get_parameter('input_map_topic').value
        open_mask_topic = self.get_parameter('open_mask_topic').value
        center_road_mask_topic = self.get_parameter('center_road_mask_topic').value
        output_coordinated_map_topic = self.get_parameter('output_coordinated_map_topic').value

        # --- CACHE CONTAINERS FOR MAPS ---
        self.latest_input_map = None
        self.latest_open_mask = None
        self.latest_center_road_mask = None
        self.coordinated_map_msg = None

        # --- QOS PROFILE MATCHING THE ENVIRONMENT ---
        # Transient Local is required to receive maps that are published latched
        map_qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # --- SUBSCRIPTIONS ---
        self.input_map_sub = self.create_subscription(
            OccupancyGrid,
            input_map_topic,
            self.input_map_callback,
            map_qos_profile
        )

        self.open_mask_sub = self.create_subscription(
            OccupancyGrid,
            open_mask_topic,
            self.open_mask_callback,
            map_qos_profile
        )

        self.center_mask_sub = self.create_subscription(
            OccupancyGrid,
            center_road_mask_topic,
            self.center_road_mask_callback,
            map_qos_profile
        )

        # --- PUBLISHERS ---
        self.coordinated_map_pub = self.create_publisher(
            OccupancyGrid,
            output_coordinated_map_topic,
            map_qos_profile
        )

        self.debug_pub = self.create_publisher(
            Image,
            '/limo/routes/debug_heatmap',
            10
        )

        self.bridge = CvBridge()

        self.get_logger().info('Routes Coordinator Node successfully initialized.')

        # --- TIMER ---
        self.timer = self.create_timer(0.05, self.timer_callback)

    # --- CALLBACK METHODS ---

    def input_map_callback(self, msg: OccupancyGrid):
        """
        Callback triggered whenever a new baseline input map is received.
        """
        self.latest_input_map = msg
        self.get_logger().info(
            f"Received Input Map. Size: {msg.info.width}x{msg.info.height}, Res: {msg.info.resolution:.3f}m",
            throttle_duration_sec=5.0
        )
        self.sync_and_process()

    def open_mask_callback(self, msg: OccupancyGrid):
        """
        Callback triggered whenever a new OPEN zone occupancy mask is received.
        """
        self.latest_open_mask = msg
        self.get_logger().info(
            f"Received OPEN mask grid.",
            throttle_duration_sec=5.0
        )
        self.sync_and_process()

    def center_road_mask_callback(self, msg: OccupancyGrid):
        """
        Callback triggered whenever a new CENTER_ROAD zone occupancy mask is received.
        """
        self.latest_center_road_mask = msg
        self.get_logger().info(
            f"Received CENTER_ROAD mask grid.",
            throttle_duration_sec=5.0
        )
        self.sync_and_process()

    # --- COORDINATION LOGIC ---

    def sync_and_process(self):
        """
        Verifies if all required grids are available, builds the combined cost map, and publishes it.
        """
        # Ensure we have received data from all three channels
        if (self.latest_input_map is None or 
            self.latest_open_mask is None or 
            self.latest_center_road_mask is None):
            return

        # Verify that spatial parameters match across grids to avoid segmentation faults on reshape
        if (self.latest_open_mask.info.width != self.latest_input_map.info.width or
            self.latest_center_road_mask.info.width != self.latest_input_map.info.width or
            self.latest_open_mask.info.height != self.latest_input_map.info.height or
            self.latest_center_road_mask.info.height != self.latest_input_map.info.height):
            self.get_logger().warn("Synchronized map frames detect a dimension mismatch!", throttle_duration_sec=10.0)
            return

        # Convert 1D message data into 2D numpy arrays for efficient mask operations
        width = self.latest_input_map.info.width
        height = self.latest_input_map.info.height
        
        open_data = np.array(self.latest_open_mask.data, dtype=np.int8).reshape((height, width))
        center_data = np.array(self.latest_center_road_mask.data, dtype=np.int8).reshape((height, width))

        # --- COST MAPPING LOGIC ---
        # Initialize the output grid entirely with 100 (Obstacles / Non-selected / Unknown areas)
        output_grid = np.full((height, width), 100, dtype=np.int8)

        # Condition 1: Free space in OPEN mask (value == 0) gets cost 50
        mask_open_free = (open_data == 0)
        output_grid[mask_open_free] = 50

        # Condition 2: Free space in CENTER_ROAD mask (value == 0) gets cost 0 (overwrites OPEN)
        mask_center_free = (center_data == 0)
        output_grid[mask_center_free] = 0

        # --- ASSEMBLE AND PUBLISH THE COORDINATED MAP ---
        coordinated_msg = OccupancyGrid()
        
        # Synchronize headers and structural info
        coordinated_msg.header.stamp = self.get_clock().now().to_msg()
        coordinated_msg.header.frame_id = self.latest_input_map.header.frame_id
        coordinated_msg.info = self.latest_input_map.info
        
        # Flatten back to 1D list expected by ROS 2 message definition
        coordinated_msg.data = output_grid.flatten().tolist()
        
        self.coordinated_map_msg = coordinated_msg

        # normalize for visualization
        vis = np.zeros((height, width, 3), dtype=np.uint8)

        # obstacles (100) → dark red
        vis[output_grid == 100] = (0, 0, 50)

        # open space (50) → blue-ish
        vis[output_grid == 50] = (50, 50, 200)

        # center road (0) → green
        vis[output_grid == 0] = (0, 255, 0)

        vis = cv2.flip(vis, 0)

        img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        img_msg.header = coordinated_msg.header
        self.debug_pub.publish(img_msg)

    def timer_callback(self):

        if self.coordinated_map_msg is None:
            return

        self.coordinated_map_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        self.coordinated_map_pub.publish(
            self.coordinated_map_msg
        )

def main(args=None):
    rclpy.init(args=args)
    node = RouteCombinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()