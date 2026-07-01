#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class RoutesBuilder(Node):

    def __init__(self):
        super().__init__('map_to_heatmap_image_publisher')

        # --- PARAMETER DECLARATIONS (WITH DEFAULTS) ---
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('image_topic', '/limo/routes/global_map_heatmap')
        self.declare_parameter('high_cost_image_topic', '/limo/routes/high_cost_heatmap')
        self.declare_parameter('debug_distance_topic', '/limo/routes/debug_distance_heatmap')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('update_rate', 0.05) # 20 Hz
        self.declare_parameter('mask_occupancy_topic', '/limo/routes/selected_mask_grid')

        # FLAG: OPEN or CENTER_ROAD
        self.declare_parameter('flag', 'OPEN')

        # Cost values for high cost debug isolation
        self.declare_parameter('high_cost_min',     60)
        self.declare_parameter('high_cost_max',    100)

        # Distance threshold for open space zones 
        self.declare_parameter('open_cost_threshold',       60)
        self.declare_parameter('open_distance_threshold', 0.05) # meters

        # Thresholds for center road zones
        self.declare_parameter('center_cost_threshold', 0)
        self.declare_parameter('center_distance_min', 0.10) # meters
        self.declare_parameter('center_distance_max', 0.12) # meters

        # --- FETCH STATIC TOPIC & RATE CONFIGS ---
        flag_value = self.get_parameter('flag').value

        VALID_FLAGS = ['OPEN', 'CENTER_ROAD']
        DEFAULT_FLAG = 'OPEN'

        if flag_value.upper() not in VALID_FLAGS:
            self.get_logger().error(
                f"Invalid flag '{flag_value}' provided. Reverting to default: '{DEFAULT_FLAG}'"
            )
            
            self.set_parameters([rclpy.parameter.Parameter('flag', rclpy.Parameter.Type.STRING, DEFAULT_FLAG)])

            self.flag = DEFAULT_FLAG
        else:
            self.flag = flag_value.upper()

        map_topic = self.get_parameter('map_topic').value
        image_topic = self.get_parameter('image_topic').value
        high_cost_image_topic = self.get_parameter('high_cost_image_topic').value
        debug_distance_topic_general = self.get_parameter('debug_distance_topic').value
        debug_distance_topic = f"{debug_distance_topic_general}_{self.flag.lower()}"
        mask_occupancy_topic = f"{self.get_parameter('mask_occupancy_topic').value}_{self.flag.lower()}"

        self.robot_frame = self.get_parameter('robot_frame').value
        update_rate = self.get_parameter('update_rate').value

        self.high_cost_min = self.get_parameter('high_cost_min').value
        self.high_cost_max = self.get_parameter('high_cost_max').value

        self.open_cost_thresh = self.get_parameter('open_cost_threshold').value
        self.open_dist_thresh = self.get_parameter('open_distance_threshold').value

        self.center_cost_thresh = self.get_parameter('center_cost_threshold').value
        self.center_dist_min = self.get_parameter('center_distance_min').value
        self.center_dist_max = self.get_parameter('center_distance_max').value

        # --- CACHE CONTAINERS ---
        self.mask_occupancy_msg = None     # Cache container for the processed occupancy grid layer

        self.bridge = CvBridge()

        # --- TF2 SETUP ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- CACHE CONTAINERS ---
        self.latest_map_msg = None
        self.base_heatmap_img = None       
        self.high_cost_heatmap_img = None  
        self.debug_distance_img = None     

        # --- QOS & SUBSCRIPTIONS ---
        map_qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.map_callback,
            map_qos_profile
        )

        # --- PUBLISHERS ---
        self.img_pub = self.create_publisher(Image, image_topic, 10)
        self.high_cost_img_pub = self.create_publisher(Image, high_cost_image_topic, 10)
        self.debug_distance_pub = self.create_publisher(Image, debug_distance_topic, 10)
        self.mask_grid_pub = self.create_publisher(OccupancyGrid, mask_occupancy_topic, map_qos_profile)

        # --- TIMERS ---
        self.timer = self.create_timer(update_rate, self.timer_callback)
        self.get_logger().info('Parameterized Multi-Topic Heatmap Node initialized (Eloquent Compatible).')

    def map_callback(self, msg: OccupancyGrid):
        """
        Saves the incoming map message and passes it to the processing pipeline.
        """
        self.latest_map_msg = msg
        self.process_map_layers(msg)

    def process_map_layers(self, msg: OccupancyGrid):
        """
        Core image processing pipeline utilizing dynamic ROS 2 parameters.
        """
        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution

        if width == 0 or height == 0:
            return

        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        mask_unknown = grid_data == -1
        mask_occupied = grid_data > 0

        # --- 1. FULL MAP PROCESSING ---
        heatmap_src = np.zeros_like(grid_data, dtype=np.uint8)
        if np.any(mask_occupied):
            costs = grid_data[mask_occupied].astype(np.float32)
            heatmap_src[mask_occupied] = (costs / 100.0 * 255.0).astype(np.uint8)

        full_color = cv2.applyColorMap(heatmap_src, cv2.COLORMAP_JET)
        full_color[mask_unknown] = [200, 200, 200]

        # --- 2. HIGH-COST LAYER ISOLATION ---
        high_cost_src = np.zeros_like(grid_data, dtype=np.uint8)
        mask_high_cost = (grid_data >= self.high_cost_min) & (grid_data <= self.high_cost_max)
        if np.any(mask_high_cost):
            high_costs = grid_data[mask_high_cost].astype(np.float32)
            high_cost_src[mask_high_cost] = (high_costs / 100.0 * 255.0).astype(np.uint8)

        high_cost_color = cv2.applyColorMap(high_cost_src, cv2.COLORMAP_JET)
        high_cost_color[~mask_high_cost] = [0, 0, 0]

        # --- 3. DISTANCE TRANSFORM 1: CYAN ZONE ---
        distance_mask_all = np.zeros_like(grid_data, dtype=np.uint8)
        distance_mask_all[grid_data <= self.open_cost_thresh] = 255  
        dist_map_all = cv2.distanceTransform(distance_mask_all, cv2.DIST_L2, 5)
        
        pixel_threshold_open = self.open_dist_thresh / res
        mask_open_zone = (
            (dist_map_all > pixel_threshold_open) & 
            (grid_data >= 0) & (grid_data <= self.open_cost_thresh)
        )

        # --- 4. DISTANCE TRANSFORM 2: PURPLE ZONE ---
        distance_mask_gt = np.zeros_like(grid_data, dtype=np.uint8)
        distance_mask_gt[grid_data <= self.center_cost_thresh] = 255  
        dist_map_gt = cv2.distanceTransform(distance_mask_gt, cv2.DIST_L2, 5)

        pixel_thresh_center_min = self.center_dist_min / res
        pixel_thresh_center_max = self.center_dist_max / res
        
        mask_center_zone = (
            (dist_map_gt >= pixel_thresh_center_min) & 
            (dist_map_gt <= pixel_thresh_center_max) & 
            (grid_data >= 0) & (grid_data <= self.center_cost_thresh)
        )

        # --- 5. COMPOSING THE GRAYSCALE BACKGROUND + BLENDED COLOURED ZONES ---
        debug_distance_color = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Base backgrounds (Free and unknown spaces)
        debug_distance_color[:] = [40, 40, 40]
        debug_distance_color[mask_unknown] = [110, 110, 110]
        
        # Grayscale mapping of original cost values (the permanent background)
        if np.any(mask_occupied):
            occ_costs = grid_data[mask_occupied].astype(np.float32)
            gray_values = (40 + (occ_costs / 100.0 * (255.0 - 40.0))).astype(np.uint8)
            debug_distance_color[mask_occupied] = gray_values[:, np.newaxis]

        # Create a colored overlay layer initialized with the background structure
        overlay = debug_distance_color.copy()
        if self.flag == 'OPEN':
            mask_colored = mask_open_zone 
            zone_color = [255, 255, 0] 
        elif self.flag == 'CENTER_ROAD':
            mask_colored = mask_center_zone 
            zone_color = [255, 0, 255] 

        if np.any(mask_colored):
            overlay[mask_colored] = zone_color
            alpha = 0.4  # Transparency coefficient (0.0 = fully transparent, 1.0 = fully opaque)
            blended = cv2.addWeighted(debug_distance_color, 1.0 - alpha, overlay, alpha, 0)
            debug_distance_color[mask_colored] = blended[mask_colored]
        
        # --- 6. GENERATE THE NEW OCCUPANCY GRID MASK DATA ---
        # Initialize an occupancy grid array filled with 100 (Occupied/Obstacle/Non-selected)
        output_grid = np.full_like(grid_data, 100, dtype=np.int8)
        
        # Keep unknown spaces as unknown (-1)
        output_grid[mask_unknown] = -1
        
        # Set the valid selected zone path as completely free space (0)
        output_grid[mask_colored] = 0

        # Create and cache the OccupancyGrid message structure
        mask_grid_msg = OccupancyGrid()
        mask_grid_msg.info = msg.info
        mask_grid_msg.data = output_grid.flatten().tolist()
        self.mask_occupancy_msg = mask_grid_msg

        # --- 7. FLIP AND ENFORCE CONTINUOUS LAYOUT TO BUFFER ---
        self.base_heatmap_img = np.ascontiguousarray(np.flipud(full_color))
        self.high_cost_heatmap_img = np.ascontiguousarray(np.flipud(high_cost_color))
        self.debug_distance_img = np.ascontiguousarray(np.flipud(debug_distance_color))

    def timer_callback(self):
        """
        Blits the robot position over the cached layers and shoots to topics at 20Hz.
        """
        if (self.base_heatmap_img is None or 
            self.high_cost_heatmap_img is None or 
            self.debug_distance_img is None or 
            self.latest_map_msg is None):
            return

        msg = self.latest_map_msg
        width = msg.info.width
        height = msg.info.height

        img_full = self.base_heatmap_img.copy()
        img_high_cost = self.high_cost_heatmap_img.copy()
        img_debug_dist = self.debug_distance_img.copy()

        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(msg.header.frame_id, self.robot_frame, now)
            
            robot_x = trans.transform.translation.x
            robot_y = trans.transform.translation.y
            
            q = trans.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            res = msg.info.resolution
            
            px = int((robot_x - origin_x) / res)
            py = int(height - 1 - ((robot_y - origin_y) / res))
            
            if 0 <= px < width and 0 <= py < height:
                def draw_robot(canvas):
                    cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)      
                    cv2.circle(canvas, (px, py), 6, (255, 255, 255), 1)  
                    arrow_length = 18
                    dx = int(arrow_length * math.cos(-yaw))
                    dy = int(arrow_length * math.sin(-yaw))
                    cv2.arrowedLine(canvas, (px, py), (px + dx, py + dy), (0, 255, 0), 2, tipLength=0.3)

                draw_robot(img_full)
                draw_robot(img_high_cost)
                draw_robot(img_debug_dist)

        except TransformException:
            pass

        # --- PUBLISH ALL THREE IMAGES ---
        timestamp = self.get_clock().now().to_msg()

        # 1. Full Map
        msg_full = self.bridge.cv2_to_imgmsg(img_full, encoding='bgr8')
        msg_full.header.stamp = timestamp
        msg_full.header.frame_id = msg.header.frame_id
        self.img_pub.publish(msg_full)

        # 2. High Cost Layer
        msg_high = self.bridge.cv2_to_imgmsg(img_high_cost, encoding='bgr8')
        msg_high.header.stamp = timestamp
        msg_high.header.frame_id = msg.header.frame_id
        self.high_cost_img_pub.publish(msg_high)

        # 3. Debug Distance Layer
        msg_debug = self.bridge.cv2_to_imgmsg(img_debug_dist, encoding='bgr8')
        msg_debug.header.stamp = timestamp
        msg_debug.header.frame_id = msg.header.frame_id
        self.debug_distance_pub.publish(msg_debug)

        # 4. Selected Mask Occupancy Grid
        if self.mask_occupancy_msg is not None:
            self.mask_occupancy_msg.header.stamp = timestamp
            self.mask_occupancy_msg.header.frame_id = msg.header.frame_id
            self.mask_grid_pub.publish(self.mask_occupancy_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoutesBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()