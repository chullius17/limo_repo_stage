import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from typing import Tuple
import math

# TF2 imports
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster

class Pen(Node):
    def __init__(self):
        # Initialize the ROS 2 node
        super().__init__('pen_node')
        
        self.declare_parameter('fixed_frame', 'base_footprint')
        self.declare_parameter('resolution', 0.1)  # 10 cm per pixel

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', 
                         rclpy.Parameter.Type.BOOL, True)])

        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.resolution = self.get_parameter('resolution').value

        # TF2 Listener setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # TF2 Broadcaster to publish the new frame visualizer
        self.tf_broadcaster = TransformBroadcaster(self)

        # Create publishers
        self.publisher_ = self.create_publisher(OccupancyGrid, 'map', 10)
        self.debug_publisher = self.create_publisher(Image, 'debug_image', 10)
        
        # Timer updated to 10Hz (0.1s) for fluid TF tracking, map is published inside
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Pre-generate the map data since it is static relative to its own frame
        self.img_array = self.generate_circle_image()
        self.rotated_img = cv2.rotate(self.img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.map_data = self.rotated_img.flatten().astype(np.int8).tolist()

    def generate_circle_image(self) -> np.ndarray:
        # Create a 40x40 black image (0 intensity)
        image_size = (40, 40)
        image = np.zeros(image_size, dtype=np.uint8)
        
        # Define circle parameters
        center = (20, 20)  
        radius = 10        
        color = 100        
        thickness = -1     
        
        # Draw the circle
        cv2.circle(image, center, radius, color, thickness)
        
        return image

    def timer_callback(self):
        now = self.get_clock().now().to_msg()

        # 1. Broadcast TF
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.fixed_frame
        t.child_frame_id = 'costmap_origin'
        t.transform.translation.x = 0.5
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        # 2. Publish map with zero stamp → RViz uses latest available TF
        from builtin_interfaces.msg import Time as TimeMsg
        zero_stamp = TimeMsg()  # sec=0, nanosec=0 → "use latest transform"
        
        msg = OccupancyGrid()
        msg.header.stamp = zero_stamp   # <-- key fix
        msg.header.frame_id = 'costmap_origin'

        width = self.rotated_img.shape[1]
        height = self.rotated_img.shape[0]
        msg.info.resolution = self.resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = -((width * self.resolution) / 2.0)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = self.map_data

        self.publisher_.publish(msg)
        self.debug_publisher.publish(self.bridge.cv2_to_imgmsg(self.img_array, "mono8"))

def main(args=None):
    rclpy.init(args=args)
    node = Pen()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()