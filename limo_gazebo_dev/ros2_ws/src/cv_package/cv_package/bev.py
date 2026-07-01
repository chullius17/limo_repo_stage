#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs import msg
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class BirdPerspective(Node):

    def __init__(self):
        super().__init__('bird_perspective')

        self.bridge = CvBridge()
        
        # Initialize storage variables
        self.depth_img = None
        self.fx = None

        self.rgb_sub = self.create_subscription(Image, '/detection/lines_and_curbs', self.rgb_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/limo/color/camera_info', self.camera_info_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/limo/depth/image_raw', self.depth_callback, 10)

        self.bird_pub = self.create_publisher(Image, '/limo/color/image_raw_bird_perspective', 10)

        # Kernel to dilate pixels in BEV (3x3 or 5x5 depending on how thick you want them to appear)
        self.kernel_inflate = np.ones((3, 3), dtype=np.uint8)

    def camera_info_callback(self, msg):
        # Extract intrinsic parameters
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        # Ensure depth is processed as float32 in meters
        tmp_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if tmp_depth.dtype == np.uint16:
            self.depth_img = tmp_depth.astype(np.float32) / 1000.0
        else:
            self.depth_img = tmp_depth

    def rgb_callback(self, msg):
        # SAFETY CHECK: Only proceed if we have all necessary data
        if self.depth_img is None or self.fx is None:
            self.get_logger().warn('Waiting for Depth and CameraInfo...', throttle_duration_sec=2.0)
            return
        
        self.get_logger().info('Processing new RGB frame...')
        rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # --- CRITICAL CORRECTION: IDENTIFY ONLY USABLE PIXELS ---
        # Find where the image is NOT black (at least one BGR channel is greater than zero)
        valid_color_mask = (rgb_img[:, :, 0] > 0) | (rgb_img[:, :, 1] > 0) | (rgb_img[:, :, 2] > 0)
        
        # Extract (v, u) coordinates of useful points only
        v_indices, u_indices = np.where(valid_color_mask)
        
        if len(u_indices) == 0:
            # If the input image is completely empty, publish an empty frame directly
            empty_bev = np.zeros((600, 600, 3), dtype=np.uint8)
            bird_msg = self.bridge.cv2_to_imgmsg(empty_bev, encoding='bgr8')
            bird_msg.header = msg.header
            self.bird_pub.publish(bird_msg)
            return

        # Extract only the colors associated with those specific pixels
        colors = rgb_img[v_indices, u_indices]

        # Extract the corresponding depth for useful pixels only
        z = self.depth_img[v_indices, u_indices]

        # Further filtering: discard pixels with invalid depth (NaN, Inf, or <= 0)
        valid_depth = (z > 0.1) & (~np.isnan(z)) & (~np.isinf(z))
        u = u_indices[valid_depth]
        v = v_indices[valid_depth]
        z = z[valid_depth]
        colors = colors[valid_depth]

        if len(u) == 0:
            return

        # Back-projection to Camera Frame (For valid pixels only!)
        x_c = (u - self.cx) * z / self.fx
        y_c = (v - self.cy) * z / self.fy
        camera_points = np.vstack((x_c, y_c, z, np.ones_like(x_c)))

        # 1. Define the rotation matrix for RotX(-pi/2)
        c = 0.0
        s = -1.0
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0,   c,  -s],
            [0.0,   s,   c]
        ])

        # 2. Define the translation vector (0, 0, 0)
        t = np.array([0.0, 0.0, 0.0])

        # 3. Construct the 4x4 Homogeneous Transformation Matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        # 4. Invert the matrix to match PyKDL's .Inverse()
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R.T
        extrinsic_matrix[:3, 3] = -R.T @ t

        # 5. Apply the transformation to camera points
        world_points = np.dot(extrinsic_matrix, camera_points)

        # Rasterization to BEV Image
        res = 0.01 
        side = 600
        bev_img = np.zeros((side, side, 3), dtype=np.uint8)

        # Mapping: World X -> Image U, World Z -> Image V
        u_bev = (world_points[0, :] / res + side / 2).astype(int)
        v_bev = (world_points[1, :] / res + side / 2).astype(int)

        mask = (u_bev >= 0) & (u_bev < side) & (v_bev >= 0) & (v_bev < side)
        
        # Now the overwriting process only handles actual colored pixels, 
        # meaning points will no longer cancel each other out with the black background!
        bev_img[v_bev[mask], u_bev[mask]] = colors[mask]

        # --- APPLY MORPHOLOGICAL DILATION TO INFLATE PIXELS ---
        bev_img = cv2.dilate(bev_img, self.kernel_inflate, iterations=1)

        # Publish result
        bird_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bird_msg.header = msg.header
        self.bird_pub.publish(bird_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BirdPerspective()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()