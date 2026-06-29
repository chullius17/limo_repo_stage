#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CurbDetector(Node):

    def __init__(self):
        super().__init__('curb_detector')
        self.bridge = CvBridge()
        
        # Subscription to the segmenter masks topic
        self.image_sub = self.create_subscription(Image, '/detection/lane_masks', self.process_image_callback, 10)
        
        # --- TWO DISTINCT PUBLISHERS ---
        # Publisher 1: All previous masks + 3x3 purple squares for human visualization
        self.debug_pub = self.create_publisher(Image, '/detection/curb_points_debug', 10)
        
        # Publisher 2: SINGLE pixels ONLY (1px lines and 1px curb) READY FOR BEV
        self.lines_pub = self.create_publisher(Image, '/detection/lines_and_curbs', 10)
        
        # Small kernels for low resolution (112x112)
        self.kernel_close = np.ones((5, 5), dtype=np.uint8)
        self.kernel_edge = np.ones((5, 5), dtype=np.uint8)
        self.kernel_bg_clean = np.ones((3, 3), dtype=np.uint8)
        
        # Cross-shaped structuring element for millimeter-accurate skeletonization
        self.kernel_skel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        # Spatial sampling step on the small map
        self.sampling_step = 3

    def process_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return
        
        high_h, high_w, _ = cv_image.shape

        # --- STEP 1: IMMEDIATE DOWN-SAMPLING TO LOW RESOLUTION ---
        LOW_RES_SIZE = 300
        low_res = cv2.resize(cv_image, (LOW_RES_SIZE, LOW_RES_SIZE), interpolation=cv2.INTER_NEAREST)

        # Create the unified mask on the small matrix
        foreground_mask = ((low_res[:, :, 0] > 0) | 
                           (low_res[:, :, 1] > 0) | 
                           (low_res[:, :, 2] > 0)).astype(np.uint8) * 255

        # MODIFICATION: Background Mask (Everything that is NOT foreground)
        background_mask = cv2.bitwise_not(foreground_mask)

        if not np.any(foreground_mask):
            empty_frame = np.zeros_like(cv_image)
            self.publish_image(self.debug_pub, empty_frame, msg.header.stamp)
            self.publish_image(self.lines_pub, empty_frame, msg.header.stamp)
            return   
        
        # --- STEP 2: HIGH-SPEED CLOSING AND EROSION (CURB) ---
        clean_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, self.kernel_close)
        eroded_foreground = cv2.erode(clean_mask, self.kernel_edge, iterations=1)
        boundary_edges = cv2.bitwise_xor(clean_mask, eroded_foreground) > 0

        # --- STEP 3: FIXED GEOMETRIC CROP OF THE LOWER ZONE ---
        boundary_edges[int(LOW_RES_SIZE * 0.91):, :]  = False

        # Crop for the background
        background_mask[int(LOW_RES_SIZE * 0.91):, :] = False
        background_mask[:int(LOW_RES_SIZE * 0.54), :]  = False

        # BACKGROUND CLEANING: Removes small isolated pixels and "class holes"
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, self.kernel_bg_clean, iterations=2)
        
        # --- STEP 4: OUTPUT IMAGES PREPARATION AND MATHEMATICAL FILTERING ---
        full_overlay_frame = cv_image.copy()
        only_lines_frame = np.zeros_like(cv_image)

        # Isolate color channels on the LOW-RES map
        b_low = low_res[:, :, 0]
        g_low = low_res[:, :, 1]
        r_low = low_res[:, :, 2]

        is_red_low = (r_low > 50) & (r_low > g_low.astype(np.int16) + 20) & (r_low > b_low.astype(np.int16) + 20)
        is_green_low = (g_low > 50) & (g_low > r_low.astype(np.int16) + 20) & (g_low > b_low.astype(np.int16) + 20)

        # Transform into a uint8 binary mask to pass it to OpenCV functions
        mask_red = is_red_low.astype(np.uint8) * 255

        # --- FAST SKELETONIZATION OF THE RED LINE ---
        # This loop erodes the line and extracts its exact centerline (1px thickness)
        skel_red = np.zeros_like(mask_red)
        done = False
        while not done and np.any(mask_red):
            eroded = cv2.erode(mask_red, self.kernel_skel)
            temp = cv2.dilate(eroded, self.kernel_skel)
            temp = cv2.subtract(mask_red, temp)
            skel_red = cv2.bitwise_or(skel_red, temp)
            mask_red = eroded.copy()
            if cv2.countNonZero(mask_red) == 0:
                done = True

        # Extract pixel indices (now the red line extracted from skel_red is 1px wide)
        raw_points_curb = np.argwhere(boundary_edges)
        raw_points_red = np.argwhere(skel_red > 0)
        raw_points_green = np.argwhere(is_green_low)
        raw_points_background = np.argwhere(background_mask > 0)

        # Scale factors for high-resolution geometric conversion
        scale_x = high_w / LOW_RES_SIZE
        scale_y = high_h / LOW_RES_SIZE

        # Internal lambda function to scale points quickly without repeating code and to keep 1px thickness
        # lambda is a "small function"
        # lambda input: output
        scale_points = lambda raw_points: np.stack([raw_points[:, 1] * scale_x, raw_points[:, 0] * scale_y], axis=-1).astype(np.int32) \
                        if len(raw_points) > 0 else np.empty((0, 2), dtype=np.int32)

        pts_curb = scale_points(raw_points_curb)
        pts_red = scale_points(raw_points_red)
        pts_green = scale_points(raw_points_green)
        pts_background = scale_points(raw_points_background) # <-- Points ready in high-resolution pixels

        # To include a part of the background points inside the curb vector
        pts_background_sampled = pts_background[::3]
        pts_curb = np.concatenate((pts_curb, pts_background_sampled), axis=0)

        # --- STEP 5: DRAW ON "ONLY_LINES_FRAME" (ONLY SINGLE PIXELS WITH THICKNESS 1) ---
        # Red Line (Now reduced to 1 pixel)
        if len(pts_red) > 0:
            u_r = np.clip(pts_red[:, 0], 0, high_w - 1)
            v_r = np.clip(pts_red[:, 1], 0, high_h - 1)
            only_lines_frame[v_r, u_r] = [0, 0, 255]

        # Green Line
        if len(pts_green) > 0:
            u_g = np.clip(pts_green[:, 0], 0, high_w - 1)
            v_g = np.clip(pts_green[:, 1], 0, high_h - 1)
            only_lines_frame[v_g, u_g] = [0, 255, 0]

        # Purple Curb (Single pixel)
        if len(pts_curb) > 0:
            pts_curb_sampled = pts_curb[::self.sampling_step]
            u_c = np.clip(pts_curb_sampled[:, 0], 0, high_w - 1)
            v_c = np.clip(pts_curb_sampled[:, 1], 0, high_h - 1)
            only_lines_frame[v_c, u_c] = [255, 0, 255]

            # --- STEP 6: DEBUG OVERLAY (WITH 3x3 SQUARES FOR HUMAN VISUALIZATION) ---
            for ox in [-1, 0, 1]:
                for oy in [-1, 0, 1]:
                    px = np.clip(pts_curb_sampled[:, 0] + ox, 0, high_w - 1)
                    py = np.clip(pts_curb_sampled[:, 1] + oy, 0, high_h - 1)
                    full_overlay_frame[py, px] = [255, 0, 255]

        # --- STEP 7: PARALLEL PUBLISHING ---
        self.publish_image(self.debug_pub, full_overlay_frame, msg.header.stamp)
        self.publish_image(self.lines_pub, only_lines_frame, msg.header.stamp)

    def publish_image(self, publisher, frame, timestamp):
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = timestamp
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = CurbDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()