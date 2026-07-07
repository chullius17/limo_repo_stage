#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading

class CurbDetector(Node):

    def __init__(self):
        super().__init__('curb_detector')
        self.bridge = CvBridge()
        
        # Threading infrastructure for the Producer-Consumer pattern
        self.data_lock = threading.Lock()
        self.latest_msg = None
        self.new_data_available = threading.Event()
        
        # Track publication timing for diagnostics
        self.last_pub_time = None
        
        # Subscription queue size set to 1 to reduce memory pressure
        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/detection/lane_masks/compressed', 
            self.image_callback, 
            1
        )
        
        # Publishers for native 112p output streams
        self.debug_pub = self.create_publisher(CompressedImage, '/detection/curb_points_debug/compressed', 10)
        self.lines_pub = self.create_publisher(Image, '/detection/lines_and_curbs/raw', 10)

        # Spin up the dedicated worker thread to handle the math pipeline
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.get_logger().info("CurbDetector ultra-streamlined worker thread initialized at native 112p.")

    def image_callback(self, msg):
        """
        Producer Callback: Thread-safe injection of incoming frame into the 1-element buffer.
        This unblocks the ROS 2 executor immediately to handle transport protocols.
        """
        with self.data_lock:
            self.latest_msg = msg
        self.new_data_available.set()

    def _worker_loop(self):
        """
        Consumer Loop: Runs independently of the ROS middleware executor.
        Processes only the freshest frame, implementing natural dropping under heavy loads.
        """
        while rclpy.ok():
            if not self.new_data_available.wait(timeout=1.0):
                continue
                
            self.new_data_available.clear()
            
            with self.data_lock:
                msg = self.latest_msg
                self.latest_msg = None
                
            if msg is None:
                continue
                
            self._process_pipeline(msg)

    def _process_pipeline(self, msg):
        """Sequential native resolution processing and rendering engine."""
        # Initialize dictionary with default zeros to prevent any future KeyError on early return
        t = {
            'decomp': 0.0, 'step1_masks': 0.0, 'step2_crop': 0.0, 
            'step3_color_iso': 0.0, 'step4_drawing': 0.0, 'step5_publish': 0.0
        }
        start_total = time.perf_counter()
        
        # Check subscriber counts dynamically to skip redundant processing steps
        has_debug_sub = self.debug_pub.get_subscription_count() > 0
        has_lines_sub = self.lines_pub.get_subscription_count() > 0
        
        # --- DECOMPRESSION ---
        t_start = time.perf_counter()
        try:
            # Native input is already 112x112 from the previous node
            native_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Decompression error in worker thread: {str(e)}")
            return
        t['decomp'] = (time.perf_counter() - t_start) * 1000
        
        native_h, native_w, _ = native_img.shape

        # --- STEP 1: MASK GENERATION (NO RESIZING) ---
        t_start = time.perf_counter()
        foreground_mask = ((native_img[:, :, 0] > 0) | 
                           (native_img[:, :, 1] > 0) | 
                           (native_img[:, :, 2] > 0)).astype(np.uint8) * 255

        background_mask = cv2.bitwise_not(foreground_mask)
        t['step1_masks'] = (time.perf_counter() - t_start) * 1000

        # Early return handling for empty environments
        if not np.any(foreground_mask):
            empty_frame = np.zeros_like(native_img)
            if has_debug_sub:
                self.publish_compressed_image(self.debug_pub, empty_frame, msg.header.stamp)
            if has_lines_sub:
                self.publish_raw_image(self.lines_pub, empty_frame, msg.header.stamp)
            return   

        # --- STEP 2: GEOMETRIC CROP (NO MORPHOLOGY) ---
        t_start = time.perf_counter()
        # Apply region-of-interest vertical cropping directly to raw mask
        background_mask[int(native_h * 0.91):, :] = False
        background_mask[:int(native_h * 0.54), :] = False
        t['step2_crop'] = (time.perf_counter() - t_start) * 1000
        
        # --- STEP 3: DIRECT COLOR ISOLATION (RAW MASKS) ---
        t_start = time.perf_counter()
        b_low = native_img[:, :, 0]
        g_low = native_img[:, :, 1]
        r_low = native_img[:, :, 2]

        # Robust color thresholding to handle potential JPEG compression noise
        is_red_low = (r_low > 50) & (r_low > g_low.astype(np.int16) + 20) & (r_low > b_low.astype(np.int16) + 20)
        is_green_low = (g_low > 50) & (g_low > r_low.astype(np.int16) + 20) & (g_low > b_low.astype(np.int16) + 20)
        t['step3_color_iso'] = (time.perf_counter() - t_start) * 1000

        # --- STEP 4: NATIVE-RESOLUTION RENDERING via DIRECT MASK INDEXING ---
        t_start = time.perf_counter()
        
        # Render the standard analytical lines directly onto a 112p canvas using fast logical indexing
        if has_lines_sub:
            only_lines_frame = np.zeros((native_h, native_w, 3), dtype=np.uint8)
            only_lines_frame[is_red_low] = [0, 0, 255]
            only_lines_frame[is_green_low] = [0, 255, 0]
            only_lines_frame[background_mask > 0] = [255, 0, 255]
        else:
            only_lines_frame = None

        # Render the debug overlay stream directly onto the native 112p canvas without any filters
        if has_debug_sub:
            full_overlay_frame = native_img.copy()
            full_overlay_frame[background_mask > 0] = [255, 0, 255]
        else:
            full_overlay_frame = None
            
        t['step4_drawing'] = (time.perf_counter() - t_start) * 1000

        # --- STEP 5: MULTI-THREADED ROS 2 PUBLISH ---
        t_start = time.perf_counter()
        if has_debug_sub and full_overlay_frame is not None:
            self.publish_compressed_image(self.debug_pub, full_overlay_frame, msg.header.stamp)
        if has_lines_sub and only_lines_frame is not None:
            self.publish_raw_image(self.lines_pub, only_lines_frame, msg.header.stamp)
        t['step5_publish'] = (time.perf_counter() - t_start) * 1000
        
        t['total'] = (time.perf_counter() - start_total) * 1000
        self.log_diagnostics(native_w, native_h, t)

    def publish_raw_image(self, publisher, frame, timestamp):
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = timestamp
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Raw image serialization error: {str(e)}")

    def publish_compressed_image(self, publisher, frame, timestamp):
        try:
            msg = CompressedImage()
            msg.header.stamp = timestamp
            msg.format = "jpeg"
            success, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                msg.data = encoded_img.tobytes()
                publisher.publish(msg)
            else:
                self.get_logger().error("OpenCV JPEG encoding routine failed.")
        except Exception as e:
            self.get_logger().error(f"Compressed image serialization error: {str(e)}")

    def log_diagnostics(self, w, h, t):
        current_time = self.get_clock().now()
        fps = 0.0
        if self.last_pub_time is not None:
            elapsed_sec = (current_time - self.last_pub_time).nanoseconds / 1e9
            if elapsed_sec > 0:
                fps = 1.0 / elapsed_sec
        self.last_pub_time = current_time
        
        self.get_logger().info(
            f"\n"
            f"================ NATIVE 112p ULTRA-STREAMLINED PROFILE ({w}x{h} @ {fps:.1f} REAL FPS) ================\n"
            f"  [Total Worker Callback Latency]  {t['total']:.2f} ms\n"
            f"  -----------------------------------------------------------------\n"
            f"  [Decompression]                  {t['decomp']:.2f} ms\n"
            f"  [Step 1: Mask Extraction]        {t['step1_masks']:.2f} ms\n"
            f"  [Step 2: Geometric Crop]         {t['step2_crop']:.2f} ms\n"
            f"  [Step 3: Color Isolation]        {t['step3_color_iso']:.2f} ms\n"
            f"  [Step 4: Native-Res Draw]        {t['step4_drawing']:.2f} ms\n"
            f"  [Step 5: ROS 2 Publish]          {t['step5_publish']:.2f} ms\n"
            f"========================================================================="
        )

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