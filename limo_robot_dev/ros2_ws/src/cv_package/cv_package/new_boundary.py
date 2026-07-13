#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading
import queue
from turbojpeg import TurboJPEG, TJPF_BGR

class CurbDetector(Node):

    def __init__(self):
        super().__init__('curb_detector')
        self.bridge = CvBridge()
        
        # Initialize TurboJPEG for hardware-accelerated encoding
        self.jpeg_encoder = TurboJPEG()
        
        # Thread-safe single-element buffer for the Producer-Consumer pattern
        self.image_queue = queue.Queue(maxsize=1)
        
        # Diagnostic tracking for cumulative averages
        self.frame_count = 0
        self.t_accumulated = {
            'conversion': 0.0, 'step1_masks': 0.0, 'step2_roi_crop': 0.0, 'step2_opening': 0.0,
            'step3_color_iso': 0.0, 'step4_drawing': 0.0, 'step5_publish': 0.0, 'total': 0.0
        }
        
        # Subscription queue size set to 1 to reduce memory pressure (Raw Image Input)
        self.image_sub = self.create_subscription(
            Image, 
            '/detection/lane_masks/raw', 
            self.image_callback, 
            1
        )
        
        # Symmetric publishers for raw and compressed analytical output streams
        self.debug_pub = self.create_publisher(CompressedImage, '/detection/lines_and_curbs/compressed', 10)
        self.lines_pub = self.create_publisher(Image, '/detection/lines_and_curbs/raw', 10)

        # Small morphological kernel exclusively for background crumb cleaning
        self.kernel_bg_clean = np.ones((3, 3), dtype=np.uint8)

        # Spin up the dedicated worker thread pointing to the consumer loop
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.get_logger().info("CurbDetector worker thread with TurboJPEG Outbound Encoder initialized.")

    def image_callback(self, msg):
        """
        Producer Callback: Quick, non-blocking injection of the raw message into the queue.
        Keeps the ROS executor completely free.
        """
        if self.image_queue.full():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.image_queue.put_nowait(msg)
        except queue.Full:
            pass

    def _worker_loop(self):
        """
        Consumer Loop: Safely fetches the freshest frame from the queue 
        and passes it to the sequential pipeline.
        """
        while rclpy.ok():
            try:
                # Fetch raw message with a timeout to keep the thread responsive to shutdowns
                msg = self.image_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            self._process_pipeline(msg)

    def _process_pipeline(self, msg):
        """Sequential native resolution processing and rendering engine."""
        t = {
            'conversion': 0.0, 'step1_masks': 0.0, 'step2_roi_crop': 0.0, 'step2_opening': 0.0, 
            'step3_color_iso': 0.0, 'step4_drawing': 0.0, 'step5_publish': 0.0, 'total': 0.0
        }
        start_total = time.perf_counter()
        
        # Check subscriber counts dynamically to skip redundant processing steps
        has_debug_sub = self.debug_pub.get_subscription_count() > 0
        has_lines_sub = self.lines_pub.get_subscription_count() > 0
        
        # --- IMAGE CONVERSION (CvBridge Raw-to-CV2) ---
        t_start = time.perf_counter()
        try:
            native_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge conversion error in worker thread: {str(e)}")
            return
        t['conversion'] = (time.perf_counter() - t_start) * 1000
        
        native_h, native_w, _ = native_img.shape

        # --- STEP 1: MASK GENERATION (NO RESIZING) ---
        t_start = time.perf_counter()
        foreground_mask = ((native_img[:, :, 0] > 40) | 
                           (native_img[:, :, 1] > 40) | 
                           (native_img[:, :, 2] > 40)).astype(np.uint8) * 255

        background_mask = cv2.bitwise_not(foreground_mask)
        t['step1_masks'] = (time.perf_counter() - t_start) * 1000

        # Early return handling for empty environments
        if not np.any(foreground_mask):
            empty_frame = np.zeros_like(native_img)
            if has_debug_sub:
                self.publish_compressed_image(self.debug_pub, empty_frame, msg.header.stamp)
            if has_lines_sub:
                self.publish_raw_image(self.lines_pub, empty_frame, msg.header.stamp)
            t['total'] = (time.perf_counter() - start_total) * 1000
            self.log_diagnostics(native_w, native_h, t)
            return   

        # --- STEP 2: ROI-RESTRICTED BACKGROUND CLEANING ---
        t_start = time.perf_counter()
        y_start = int(native_h * 0.54)
        y_end = int(native_h * 0.91)
        roi_slice = background_mask[y_start:y_end, :]
        t_crop_init = (time.perf_counter() - t_start) * 1000

        # Isolated timing for the morphological opening operation
        t_opening_start = time.perf_counter()
        roi_cleaned = cv2.morphologyEx(roi_slice, cv2.MORPH_OPEN, self.kernel_bg_clean, iterations=1)
        t['step2_opening'] = (time.perf_counter() - t_opening_start) * 1000
        
        # Timing for mask reconstruction overhead
        t_recon_start = time.perf_counter()
        background_mask = np.zeros_like(background_mask)
        background_mask[y_start:y_end, :] = roi_cleaned
        t['step2_roi_crop'] = t_crop_init + (time.perf_counter() - t_recon_start) * 1000
        
        # --- STEP 3: DIRECT COLOR ISOLATION (RAW MASKS) ---
        t_start = time.perf_counter()
        b_low = native_img[:, :, 0]
        g_low = native_img[:, :, 1]
        r_low = native_img[:, :, 2]

        # Robust color thresholding to handle potential sensor noise
        is_red_low = (r_low > 150) & (r_low > g_low.astype(np.int16) + 50) & (r_low > b_low.astype(np.int16) + 50)
        is_green_low = (g_low > 150) & (g_low > r_low.astype(np.int16) + 50) & (g_low > b_low.astype(np.int16) + 50)
        t['step3_color_iso'] = (time.perf_counter() - t_start) * 1000

        # --- STEP 4: NATIVE-RESOLUTION RENDERING via DIRECT MASK INDEXING ---
        t_start = time.perf_counter()
        
        # Render analytical lines onto a 112p canvas if any subscriber is active
        if has_lines_sub or has_debug_sub:
            only_lines_frame = np.zeros((native_h, native_w, 3), dtype=np.uint8)
            only_lines_frame[is_red_low] = [0, 0, 255]
            only_lines_frame[is_green_low] = [0, 255, 0]
            only_lines_frame[background_mask > 0] = [255, 0, 255]
        else:
            only_lines_frame = None
            
        t['step4_drawing'] = (time.perf_counter() - t_start) * 1000

        # --- STEP 5: ROS 2 PUBLISH ---
        t_start = time.perf_counter()
        # Publish the exact same analytical output frame in both compressed and raw streams
        if has_debug_sub and only_lines_frame is not None:
            self.publish_compressed_image(self.debug_pub, only_lines_frame, msg.header.stamp)
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
            
            # TurboJPEG direct encoding from BGR numpy array to JPEG byte string
            encoded_img = self.jpeg_encoder.encode(frame, quality=80, pixel_format=TJPF_BGR)
            
            msg.data = encoded_img
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"TurboJPEG compression serialization error: {str(e)}")

    def log_diagnostics(self, w, h, t):
        self.frame_count += 1
        for key in self.t_accumulated:
            self.t_accumulated[key] += t[key]
            
        avg = {key: self.t_accumulated[key] / self.frame_count for key in self.t_accumulated}
        fps = 1000.0 / avg['total'] if avg['total'] > 0 else 0.0
        
        self.get_logger().info(
            f"\n"
            f"================ NATIVE 112p PROFILED WORKER ({w}x{h} @ {fps:.1f} INTERNAL FPS) ================\n"
            f"  Frames processed: {self.frame_count}\n"
            f"  [Total Latency]                 Current: {t['total']:.2f} ms | Avg: {avg['total']:.2f} ms\n"
            f"  -----------------------------------------------------------------\n"
            f"  [CvBridge Conversion]           Current: {t['conversion']:.2f} ms | Avg: {avg['conversion']:.2f} ms\n"
            f"  [Step 1: Mask Extraction]       Current: {t['step1_masks']:.2f} ms | Avg: {avg['step1_masks']:.2f} ms\n"
            f"  [Step 2a: ROI Crop & Setup]     Current: {t['step2_roi_crop']:.2f} ms | Avg: {avg['step2_roi_crop']:.2f} ms\n"
            f"  [Step 2b: Isolated Opening]     Current: {t['step2_opening']:.2f} ms | Avg: {avg['step2_opening']:.2f} ms\n"
            f"  [Step 3: Color Isolation]       Current: {t['step3_color_iso']:.2f} ms | Avg: {avg['step3_color_iso']:.2f} ms\n"
            f"  [Step 4: Native-Res Draw]       Current: {t['step4_drawing']:.2f} ms | Avg: {avg['step4_drawing']:.2f} ms\n"
            f"  [Step 5: ROS 2 Publish]         Current: {t['step5_publish']:.2f} ms | Avg: {avg['step5_publish']:.2f} ms\n"
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