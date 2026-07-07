#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This initializes the CUDA context on the main thread
from pycuda.compiler import SourceModule
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
from collections import deque
import threading
import queue
from turbojpeg import TurboJPEG, TJPF_BGR  # Changed to TJPF_BGR to fix color channel swapping


class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # Capture the active CUDA context from the main thread
        self.cuda_context = cuda.Context.get_current()

        self.mask_pub = self.create_publisher(CompressedImage, '/detection/lane_masks/compressed', 10)
        self.bridge = CvBridge()
        self.jpeg_encoder = TurboJPEG()

        latest_frame_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            latest_frame_qos
        )

        self.window_size = 100
        self.telemetry_stats = {
            '1_cv_bridge': deque(maxlen=self.window_size),
            '4_gpu_inference': deque(maxlen=self.window_size),
            '6_post_filter': deque(maxlen=self.window_size),
            '7_post_nms': deque(maxlen=self.window_size),
            '9_post_canvas': deque(maxlen=self.window_size),
            '11_ros_publish_enqueue': deque(maxlen=self.window_size),
            'total_pipeline': deque(maxlen=self.window_size),
            'async_encode_publish': deque(maxlen=self.window_size),
        }

        self.inference_queue = queue.Queue(maxsize=1)
        self.pub_queue = queue.Queue(maxsize=1)

        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.pub_thread = threading.Thread(target=self._publish_worker, daemon=True)
        self.inference_thread.start()
        self.pub_thread.start()

        # Unified CUDA Source Module featuring Preprocessing and Mask Decoding
        cuda_code = """
        __global__ void resize_preprocess_kernel(
            const unsigned char *bgr_in, float *rgb_out,
            int H_in, int W_in, int H_out, int W_out)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total = H_out * W_out;
            if (idx >= total) return;

            int row = idx / W_out;
            int col = idx % W_out;

            float scale_y = (float)H_in / H_out;
            float scale_x = (float)W_in / W_out;
            float sy = (row + 0.5f) * scale_y - 0.5f;
            float sx = (col + 0.5f) * scale_x - 0.5f;

            int y0 = (int)floorf(sy);
            int x0 = (int)floorf(sx);
            if (y0 < 0) y0 = 0;
            if (y0 > H_in - 1) y0 = H_in - 1;
            if (x0 < 0) x0 = 0;
            if (x0 > W_in - 1) x0 = W_in - 1;
            int y1 = y0 + 1; if (y1 > H_in - 1) y1 = H_in - 1;
            int x1 = x0 + 1; if (x1 > W_in - 1) x1 = W_in - 1;

            float wy = sy - y0; if (wy < 0.0f) wy = 0.0f; if (wy > 1.0f) wy = 1.0f;
            float wx = sx - x0; if (wx < 0.0f) wx = 0.0f; if (wx > 1.0f) wx = 1.0f;

            #pragma unroll
            for (int c = 0; c < 3; c++) {
                float v00 = bgr_in[(y0 * W_in + x0) * 3 + c];
                float v01 = bgr_in[(y0 * W_in + x1) * 3 + c];
                float v10 = bgr_in[(y1 * W_in + x0) * 3 + c];
                float v11 = bgr_in[(y1 * W_in + x1) * 3 + c];

                float top = v00 + wx * (v01 - v00);
                float bot = v10 + wx * (v11 - v10);
                float val = (top + wy * (bot - top)) / 255.0f;

                int target_c = 2 - c;
                rgb_out[target_c * (H_out * W_out) + row * W_out + col] = val;
            }
        }

        __global__ void decode_masks_gpu_kernel(
            const float *proto,          // (32, 112, 112) -> CHW format
            const float *coeffs,         // (num_boxes, 32)
            const int *class_ids,        // (num_boxes)
            const int *x1, const int *y1, const int *x2, const int *y2,
            int num_boxes,
            unsigned char *mask_out)     // (112, 112) output buffer
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_pixels = 112 * 112;
            if (idx >= total_pixels) return;

            int row = idx / 112;
            int col = idx % 112;

            // Tracks the highest score for each class locally
            float best_val[4] = {-10.0f, -10.0f, -10.0f, -10.0f};

            for (int i = 0; i < num_boxes; i++) {
                int c = class_ids[i];
                if (c < 0 || c > 3) continue;

                // Geometric boundary check replacing python slices
                if (row >= y1[i] && row <= y2[i] && col >= x1[i] && col <= x2[i]) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int ch = 0; ch < 32; ch++) {
                        sum += coeffs[i * 32 + ch] * proto[ch * total_pixels + idx];
                    }
                    if (sum > best_val[c]) {
                        best_val[c] = sum;
                    }
                }
            }

            // Apply priority sorting (3, 0, 2, 1) to settle conflicts on-chip
            unsigned char final_class = 4;
            int priorities[4] = {3, 0, 2, 1};
            for (int k = 0; k < 4; k++) {
                int p = priorities[k];
                if (best_val[p] > 0.0f) {
                    final_class = p;
                }
            }

            mask_out[idx] = final_class;
        }
        """
        self.mod = SourceModule(cuda_code)
        self.gpu_preprocess_kernel = self.mod.get_function("resize_preprocess_kernel")
        self.gpu_decode_kernel = self.mod.get_function("decode_masks_gpu_kernel")

        try:
            package_share_dir = get_package_share_directory('cv_package')
        except Exception:
            package_share_dir = os.path.dirname(os.path.realpath(__file__))

        default_model_path = os.path.join(package_share_dir, 'best.engine')
        self.declare_parameter('model_path', default_model_path)
        self.model_path = self.get_parameter('model_path').value

        self.get_logger().info(f'Initializing TensorRT Segmentation with model: {self.model_path}')

        try:
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(self.model_path, 'rb') as f:
                self.runtime = trt.Runtime(self.trt_logger)
                self.engine = self.runtime.deserialize_cuda_engine(f.read())

            self.trt_context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            self.allocate_buffers()
            self.get_logger().info("TensorRT engine and GPU contexts successfully created!")
        except Exception as e:
            self.get_logger().error(f"CRITICAL: Failed to load TensorRT Engine: {str(e)}")
            self.engine = None
            return

    def allocate_buffers(self):
        self.bindings = []
        self.output_info = []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.host_input = host_mem
                self.device_input = device_mem
            else:
                self.output_info.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'is_proto': (len(shape) == 4 or 112 in shape)
                })

        self.device_raw_input = None
        self.raw_input_shape = None
        self.host_raw_pinned = None

        # Pre-allocate static GPU execution contexts for variables after NMS
        self.MAX_BOXES = 64
        self.gpu_coeffs = cuda.mem_alloc(self.MAX_BOXES * 32 * 4)
        self.gpu_class_ids = cuda.mem_alloc(self.MAX_BOXES * 4)
        self.gpu_x1 = cuda.mem_alloc(self.MAX_BOXES * 4)
        self.gpu_y1 = cuda.mem_alloc(self.MAX_BOXES * 4)
        self.gpu_x2 = cuda.mem_alloc(self.MAX_BOXES * 4)
        self.gpu_y2 = cuda.mem_alloc(self.MAX_BOXES * 4)
        
        # Pinned 12KB host container for fast asynchronous PCIe download
        self.gpu_mask_out = cuda.mem_alloc(112 * 112)
        self.host_mask_out = cuda.pagelocked_empty((112, 112), dtype=np.uint8)

    def image_callback(self, msg):
        t_cv_bridge_start = time.perf_counter()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return
            
        cv_time = (time.perf_counter() - t_cv_bridge_start) * 1000
        self.telemetry_stats['1_cv_bridge'].append(cv_time)

        if self.inference_queue.full():
            try:
                self.inference_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.inference_queue.put_nowait((cv_image, msg.header.stamp))
        except queue.Full:
            pass

    def _inference_worker(self):
        self.cuda_context.push()

        while rclpy.ok():
            try:
                item = self.inference_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            cv_image, stamp = item
            self._process_frame(cv_image, stamp)

        self.cuda_context.pop()

    def _process_frame(self, cv_image, stamp):
        t_start = time.perf_counter()
        h, w, _ = cv_image.shape

        try:
            if self.raw_input_shape != (h, w):
                if self.device_raw_input is not None:
                    self.device_raw_input.free()
                self.device_raw_input = cuda.mem_alloc(h * w * 3)
                self.raw_input_shape = (h, w)
                self.host_raw_pinned = cuda.pagelocked_empty((h, w, 3), dtype=np.uint8)

            np.copyto(self.host_raw_pinned, cv_image)
            cuda.memcpy_htod_async(self.device_raw_input, self.host_raw_pinned, self.stream)

            out_h, out_w = 448, 448
            out_total = out_h * out_w
            block_size = 256
            grid_size = int((out_total + block_size - 1) / block_size)

            self.gpu_preprocess_kernel(
                self.device_raw_input,
                self.device_input,
                np.int32(h),
                np.int32(w),
                np.int32(out_h),
                np.int32(out_w),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
                stream=self.stream
            )

            self.trt_context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # CRITICAL OPTIMIZATION: Extract structural layouts and leave Proto inside the GPU VRAM
            proto_info = None
            pred_info = None
            for out in self.output_info:
                if out['is_proto']:
                    proto_info = out
                else:
                    pred_info = out
                    # Download only the bounding boxes predictions to Host
                    cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

            self.stream.synchronize()
            t_gpu_done = time.perf_counter()

            predictions = np.squeeze(pred_info['host'].reshape(pred_info['shape']))

        except Exception as e:
            self.get_logger().error(f"GPU Pipeline failed: {str(e)}")
            return

        dt_filter = dt_nms = dt_canvas = 0.0
        mask_overlay = np.full((112, 112), 4, dtype=np.uint8)

        t_post_start = time.perf_counter()

        if predictions.ndim == 2:
            predictions = predictions.T
            num_classes = 4
            scores = predictions[:, 4:4 + num_classes]
            max_conf = scores.max(axis=1)
            mask_threshold = max_conf > 0.3

            filtered_preds = predictions[mask_threshold]
            filtered_scores = scores[mask_threshold]
            filtered_class_ids = np.argmax(filtered_scores, axis=1)
            filtered_confs = max_conf[mask_threshold]
            t_filter = time.perf_counter()
            dt_filter = (t_filter - t_post_start) * 1000

            if len(filtered_preds) > 0:
                boxes = filtered_preds[:, 0:4]
                cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                xmin = (cx - bw / 2).astype(np.int32)
                ymin = (cy - bh / 2).astype(np.int32)
                bw_int = bw.astype(np.int32)
                bh_int = bh.astype(np.int32)

                nms_boxes = np.stack([xmin, ymin, bw_int, bh_int], axis=1).tolist()
                indices = cv2.dnn.NMSBoxes(nms_boxes, [float(c) for c in filtered_confs], score_threshold=0.3, nms_threshold=0.45)
                t_nms = time.perf_counter()
                dt_nms = (t_nms - t_filter) * 1000

                if len(indices) > 0:
                    indices = indices.flatten()
                    filtered_preds = filtered_preds[indices]
                    filtered_class_ids = filtered_class_ids[indices]
                    boxes = boxes[indices]

                    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = np.clip((cx - bw / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                    y1 = np.clip((cy - bh / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                    x2 = np.clip((cx + bw / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                    y2 = np.clip((cy + bh / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)

                    masks_coeffs = filtered_preds[:, 4 + num_classes: 4 + num_classes + 32]

                    # --- GPU MASK DECODING ROUTINE ---
                    num_boxes = min(len(filtered_preds), self.MAX_BOXES)
                    
                    # Upload small post-processed configuration vectors back to GPU
                    cuda.memcpy_htod_async(self.gpu_coeffs, masks_coeffs[:num_boxes].astype(np.float32), self.stream)
                    cuda.memcpy_htod_async(self.gpu_class_ids, filtered_class_ids[:num_boxes].astype(np.int32), self.stream)
                    cuda.memcpy_htod_async(self.gpu_x1, x1[:num_boxes].astype(np.int32), self.stream)
                    cuda.memcpy_htod_async(self.gpu_y1, y1[:num_boxes].astype(np.int32), self.stream)
                    cuda.memcpy_htod_async(self.gpu_x2, x2[:num_boxes].astype(np.int32), self.stream)
                    cuda.memcpy_htod_async(self.gpu_y2, y2[:num_boxes].astype(np.int32), self.stream)

                    # Trigger parallel decoding matrix multiplication inside GPU
                    dec_block = 256
                    dec_grid = int((112 * 112 + dec_block - 1) / dec_block)
                    self.gpu_decode_kernel(
                        proto_info['device'],
                        self.gpu_coeffs,
                        self.gpu_class_ids,
                        self.gpu_x1, self.gpu_y1, self.gpu_x2, self.gpu_y2,
                        np.int32(num_boxes),
                        self.gpu_mask_out,
                        block=(dec_block, 1, 1),
                        grid=(dec_grid, 1),
                        stream=self.stream
                    )

                    # Async download of the tiny 12KB matrix layout
                    cuda.memcpy_dtoh_async(self.host_mask_out, self.gpu_mask_out, self.stream)
                    self.stream.synchronize()
                    mask_overlay = self.host_mask_out

                    t_canvas = time.perf_counter()
                    dt_canvas = (t_canvas - t_nms) * 1000

        t_pub_start = time.perf_counter()
        
        if self.pub_queue.full():
            try:
                self.pub_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.pub_queue.put_nowait((mask_overlay, stamp))
        except queue.Full:
            pass
            
        t_pub_end = time.perf_counter()
        dt_pub = (t_pub_end - t_pub_start) * 1000

        t_end = time.perf_counter()

        dt_gpu_pipeline = (t_gpu_done - t_start) * 1000
        dt_total_pipe = (t_end - t_start) * 1000

        self.telemetry_stats['4_gpu_inference'].append(dt_gpu_pipeline)
        self.telemetry_stats['6_post_filter'].append(dt_filter)
        self.telemetry_stats['7_post_nms'].append(dt_nms)
        self.telemetry_stats['9_post_canvas'].append(dt_canvas)
        self.telemetry_stats['11_ros_publish_enqueue'].append(dt_pub)
        self.telemetry_stats['total_pipeline'].append(dt_total_pipe)

        avg_pipeline = np.mean(self.telemetry_stats['total_pipeline'])
        fps = 1000.0 / avg_pipeline if avg_pipeline > 0 else 0.0
        avg_async_pub = np.mean(self.telemetry_stats['async_encode_publish']) if len(self.telemetry_stats['async_encode_publish']) > 0 else 0.0

        self.get_logger().info(
            f"\n"
            f"====== PERFORMANCE REPORT (GPU ON-CHIP DECODING) ======\n"
            f"  OUTPUT RESOLUTION:           112x112 (NATIVE INTERLEAVED)\n"
            f"  1. Total Asynchronous GPU:   {dt_gpu_pipeline:.2f} ms\n"
            f"  [Post-Processing Sub-steps]:\n"
            f"     --> Filter (conf thresh):  {dt_filter:.2f} ms\n"
            f"     --> NMS:                   {dt_nms:.2f} ms\n"
            f"     --> Mask Decode (CUDA KNL): {dt_canvas:.2f} ms\n"
            f"     --> Publish Enqueue:       {dt_pub:.2f} ms\n"
            f"  [Async Worker] LUT + TurboJPEG (112p): Avg: {avg_async_pub:.2f} ms\n"
            f"-----------------------------------------\n"
            f"  TOTAL INFERENCE TIME:        {dt_total_pipe:.2f} ms | INTERNAL FPS: {fps:.1f}\n"
            f"========================================="
        )

    def _publish_worker(self):
        color_lut = np.array([
            [0, 255, 0],    # Class 0: Dashed Lines (Green: B=0, G=255, R=0)
            [255, 0, 0],    # Class 1: Parking Lots (Blue in BGR: B=255, G=0, R=0)
            [0, 0, 255],    # Class 2: Solid Lines (Red in BGR: B=0, G=0, R=255)
            [80, 80, 80],   # Class 3: Road Surface
            [0, 0, 0]       # Class 4: Background
        ], dtype=np.uint8)

        while rclpy.ok():
            try:
                item = self.pub_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            mask_overlay, stamp = item

            t0 = time.perf_counter()
            try:
                color_frame = color_lut[mask_overlay]

                encoded_img = self.jpeg_encoder.encode(
                    color_frame,
                    pixel_format=TJPF_BGR,
                    quality=80
                )

                ros_mask_msg = CompressedImage()
                ros_mask_msg.header.stamp = stamp
                ros_mask_msg.format = "jpeg"
                ros_mask_msg.data = encoded_img
                self.mask_pub.publish(ros_mask_msg)

            except Exception as e:
                self.get_logger().error(f"Failed to publish native-res 112p compressed image: {str(e)}")

            t1 = time.perf_counter()
            self.telemetry_stats['async_encode_publish'].append((t1 - t0) * 1000)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()