#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage # CHANGE: Added CompressedImage for efficient transport
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule 
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
from collections import deque
import numba as nb
import threading
import queue

# --- OPTIMIZATION #2: FUSED MASK COMPUTATION VIA NUMBA, PARALLELIZED OVER BOXES ---
# Rispetto alla versione originale:
#  - parallel=True + nb.prange sulle box: sfrutta i 4 core Cortex-A57 del Nano
#    invece di girare single-thread.
#  - il loop k interno (32 coeff) e' sostituito da un dot vettorizzato riga per riga
#    (masks_coeffs[i] @ proto[:, y, xa:xb+1]), che numba compila come SIMD invece
#    del loop scalare k=0..31 originale.
@nb.njit(fastmath=True, parallel=True)
def numba_fused_mask_processing(masks_coeffs, proto, class_ids, x1, y1, x2, y2):
    num_boxes = masks_coeffs.shape[0]

    # Un canvas per-box (poi ridotto) evita race condition tra thread paralleli
    # che scriverebbero sullo stesso canvas condiviso.
    canvases = np.full((num_boxes, 4, 112, 112), -10.0, dtype=np.float32)

    for i in nb.prange(num_boxes):
        c_id = class_ids[i]
        if c_id < 0 or c_id > 3:
            continue

        xa, ya, xb, yb = x1[i], y1[i], x2[i], y2[i]
        coeffs_i = masks_coeffs[i]

        for y in range(ya, yb + 1):
            # dot vettorizzato sulla riga invece del loop k manuale
            row_vals = np.zeros(xb - xa + 1, dtype=np.float32)
            for k in range(32):
                ck = coeffs_i[k]
                proto_row = proto[k, y, xa:xb + 1]
                for xi in range(row_vals.shape[0]):
                    row_vals[xi] += ck * proto_row[xi]

            for xi in range(row_vals.shape[0]):
                x = xa + xi
                if row_vals[xi] > canvases[i, c_id, y, x]:
                    canvases[i, c_id, y, x] = row_vals[xi]

    # Riduzione sequenziale (poco costosa) delle canvas per-box in un unico overlay
    low_res_overlay = np.full((112, 112), 4, dtype=np.uint8)
    priorities = [3, 0, 2, 1]

    best_val = np.full((4, 112, 112), -10.0, dtype=np.float32)
    for i in range(num_boxes):
        for c in range(4):
            for y in range(112):
                for x in range(112):
                    v = canvases[i, c, y, x]
                    if v > best_val[c, y, x]:
                        best_val[c, y, x] = v

    for p in priorities:
        for y in range(112):
            for x in range(112):
                if best_val[p, y, x] > 0.0:
                    low_res_overlay[y, x] = p

    return low_res_overlay


class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # CHANGE: Type updated to CompressedImage and topic adjusted to show compression type
        self.mask_pub = self.create_publisher(CompressedImage, '/detection/lane_masks/compressed', 10)
        self.bridge = CvBridge()
        
        # --- OPTIMIZATION #4: QoS "solo ultimo frame" (anti-lag) ---
        # depth=1 + KEEP_LAST + BEST_EFFORT: se il callback e' piu' lento del
        # publisher della camera, DDS scarta i frame vecchi in coda invece di
        # accumularli. Il nodo elabora sempre il frame piu' recente disponibile,
        # mai un backlog: si perdono frame, ma non si accumula latenza.
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

        # --- ULTRA-GRANULAR TELEMETRY STORAGE ---
        self.window_size = 100
        self.telemetry_stats = {
            '1_cv_bridge': deque(maxlen=self.window_size),
            '2_preprocessing_cpu': deque(maxlen=self.window_size),
            '3_gpu_h2d_raw': deque(maxlen=self.window_size),
            '3b_gpu_preprocessing': deque(maxlen=self.window_size),
            '4_gpu_inference': deque(maxlen=self.window_size),
            '5_gpu_d2h': deque(maxlen=self.window_size),
            '6_post_filter': deque(maxlen=self.window_size),
            '7_post_nms': deque(maxlen=self.window_size),
            '8_post_matmul': deque(maxlen=self.window_size),
            '9_post_canvas': deque(maxlen=self.window_size),
            '10_post_upsample': deque(maxlen=self.window_size),
            '11_ros_publish_enqueue': deque(maxlen=self.window_size),
            'total_pipeline': deque(maxlen=self.window_size),
            'async_encode_publish': deque(maxlen=self.window_size),
        }

        # --- OPTIMIZATION #1: ASYNC ENCODE+PUBLISH WORKER ---
        # JPEG encode a piena risoluzione e' CPU-bound e non richiede la GPU.
        # Disaccoppiandolo dal thread di callback il collo di bottiglia della
        # pipeline diventa il singolo stage piu' lento (inferenza, ~35ms) invece
        # della somma di tutti gli stage (~104ms). maxsize=1 con drop-latest
        # garantisce che pubblichiamo sempre il frame piu' recente disponibile,
        # sacrificando frame vecchi piuttosto che accumulare latenza.
        self.pub_queue = queue.Queue(maxsize=1)
        self.pub_thread = threading.Thread(target=self._publish_worker, daemon=True)
        self.pub_thread.start()

        # --- CUDA KERNEL FOR GPU PREPROCESSING ---
        cuda_code = """
        __global__ void preprocess_kernel(const unsigned char *bgr_in, float *rgb_out, int H, int W) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_elements = H * W * 3;
            
            if (idx < total_elements) {
                int pixel_idx = idx / 3;
                int c = idx % 3;
                int row = pixel_idx / W;
                int col = pixel_idx % W;
                int target_c = 2 - c; 
                int dst_idx = target_c * (H * W) + row * W + col;
                rgb_out[dst_idx] = (float)bgr_in[idx] / 255.0f;
            }
        }
        """
        self.mod = SourceModule(cuda_code)
        self.gpu_preprocess_kernel = self.mod.get_function("preprocess_kernel")

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
        
        self.get_logger().info("GPU Accelerated 4-Class Segmenter Node started!")

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
                    'shape': shape
                })
                
        self.raw_input_size = 448 * 448 * 3
        self.device_raw_input = cuda.mem_alloc(self.raw_input_size)

    def _publish_worker(self):
        """Thread separato: encode JPEG + publish, disaccoppiato dal callback GPU."""
        while rclpy.ok():
            item = self.pub_queue.get()
            if item is None:
                continue
            color_mask_high, stamp = item

            t0 = time.perf_counter()
            try:
                success, encoded_img = cv2.imencode(
                    '.jpg', color_mask_high, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if success:
                    ros_mask_msg = CompressedImage()
                    ros_mask_msg.header.stamp = stamp
                    ros_mask_msg.format = "jpeg"
                    ros_mask_msg.data = encoded_img.tobytes()
                    self.mask_pub.publish(ros_mask_msg)
                else:
                    self.get_logger().error("Failed to encode BGR mask to JPEG format")
            except Exception as e:
                self.get_logger().error(f"Failed to publish compressed output image: {str(e)}")
            t1 = time.perf_counter()
            self.telemetry_stats['async_encode_publish'].append((t1 - t0) * 1000)

    def image_callback(self, msg):
        if self.engine is None:
            return
        
        t_start = time.perf_counter()

        # --- 1. CV_BRIDGE CONVERSION ---
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return
        t_cv_bridge = time.perf_counter()
        
        # --- 2. PREPROCESSING (CPU Phase: Only Resize) ---
        h, w, _ = cv_image.shape
        resized_img = cv2.resize(cv_image, (448, 448), interpolation=cv2.INTER_LINEAR)
        t_preproc_cpu = time.perf_counter()

        # --- 3. GPU HOST TO DEVICE TRANSFER (Raw uint8) ---
        try:
            cuda.memcpy_htod_async(self.device_raw_input, resized_img.tobytes(), self.stream)
            self.stream.synchronize()
            t_h2d_raw = time.perf_counter()
            
            # --- 3b. GPU PREPROCESSING KERNEL ---
            block_size = 256
            grid_size = int((self.raw_input_size + block_size - 1) / block_size)
            
            self.gpu_preprocess_kernel(
                self.device_raw_input, 
                self.device_input, 
                np.int32(448), 
                np.int32(448),
                block=(block_size, 1, 1),
                grid=(grid_size, 1),
                stream=self.stream
            )
            self.stream.synchronize()
            t_preproc_gpu = time.perf_counter()
            
            # --- 4. TENSORRT INFERENCE ---
            self.trt_context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            self.stream.synchronize()
            t_infer = time.perf_counter()
            
            # --- 5. GPU DEVICE TO HOST TRANSFER ---
            for out in self.output_info:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
            t_d2h = time.perf_counter()
            
            out1 = self.output_info[0]['host'].reshape(self.output_info[0]['shape'])
            out2 = self.output_info[1]['host'].reshape(self.output_info[1]['shape'])
            
            if len(self.output_info[0]['shape']) == 4:
                proto = np.squeeze(out1)
                predictions = np.squeeze(out2)
            else:
                predictions = np.squeeze(out1)
                proto = np.squeeze(out2)

        except Exception as e:
            self.get_logger().error(f"GPU Preproc/Inference Step failed: {str(e)}")
            return
        
        dt_filter = dt_nms = dt_matmul = dt_canvas = dt_upsample = 0.0
        mask_overlay = np.full((112, 112), 4, dtype=np.uint8)

        t_post_start = time.perf_counter()

        if predictions.ndim == 2:
            # --- OPTIMIZATION #3: FILTER PRIMA DELL'ARGMAX COMPLETO ---
            # Originale: argmax + fancy-index su TUTTE le ~8400 predizioni prima
            # del threshold. Ora: max() su tutte le righe (piu' economico di
            # argmax), si filtra per soglia, e l'argmax vero e proprio (piu'
            # costoso) si calcola solo sui sopravvissuti al threshold.
            t0 = time.perf_counter()

            # NOTE: see effects on neural network
            predictions = predictions.T
            t1 = time.perf_counter()

            num_classes = 4
            scores = predictions[:, 4:4+num_classes]
            t2 = time.perf_counter()

            max_conf = scores.max(axis=1)
            t3 = time.perf_counter()

            mask_threshold = max_conf > 0.3
            t4 = time.perf_counter()

            filtered_preds = predictions[mask_threshold]
            t5 = time.perf_counter()

            filtered_scores = scores[mask_threshold]
            t6 = time.perf_counter()

            filtered_class_ids = np.argmax(filtered_scores, axis=1)
            t7 = time.perf_counter()

            filtered_confs = max_conf[mask_threshold]
            t8 = time.perf_counter()
            t_filter = time.perf_counter()

            print("transpose :", (t1-t0)*1000)
            print("scores    :", (t2-t1)*1000)
            print("max       :", (t3-t2)*1000)
            print("threshold :", (t4-t3)*1000)
            print("pred copy :", (t5-t4)*1000)
            print("score copy:", (t6-t5)*1000)
            print("argmax    :", (t7-t6)*1000)
            print("conf copy :", (t8-t7)*1000)

            dt_filter = (t_filter - t_post_start) * 1000
            
            if len(filtered_preds) > 0:
                # --- 7. POST-PROC: NMS BOXES ---
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
                    
                    # Compute specialized box bounds scaled down to the 112x112 layout
                    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = np.clip((cx - bw / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                    y1 = np.clip((cy - bh / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                    x2 = np.clip((cx + bw / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                    y2 = np.clip((cy + bh / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                    
                    # --- 8 & 9. FUSED NUMBA POST-PROCESSING (parallel over boxes) ---
                    masks_coeffs = filtered_preds[:, 4+num_classes : 4+num_classes+32]
                    
                    mask_overlay = numba_fused_mask_processing(
                        masks_coeffs, 
                        proto, 
                        filtered_class_ids, 
                        x1, y1, x2, y2
                    )
                    
                    t_canvas = time.perf_counter()
                    dt_matmul = 0.0 
                    dt_canvas = (t_canvas - t_nms) * 1000
        
        # --- 10. POST-PROC: COLOR GENERATION & ORIGINAL-RESOLUTION UPSAMPLE ---
        t_upsample_start = time.perf_counter()
        
        # Color map dictionary translated exactly to a fast BGR matrix lookup table
        color_lut = np.array([
            [0, 255, 0],    # Class 0: Dashed Lines -> Green
            [255, 0, 0],    # Class 1: Parking Lots -> Blue
            [0, 0, 255],    # Class 2: Solid Lines  -> Red
            [80, 80, 80],   # Class 3: Road Surface -> Gray
            [0, 0, 0]       # Class 4: Background   -> Black
        ], dtype=np.uint8)
        
        # Convert index map to BGR low-res image
        color_mask_low = color_lut[mask_overlay]
        
        # Scale up back to matching original camera resolution (w, h)
        color_mask_high = cv2.resize(color_mask_low, (w, h), interpolation=cv2.INTER_NEAREST)
        
        t_upsample_end = time.perf_counter()
        dt_upsample = (t_upsample_end - t_upsample_start) * 1000

        # --- 11. ENQUEUE FOR ASYNC ENCODE+PUBLISH (OPTIMIZATION #1) ---
        # Non incodifichiamo/pubblichiamo piu' qui in modo sincrono: mettiamo il
        # frame in coda (maxsize=1, drop-latest) e il thread worker si occupa di
        # encode JPEG + publish in parallelo rispetto al prossimo ciclo GPU.
        t_pub_start = time.perf_counter()
        if self.pub_queue.full():
            try:
                self.pub_queue.get_nowait()  # scarta il frame vecchio non ancora pubblicato
            except queue.Empty:
                pass
        try:
            self.pub_queue.put_nowait((color_mask_high, msg.header.stamp))
        except queue.Full:
            pass
        t_pub_end = time.perf_counter()
        dt_pub = (t_pub_end - t_pub_start) * 1000

        t_end = time.perf_counter()

        # --- METRICS CALCULATIONS ---
        dt_cv_bridge      = (t_cv_bridge - t_start) * 1000
        dt_preproc_cpu    = (t_preproc_cpu - t_cv_bridge) * 1000
        dt_h2d_raw        = (t_h2d_raw - t_preproc_cpu) * 1000
        dt_preproc_gpu    = (t_preproc_gpu - t_h2d_raw) * 1000
        dt_inference      = (t_infer - t_preproc_gpu) * 1000
        dt_d2h            = (t_d2h - t_infer) * 1000
        dt_total_pipe     = (t_end - t_start) * 1000

        self.telemetry_stats['1_cv_bridge'].append(dt_cv_bridge)
        self.telemetry_stats['2_preprocessing_cpu'].append(dt_preproc_cpu)
        self.telemetry_stats['3_gpu_h2d_raw'].append(dt_h2d_raw)
        self.telemetry_stats['3b_gpu_preprocessing'].append(dt_preproc_gpu)
        self.telemetry_stats['4_gpu_inference'].append(dt_inference)
        self.telemetry_stats['5_gpu_d2h'].append(dt_d2h)
        self.telemetry_stats['6_post_filter'].append(dt_filter)
        self.telemetry_stats['7_post_nms'].append(dt_nms)
        self.telemetry_stats['8_post_matmul'].append(dt_matmul)
        self.telemetry_stats['9_post_canvas'].append(dt_canvas)
        self.telemetry_stats['10_post_upsample'].append(dt_upsample)
        self.telemetry_stats['11_ros_publish_enqueue'].append(dt_pub)
        self.telemetry_stats['total_pipeline'].append(dt_total_pipe)

        avg_pipeline = np.mean(self.telemetry_stats['total_pipeline'])
        fps = 1000.0 / avg_pipeline if avg_pipeline > 0 else 0.0
        avg_async_pub = (
            np.mean(self.telemetry_stats['async_encode_publish'])
            if len(self.telemetry_stats['async_encode_publish']) > 0 else 0.0
        )

        self.get_logger().info(
            f"\n"
            f"====== ULTRA-GRANULAR BREAKDOWN REPORT (COMPRESSED HIGH-RES) ======\n"
            f"  1. CvBridge Image Convert:  {dt_cv_bridge:.2f} ms  (Avg: {np.mean(self.telemetry_stats['1_cv_bridge']):.2f} ms)\n"
            f"  2. Preprocessing CPU (Resize): {dt_preproc_cpu:.2f} ms  (Avg: {np.mean(self.telemetry_stats['2_preprocessing_cpu']):.2f} ms)\n"
            f"  3. Cuda H2D Copy (Raw uint8): {dt_h2d_raw:.2f} ms  (Avg: {np.mean(self.telemetry_stats['3_gpu_h2d_raw']):.2f} ms)\n"
            f"  3b. CUDA Preprocess Kernel:   {dt_preproc_gpu:.2f} ms  (Avg: {np.mean(self.telemetry_stats['3b_gpu_preprocessing']):.2f} ms)\n"
            f"  4. TRT GPU Core Inference:   {dt_inference:.2f} ms  (Avg: {np.mean(self.telemetry_stats['4_gpu_inference']):.2f} ms)\n"
            f"  5. Cuda D2H Copy:            {dt_d2h:.2f} ms  (Avg: {np.mean(self.telemetry_stats['5_gpu_d2h']):.2f} ms)\n"
            f"  [Post-Processing Sub-steps]:\n"
            f"     --> 6. Filter & Thresh:   {dt_filter:.2f} ms  (Avg: {np.mean(self.telemetry_stats['6_post_filter']):.2f} ms)\n"
            f"     --> 7. OpenCV NMSBoxes:   {dt_nms:.2f} ms  (Avg: {np.mean(self.telemetry_stats['7_post_nms']):.2f} ms)\n"
            f"     --> 8 & 9. Numba Fused:   {dt_canvas:.2f} ms  (Avg: {np.mean(self.telemetry_stats['9_post_canvas']):.2f} ms)\n"
            f"     --> 10. HR Upsample (LUT): {dt_upsample:.2f} ms  (Avg: {np.mean(self.telemetry_stats['10_post_upsample']):.2f} ms)\n"
            f"  11. Publish Enqueue (async): {dt_pub:.2f} ms  (Avg: {np.mean(self.telemetry_stats['11_ros_publish_enqueue']):.2f} ms)\n"
            f"  [Async Worker] Encode+Pub:   (Avg: {avg_async_pub:.2f} ms, non sommato al totale)\n"
            f"-----------------------------------------\n"
            f"  TOTAL END-TO-END PIPELINE:   {dt_total_pipe:.2f} ms  (Avg: {avg_pipeline:.2f} ms) | Real FPS: {fps:.1f}\n"
            f"========================================="
        )

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Lane Segmenter Node.")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()