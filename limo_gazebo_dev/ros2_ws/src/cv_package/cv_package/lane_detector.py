#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import onnxruntime as ort
from ament_index_python.packages import get_package_share_directory

class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        self.bridge = CvBridge()
        
        # Subscriber to the Limo's camera
        self.rgb_sub = self.create_subscription(Image, '/limo/color/image_raw', self.image_callback, 10)

        # --- PATH RESOLUTION ---
        try:
            package_share_dir = get_package_share_directory('cv_package')
        except Exception:
            package_share_dir = os.path.dirname(os.path.realpath(__file__))

        default_model_path = os.path.join(package_share_dir, 'best.onnx')
        self.declare_parameter('model_path', default_model_path)
        self.model_path = self.get_parameter('model_path').value

        self.get_logger().info(f'Initializing ONNX Runtime Segmentation with model: {self.model_path}')
        
        # --- LOAD ONNX MODEL VIA ONNXRUNTIME ---
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self.ort_session = ort.InferenceSession(self.model_path, sess_options=opts)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.get_logger().info("ONNX Runtime session successfully created!")
        except Exception as e:
            self.get_logger().error(f"CRITICAL: Failed to load ONNX model via ONNXRuntime: {str(e)}")
            self.ort_session = None
            return
        
        # Publisher for the results
        self.image_pub = self.create_publisher(Image, '/detection/lane_overlay', 10)
        self.mask_pub = self.create_publisher(Image, '/detection/lane_masks', 10)
        self.get_logger().info("Clean & Fast 4-Class Segmenter Node started!")

    def image_callback(self, msg):
        if self.ort_session is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return
        
        h, w, _ = cv_image.shape

        # Preprocessing
        input_img = cv2.resize(cv_image, (448, 448))
        input_img = input_img[:, :, ::-1].transpose(2, 0, 1)   
        input_tensor = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0     # (1, 3, 448, 448)

        try:
            # Inference
            outputs = self.ort_session.run(None, {self.input_name: input_tensor})
            predictions = np.squeeze(outputs[0])  # Shape depends on number of classes
            proto = np.squeeze(outputs[1])        # [32, 112, 112] = 32 prototype masks in a space 112x112
        except Exception as e:
            self.get_logger().error(f"Inference failed: {str(e)}")
            return
        
        mask_overlay = np.zeros_like(cv_image)
        
        if predictions.ndim == 2:
            predictions = predictions.T   # Works for this model
            
            # --- UPDATED FOR 4 CLASSES ---
            # Columns 0-3: Box coordinates (cx, cy, bw, bh)
            # Columns 4-7: Class scores (4 classes total)
            # Columns 8-39: Mask coefficients (32 prototypes)
            num_classes = 4
            scores = predictions[:, 4:4+num_classes]
            class_ids = np.argmax(scores, axis=1)
            confidences = scores[np.arange(len(predictions)), class_ids]
            
            mask_threshold = confidences > 0.1
            filtered_preds = predictions[mask_threshold]
            filtered_class_ids = class_ids[mask_threshold]
            
            if len(filtered_preds) > 0:
                boxes = filtered_preds[:, 0:4]
                
                # Dynamic mask coefficient slicing based on the 4 classes
                masks_coeffs = filtered_preds[:, 4+num_classes : 4+num_classes+32]
                proto_reshaped = proto.reshape(32, -1)
                
                # Linear combination of all masks [N, 112, 112]
                raw_masks = np.matmul(masks_coeffs, proto_reshaped).reshape(-1, 112, 112)
                
                # Vectorized coordinate mapping back to proto scale (112x112)
                cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = np.clip((cx - bw / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                y1 = np.clip((cy - bh / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                x2 = np.clip((cx + bw / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                y2 = np.clip((cy + bh / 2) * (112.0 / 448.0), 0, 111).astype(np.int32)
                
                # Initialize background logit canvases for each class
                # We do -10 to initialize the sigmoids at almost zero
                low_res_canvases = {
                    0: np.zeros((112, 112), dtype=np.float32) - 10.0, # Dashed lines
                    1: np.zeros((112, 112), dtype=np.float32) - 10.0, # Solid lines
                    2: np.zeros((112, 112), dtype=np.float32) - 10.0, # Road surface
                    3: np.zeros((112, 112), dtype=np.float32) - 10.0  # Fourth class (if present)
                }
                
                # Crop execution at low resolution to eliminate the background cloud noise
                for i in range(len(filtered_preds)):
                    c_id = filtered_class_ids[i]
                    if c_id in low_res_canvases:
                        cropped_mask = np.zeros((112, 112), dtype=np.float32) - 10.0
                        cropped_mask[y1[i]:y2[i], x1[i]:x2[i]] = raw_masks[i, y1[i]:y2[i], x1[i]:x2[i]]
                        low_res_canvases[c_id] = np.maximum(low_res_canvases[c_id], cropped_mask)
                
                # --- HIGH RESOLUTION UPSAMPLING & BLENDING ---
                # Class 3: Road Surface (Rendered first as a background layer)
                if np.any(filtered_class_ids == 3):
                    combined_road = 1 / (1 + np.exp(-low_res_canvases[3]))
                    full_mask_road = cv2.resize(combined_road, (w, h)) > 0.5
                    mask_overlay[full_mask_road] = (80, 80, 80) # Dark Gray color for the road
                
                # Class 0: Dashed Lines
                if np.any(filtered_class_ids == 0):
                    combined_dashed = 1 / (1 + np.exp(-low_res_canvases[0]))
                    full_mask_dashed = cv2.resize(combined_dashed, (w, h)) > 0.5
                    mask_overlay[full_mask_dashed] = (0, 255, 0) # Green
                    
                # Class 2: Solid Lines
                if np.any(filtered_class_ids == 2):
                    combined_solid = 1 / (1 + np.exp(-low_res_canvases[2]))
                    full_mask_solid = cv2.resize(combined_solid, (w, h)) > 0.5
                    mask_overlay[full_mask_solid] = (0, 0, 255) # Red

                # Class 1: Parking lots 
                if np.any(filtered_class_ids == 1):
                    combined_c3 = 1 / (1 + np.exp(-low_res_canvases[1]))
                    full_mask_c3 = cv2.resize(combined_c3, (w, h)) > 0.5
                    mask_overlay[full_mask_c3] = (255, 0, 0) # Blue

        # Final alpha blending overlay
        annotated_frame = cv2.addWeighted(cv_image, 1.0, mask_overlay, 0.5, 0)

        try:
            # Publish ROS Image Messages
            ros_mask_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding='bgr8')
            ros_mask_msg.header.stamp = msg.header.stamp
            self.mask_pub.publish(ros_mask_msg)
            ros_output_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            ros_output_msg.header.stamp = msg.header.stamp
            self.image_pub.publish(ros_output_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish output image: {str(e)}")

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