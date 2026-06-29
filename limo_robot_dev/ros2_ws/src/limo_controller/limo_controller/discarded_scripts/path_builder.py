#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import json
from tf2_ros import Buffer, TransformListener
import math

# Import the custom service definition
from limo_interfaces.srv import GetSequencePlan


class PathBuilder(Node):

    def __init__(self):
        super().__init__('route_grid_subscriber')

        # --- PATH BUFFER ---
        # Holds the active global path in odom meters coordinates
        self.current_path = []
        self.waiting_for_service = False

        # --- PARAMETRO GOALS ---
        self.declare_parameter('goals', '[{"x": 0.0, "y": 0.0}]')
        raw = self.get_parameter('goals').value
        data = json.loads(raw)
        self.goals = [(g["x"], g["y"]) for g in data]

        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- SERVICE CLIENT ---
        # Connects to the newly built A* sequence service node
        self.client = self.create_client(GetSequencePlan, '/plan_sequence_path')

        # --- SUB ---
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/limo/debug/route_grid',
            self.callback,
            10
        )

        # --- PUB IMAGE ---
        self.image_pub = self.create_publisher(
            Image,
            '/limo/debug/route_view',
            10
        )

        self.get_logger().info("RouteGridSubscriber pronto con supporto Client A*")

    def call_planning_service(self):
        """
        Sends an asynchronous planning request with the goals parameter array.
        """
        if not self.client.service_is_ready():
            self.get_logger().warn("Servizio /plan_sequence_path non disponibile, attendo...", throttle_duration_sec=2.0)
            return

        self.waiting_for_service = True
        
        request = GetSequencePlan.Request()
        for gx, gy in self.goals:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = gx
            pose.pose.position.y = gy
            request.goals.append(pose)

        # Fire asynchronous service call request
        future = self.client.call_async(request)
        future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        """
        Stores the resulting unrolled path sequence into memory upon recovery.
        """
        try:
            response = future.result()
            if response and response.plan.poses:
                self.current_path = response.plan.poses
                self.get_logger().info(f"Nuovo percorso A* ricevuto dal server. Nodi: {len(self.current_path)}")
            else:
                self.get_logger().error("Il server A* ha restituito un percorso vuoto.")
        except Exception as e:
            self.get_logger().error(f"Chiamata di servizio fallita: {e}")
        
        self.waiting_for_service = False

    def grid_to_image(self, data, width, height):
        img = np.zeros((height, width), dtype=np.uint8)
        img[data == 0] = 50        # free
        img[data == 100] = 255     # route
        img[data == -1] = 0        # unknown
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        img = self.grid_to_image(data, width, height)

        cx = width // 2
        cy = height // 2

        # --- ASSE ROBOT BASE GRAPHICS ---
        axis_len = 30
        cv2.arrowedLine(img, (cx, cy), (cx, cy - axis_len), (0, 0, 255), 2, tipLength=0.2) # X
        cv2.arrowedLine(img, (cx, cy), (cx - axis_len, cy), (0, 255, 0), 2, tipLength=0.2) # Y
        cv2.circle(img, (cx, cy), 3, (255, 255, 0), -1)

        # --- ROBOT ODOMETRY POSITION ---
        try:
            tf = self.tf_buffer.lookup_transform("odom", "base_footprint", rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF non disponibile: {e}", throttle_duration_sec=2.0)
            return

        robot_x = tf.transform.translation.x
        robot_y = tf.transform.translation.y
        q = tf.transform.rotation

        robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)

        # -------------------------------------------------------------
        # 1. TRIGGER PLANNING IF PATH STORAGE IS EMPTY
        # -------------------------------------------------------------
        if not self.current_path and not self.waiting_for_service:
            self.get_logger().info("Richiesta di un nuovo percorso asincrono...")
            self.call_planning_service()

        # -------------------------------------------------------------
        # 2. DRAW CALCULATED PATH (ODOM METERS -> ROBOT FRAME -> PIXEL)
        # -------------------------------------------------------------
        last_pixel = None
        for pose_stamped in self.current_path:
            px_odom = pose_stamped.pose.position.x
            py_odom = pose_stamped.pose.position.y

            # Transform from absolute Odom back to Robot local frame
            dx = px_odom - robot_x
            dy = py_odom - robot_y
            rx = cos_y * dx + sin_y * dy
            ry = -sin_y * dx + cos_y * dy

            # Transform from robot meters into image pixels
            x_path_px = int(cx - ry / resolution)
            y_path_px = int(cy - rx / resolution)

            if 0 <= x_path_px < width and 0 <= y_path_px < height:
                # Draw standard tracking dots in green (0, 255, 0)
                cv2.circle(img, (x_path_px, y_path_px), 1, (0, 255, 0), -1)
                
                # Draw sequential path lines joining waypoints
                if last_pixel is not None:
                    cv2.line(img, last_pixel, (x_path_px, y_path_px), (0, 255, 0), 1)
                last_pixel = (x_path_px, y_path_px)

        # --- DRAW IDEAL GOALS ---
        for gx, gy in self.goals:
            dx = gx - robot_x
            dy = gy - robot_y
            goal_x_r = cos_y * dx + sin_y * dy
            goal_y_r = -sin_y * dx + cos_y * dy

            x_px = int(cx - goal_y_r / resolution)
            y_px = int(cy - goal_x_r / resolution)

            if 0 <= x_px < width and 0 <= y_px < height:
                cv2.circle(img, (x_px, y_px), 5, (0, 0, 255), -1) # Red circle
                cv2.putText(img, "G", (x_px + 3, y_px - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # --- PUBLISH DRAWN IMAGE ---
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        img_msg.header = msg.header
        self.image_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()