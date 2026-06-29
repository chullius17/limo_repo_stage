#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import math

from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs.msg import Image  

from tf2_ros import Buffer, TransformListener

class MissionVisualizer(Node):

    def __init__(self):
        super().__init__('mission_visualizer')

        self.declare_parameter('grid_spacing_m', 1.0)
        self.grid_spacing_m = self.get_parameter('grid_spacing_m').value

        map_qos_profile = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # ── MAP ─────────────────────────────
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_cb,
            map_qos_profile
        )

        # ── INPUT GOALS ─────────────────────
        self.create_subscription(
            PoseArray,
            '/limo/mission/goals',
            self.goals_cb,
            10
        )

        # ── A* GOALS ────────────────────────
        self.create_subscription(
            PoseArray,
            '/limo/mission/goals_astar',
            self.goals_astar_cb,
            10
        )

        # ── PATHS ────────────────────────────
        self.create_subscription(
            Path,
            '/limo/mission/paths',
            self.path_cb,
            10
        )

        # ── MOVING REFERENCE POINT ───────────
        self.create_subscription(
            Path,
            '/limo/control/ref_pt',
            self.path_cb,
            10
        )

        self.create_subscription(
            PoseStamped,
            'limo/control/moving_ref',
            self.ref_cb,
            10
        )

        self.nn_pt = None

        self.create_subscription(
            PoseStamped,
            '/limo/control/nearest_point',
            self.nn_cb,
            10
        )

        # ── QUEUED GOALS (dalla UI, non ancora inviati) ──────────────────
        self.create_subscription(
            PoseArray,
            '/mission/queued_goals',
            self.queued_goals_cb,
            10
        )

        # ── IMAGE PUBLISHER ──────────────────
        self.image_pub = self.create_publisher(
            Image,
            'limo/mission_visualizer/image',
            10
        )

        # ── TF (robot frame) ────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── STATE ────────────────────────────
        self.map_img = None
        self.map_meta = None

        self.goals = []
        self.goals_astar = []
        self.paths = []

        self.ref_pt = None

        self.queued_goals = []

        # Flags to avoid log spamming on continuous receptions
        self.map_received = False
        self.tf_received = False

        self.create_timer(0.1, self.render)

    def queued_goals_cb(self, msg):
        self.queued_goals = msg.poses

    def map_cb(self, msg: OccupancyGrid):
        if not self.map_received:
            self.get_logger().info("Map received for the first time.")
            self.map_received = True

        w = msg.info.width
        h = msg.info.height

        grid = np.array(msg.data).reshape((h, w))

        img = np.zeros((h, w), dtype=np.uint8)

        img[grid == -1] = 127   # unknown
        img[grid == 0]  = 255   # free
        img[grid > 50]  = 0     # obstacle

        self.map_img = img
        self.map_meta = msg.info

    def goals_cb(self, msg):
        self.get_logger().info(f"Received {len(msg.poses)} input goals.")
        self.goals = msg.poses

    def goals_astar_cb(self, msg):
        self.get_logger().info(f"Received {len(msg.poses)} A* goals.")
        self.goals_astar = msg.poses
        self.paths = []  # reset paths quando arriva una nuova missione

    def path_cb(self, msg):
        self.paths.append(msg.poses)  # accumula invece di sovrascrivere

    def ref_cb(self, msg):
        self.get_logger().info(f"Controller is working: the moving reference will be projected.")
        self.ref_pt = msg.pose

    def nn_cb(self, msg):
        self.nn_pt = msg.pose

    def world_to_map(self, x, y):
        if self.map_meta is None:
            return None

        info = self.map_meta

        mx = (x - info.origin.position.x) / info.resolution
        my = (y - info.origin.position.y) / info.resolution

        return int(mx), int(my)

    def map_to_image(self, mx, my):
        h = self.map_img.shape[0]
        return mx, h - 1 - my
        
    def render(self):
        # ── LOGGING STATUS (Throttled every 4 seconds) ──────────────────
        if self.map_img is None:
            self.get_logger().warn("Cannot render: Map topic '/map' is empty or not received yet.", throttle_duration_sec=4.0)
            return

        # Check other inputs status
        if not self.goals:
            self.get_logger().info("Topic '/limo/mission/goals' is currently empty or not received.", throttle_duration_sec=4.0)
        
        if not self.goals_astar:
            self.get_logger().info("Topic '/limo/mission/goals_astar' is currently empty or not received.", throttle_duration_sec=4.0)
            
        if not self.paths:
            self.get_logger().info("Topic '/limo/mission/paths' is currently empty or not received.", throttle_duration_sec=4.0)

        # ────────────────────────────────────────────────────────────────

        img = cv2.cvtColor(self.map_img, cv2.COLOR_GRAY2BGR)
        img = cv2.flip(img, 0)

        # ── GRID OVERLAY ─────────────────────────────────────────────────────
        if self.map_meta is not None:
            info = self.map_meta
            spacing_m = self.grid_spacing_m
            spacing_px = int(spacing_m / info.resolution)

            if spacing_px > 0:
                h_img, w_img = img.shape[:2]

                # Origine della mappa in pixel immagine
                # origin_x/y = coordinate odom del pixel (0,0) della mappa ROS
                # Nel sistema immagine (dopo flip), row=0 è y_max
                origin_col = 0  # col pixel corrispondente a info.origin.position.x
                origin_row = 0  # row ROS corrispondente a info.origin.position.y

                # In world coords, la griglia parte da 0.0 (o dall'origine)
                # Calcoliamo il primo offset in pixel rispetto all'origine della mappa
                ox_m = info.origin.position.x  # x_world del pixel col=0
                oy_m = info.origin.position.y  # y_world del pixel row=0

                # Prima linea verticale a sinistra di x=0 world
                first_col = int((-ox_m % spacing_m) / info.resolution)
                if (-ox_m % spacing_m) < 0:
                    first_col += spacing_px

                # Prima linea orizzontale sotto y=0 world (in coord immagine = sopra)
                first_row_ros = int((-oy_m % spacing_m) / info.resolution)

                grid_color = (160, 160, 160)
                axis_color = (200, 100, 100)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.28
                font_thickness = 1
                text_color = (140, 140, 140)

                # ── LINEE VERTICALI (x = costante) ──
                col = first_col
                while col < w_img:
                    x_world = ox_m + col * info.resolution
                    is_axis = abs(x_world) < info.resolution * 0.5
                    color = axis_color if is_axis else grid_color
                    thickness = 1 if is_axis else 1
                    alpha = 0.5 if is_axis else 0.25

                    overlay = img.copy()
                    cv2.line(overlay, (col, 0), (col, h_img - 1), color, thickness)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                    # Etichetta numerica: in basso nell'immagine
                    label = f"{x_world:.1f}"
                    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                    tx = col - text_size[0] // 2
                    ty = h_img - 4
                    # Piccolo sfondo semi-trasparente per leggibilità
                    bg_overlay = img.copy()
                    cv2.rectangle(bg_overlay,
                                (tx - 1, ty - text_size[1] - 1),
                                (tx + text_size[0] + 1, ty + 1),
                                (50, 50, 50), -1)
                    cv2.addWeighted(bg_overlay, 0.4, img, 0.6, 0, img)
                    cv2.putText(img, label, (tx, ty),
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    col += spacing_px

                # ── LINEE ORIZZONTALI (y = costante) ──
                # In immagine (dopo flip): row_img = h_img - 1 - row_ros
                row_ros = first_row_ros
                while row_ros < h_img:
                    row_img = h_img - 1 - row_ros
                    y_world = oy_m + row_ros * info.resolution
                    is_axis = abs(y_world) < info.resolution * 0.5
                    color = axis_color if is_axis else grid_color
                    alpha = 0.5 if is_axis else 0.25

                    overlay = img.copy()
                    cv2.line(overlay, (0, row_img), (w_img - 1, row_img), color, 1)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                    label = f"{y_world:.1f}"
                    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                    tx = 3
                    ty = row_img + text_size[1] // 2
                    ty = max(text_size[1] + 2, min(ty, h_img - 3))
                    bg_overlay = img.copy()
                    cv2.rectangle(bg_overlay,
                                (tx - 1, ty - text_size[1] - 1),
                                (tx + text_size[0] + 1, ty + 1),
                                (50, 50, 50), -1)
                    cv2.addWeighted(bg_overlay, 0.4, img, 0.6, 0, img)
                    cv2.putText(img, label, (tx, ty),
                                font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    row_ros += spacing_px

        for g in self.goals:
            x, y = g.position.x, g.position.y
            mx, my = self.world_to_map(x, y)
            px, py = self.map_to_image(mx, my)
            cv2.circle(img, (px, py), 4, (0, 255, 0), -1)

        for g in self.goals_astar:
            x, y = g.position.x, g.position.y
            mx, my = self.world_to_map(x, y)
            px, py = self.map_to_image(mx, my)
            cv2.circle(img, (px, py), 5, (255, 255, 0), 2)

        for i, g in enumerate(self.queued_goals):
            x, y = g.position.x, g.position.y
            mx, my = self.world_to_map(x, y)
            px, py = self.map_to_image(mx, my)
            half = 6
            cv2.rectangle(img,
                        (px - half, py - half),
                        (px + half, py + half),
                        (255, 80, 0),   # blu BGR
                        2)
            # numero d'ordine dentro il quadrato
            cv2.putText(img, str(i + 1), (px - 4, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 80, 0), 1, cv2.LINE_AA)

        if self.ref_pt is not None:
            x_ref, y_ref = self.ref_pt.position.x, self.ref_pt.position.y
            mx_ref, my_ref = self.world_to_map(x_ref, y_ref)
            px_ref, py_ref = self.map_to_image(mx_ref, my_ref)
            cv2.circle(img, (px_ref, py_ref), 5, (255, 0, 255), 2)

        if self.nn_pt is not None:
            x_nn = self.nn_pt.position.x
            y_nn = self.nn_pt.position.y
            mx_nn, my_nn = self.world_to_map(x_nn, y_nn)
            px_nn, py_nn = self.map_to_image(mx_nn, my_nn)

            # yaw dal quaternione
            q = self.nn_pt.orientation
            yaw_nn = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            # freccia: 12px nella direzione di yaw (y flippata)
            arrow_len = 12
            px_tip = int(px_nn + arrow_len * math.cos(yaw_nn))
            py_tip = int(py_nn - arrow_len * math.sin(yaw_nn))  # minus per flip y

            cv2.circle(img, (px_nn, py_nn), 4, (0, 140, 255), -1)          # arancione
            cv2.arrowedLine(img, (px_nn, py_nn), (px_tip, py_tip),
                            (0, 140, 255), 2, tipLength=0.4)

        for path in self.paths:
            if len(path) < 2:
                continue

            for i in range(len(path) - 1):
                x1 = path[i].pose.position.x
                y1 = path[i].pose.position.y
                x2 = path[i+1].pose.position.x
                y2 = path[i+1].pose.position.y

                mx1, my1 = self.world_to_map(x1, y1)
                px1, py1 = self.map_to_image(mx1, my1)

                mx2, my2 = self.world_to_map(x2, y2)
                px2, py2 = self.map_to_image(mx2, my2)

                cv2.line(img, (px1, py1), (px2, py2), (255, 255, 0), 2)

        try:
            tf = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            mx, my = self.world_to_map(x, y)

            h = self.map_img.shape[0]
            my = h - 1 - my

            cv2.circle(img, (mx, my), 6, (0, 0, 255), -1)
            
            if not self.tf_received:
                self.get_logger().info("TF 'map' -> 'base_link' successfully resolved.")
                self.tf_received = True
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {str(e)}", throttle_duration_sec=4.0)
            self.tf_received = False

        # ── PUBLISH IMAGE ────────────────────
        img_msg = Image()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'map'
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        img_msg.encoding = 'bgr8'
        img_msg.is_bigendian = 0
        img_msg.step = img.shape[1] * 3
        img_msg.data = img.tobytes()

        self.image_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)

    node = MissionVisualizer()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info("MissionVisualizer running...")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("MissionVisualizer interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()