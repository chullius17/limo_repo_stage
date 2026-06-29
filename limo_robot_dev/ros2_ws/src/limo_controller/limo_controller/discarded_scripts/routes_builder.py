#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

class RoutesBuilder(Node):
    def __init__(self):
        super().__init__('reference_builder_node')

        # --- PARAMETRI ---
        self.declare_parameter('input_map_topic', '/limo/global_map_combined')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('cost_threshold', 80)

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                             rclpy.Parameter.Type.BOOL, True)])

        input_topic = self.get_parameter('input_map_topic').value
        self.robot_frame = self.get_parameter('robot_frame').value

        #Useful parameters
        self.MIN_DIST   = 0.08
        self.MAX_DIST   = 0.10

        self.bridge = CvBridge()

        # --- SUBSCRIPTION & PUBLISHERS ---
        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            input_topic, 
            self.map_callback, 
            10
        )
        
        # DEBUG 1: Costmap classica (Scala di grigi + Assi/Cerchio)
        self.image_pub = self.create_publisher(
            Image, 
            '/limo/debug/costmap4reference', 
            10
        )

        # DEBUG 3: Generazione candidate route mask
        self.route_pub = self.create_publisher(
            Image,
            '/limo/debug/candidate_route', 
            10
        )

        self.route_grid_pub = self.create_publisher(
            OccupancyGrid,
            '/limo/debug/route_grid',
            10
        )

        self.get_logger().info(f'ReferenceBuilder inizializzato. In ascolto su: {input_topic}')

    def mask_to_occupancy_grid(self, mask, msg):
        grid = OccupancyGrid()
        grid.header = msg.header

        grid.info = msg.info  # riusa geometria mappa

        data = np.zeros(mask.shape, dtype=np.int8)

        data[:] = 0               # free
        data[mask] = 100          # route

        # unknown (se vuoi mantenerli coerenti)
        # data[~known_mask] = -1

        grid.data = data.flatten().tobytes()

        return grid

    def map_callback(self, msg: OccupancyGrid):
        # 1. Recupero delle dimensioni e della risoluzione della mappa
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution  # Metri per singolo pixel

        if width == 0 or height == 0 or resolution == 0.0:
            return

        cost_threshold = self.get_parameter('cost_threshold').value

        # 2. Conversione dei dati grezzi in una matrice NumPy (costi ROS: da -1 a 100)
        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # 3. Identificazione del centro (Origine del Robot) e calcolo raggio in pixel
        center_x = width // 2
        center_y = height // 2

        # =====================================================================
        # >>> GENERAZIONE BASE DEBUG 1: COSTMAP CLASSICA IN SCALA DI GRIGI <<<
        # =====================================================================
        gray_image = np.zeros_like(grid_data, dtype=np.uint8)
        gray_image[grid_data >= 0] = (grid_data[grid_data >= 0] / 100.0 * 255.0).astype(np.uint8)
        gray_image[grid_data == -1] =  50
        
        # Convertiamo in BGR per poter disegnare assi e cerchi a colori
        color_debug_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        # =====================================================================
        # >>> MASK TO IDENTIFY WALLS <<<
        # =====================================================================
        wall_mask = grid_data > cost_threshold  # We don't consider the unknown
        known_mask = grid_data >= 0
        free_mask = ~wall_mask

        dist_px = cv2.distanceTransform(free_mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Converti soglie metriche in pixel
        d_min_px = self.MIN_DIST / resolution 
        d_max_px = self.MAX_DIST / resolution 

        route_mask = (
            (dist_px >= d_min_px) &
            (dist_px <= d_max_px) &
            known_mask
        )

        route_debug_img = color_debug_img.copy()
        route_debug_img[route_mask] = [0, 255, 255]

        # =====================================================================
        # >>> DISEGNO DEGLI ELEMENTI DI RIFERIMENTO SU TUTTE LE IMMAGINI <<<
        # =====================================================================
        
        lunghezza_assi = 30 
        spessore = 2

        # Applichiamo la grafica di riferimento in un ciclo su entrambe le immagini di debug
        for img in [color_debug_img, route_debug_img]:
            # Asse X: Rosso -> Avanti (-Y OpenCV)
            cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y - lunghezza_assi), (0, 0, 255), spessore, tipLength=0.2)

            # Asse Y: Verde -> Sinistra (-X OpenCV)
            cv2.arrowedLine(img, (center_x, center_y), (center_x - lunghezza_assi, center_y), (0, 255, 0), spessore, tipLength=0.2)

        # Origine (Centro): Usiamo il Bianco sulla mappa classica e il Ciano/Azzurro sul flood bianco per contrasto
        cv2.circle(color_debug_img, (center_x, center_y), 3, (255, 255, 255), -1)
        cv2.circle(route_debug_img, (center_x, center_y), 3, (255, 255, 0), -1)

        # =====================================================================
        # >>> PUBBLICAZIONE DEI MESSAGGI ROS2 <<<
        # =====================================================================

        # 1. Pubblica Costmap Classica
        img_msg = self.bridge.cv2_to_imgmsg(color_debug_img, encoding='bgr8')
        img_msg.header = msg.header 
        self.image_pub.publish(img_msg)

        # 3. Pubblica Route Mask
        route_msg = self.bridge.cv2_to_imgmsg(route_debug_img, encoding='bgr8')
        route_msg.header = msg.header
        self.route_pub.publish(route_msg)

        # =====================================================================
        # >>> PUBBLICAZIONE GRID OUTPUT <<<
        # =====================================================================

        route_grid_msg = self.mask_to_occupancy_grid(route_mask, msg)
        self.route_grid_pub.publish(route_grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoutesBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()