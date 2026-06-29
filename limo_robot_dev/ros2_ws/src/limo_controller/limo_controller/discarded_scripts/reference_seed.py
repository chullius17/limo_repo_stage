#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

class ReferenceBuilder(Node):
    def __init__(self):
        super().__init__('reference_builder_node')

        # --- PARAMETRI ---
        self.declare_parameter('input_map_topic', '/limo/global_map_combined')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('radius_m', 1.3) 
        self.declare_parameter('cost_threshold', 90)

        self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                             rclpy.Parameter.Type.BOOL, True)])

        input_topic = self.get_parameter('input_map_topic').value
        self.robot_frame = self.get_parameter('robot_frame').value

        #Useful parameters
        self.MIN_DIST   = 0.10
        self.MAX_DIST   = 0.12

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
            '/limo/debug/reference_map', 
            10
        )

        # DEBUG 2: Flood Fill Mask (Bianco/Nero + Assi/Cerchio)
        self.flood_pub = self.create_publisher(
            Image,
            '/limo/debug/flood_mask',
            10
        )

        # DEBUG 3: Generazione candidate route mask
        self.route_pub = self.create_publisher(
            Image,
            '/limo/debug/candidate_route', 
            10
        )

        self.get_logger().info(f'ReferenceBuilder inizializzato. In ascolto su: {input_topic}')

    def map_callback(self, msg: OccupancyGrid):
        # 1. Recupero delle dimensioni e della risoluzione della mappa
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution  # Metri per singolo pixel

        if width == 0 or height == 0 or resolution == 0.0:
            return

        # Leggiamo i parametri dinamici
        radius_meters = self.get_parameter('radius_m').value
        cost_threshold = self.get_parameter('cost_threshold').value

        # 2. Conversione dei dati grezzi in una matrice NumPy (costi ROS: da -1 a 100)
        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # 3. Identificazione del centro (Origine del Robot) e calcolo raggio in pixel
        center_x = width // 2
        center_y = height // 2
        radius_px = int(radius_meters / resolution)

        # =====================================================================
        # >>> GENERAZIONE BASE DEBUG 1: COSTMAP CLASSICA IN SCALA DI GRIGI <<<
        # =====================================================================
        gray_image = np.zeros_like(grid_data, dtype=np.uint8)
        gray_image[grid_data >= 0] = (grid_data[grid_data >= 0] / 100.0 * 255.0).astype(np.uint8)
        gray_image[grid_data == -1] = 100 
        
        # Convertiamo in BGR per poter disegnare assi e cerchi a colori
        color_debug_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # =====================================================================
        # >>> ALGORITMO DI FLOOD FILL (BFS) CON FALLBACK VERTICALE <<<
        # =====================================================================
        # Maschera booleana: inizialmente tutto False (Nero / Rifiutato)
        flood_mask = np.zeros((height, width), dtype=bool)
        queue = deque()
        
        # Variabili di appoggio per il punto di partenza (seed)
        start_y = center_y
        start_x = center_x
        found_seed = False

        # Verifica preliminare sul centro geometrico
        center_cost = grid_data[center_y, center_x]
        if 0 <= center_cost < cost_threshold:
            found_seed = True

        else:
            # BUG/OSTACOLO DETECTED: Il centro non è libero. Cerca il primo punto libero salendo
            self.get_logger().warn(
                f"Centro mappa occupato o sconosciuto (costo: {center_cost}). "
                f"Ricerca del primo punto valido salendo verticalmente da y={center_y}..."
            )
            
            # 1. Extract the column slice from the top (y=0) up to the center.
            # We reverse it with [::-1] so that index 0 corresponds to center_y - 1 (closest to the robot).
            column_slice = grid_data[0:center_y, center_x][::-1]

            # 2. Create a boolean mask based on the cost threshold condition
            mask = (column_slice >= 0) & (column_slice < cost_threshold)

            # 3. Find relative indices where the condition is met
            valid_indices = np.where(mask)[0]

            # 4. If at least one valid cell is found, pick the closest one to the center
            if valid_indices.size > 0:
                first_relative_idx = valid_indices[0]
                
                # Calculate absolute y coordinate by subtracting the offset from the center
                start_y = (center_y - 1) - first_relative_idx
                start_x = center_x
                cost = column_slice[first_relative_idx]
                found_seed = True
                
                self.get_logger().info(f"Nuovo seed di partenza trovato a x={start_x}, y={start_y} (costo: {cost})")

        # Se abbiamo trovato un punto di partenza valido (il centro o uno sottostante), avviamo la BFS
        if found_seed:
            queue.append((start_y, start_x))
            flood_mask[start_y, start_x] = True
        else:
            self.get_logger().error("Nessun punto libero trovato scendendo dal centro lungo l'intera colonna!")

        # Ottimizzazione: quadrato del raggio per evitare sqrt nel ciclo
        radius_px_sq = radius_px ** 2
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 4-connessioni

        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if not flood_mask[nr, nc]:
                        # Limite metrico del cerchio rispetto al CENTRO REALE del robot
                        if ((nc - center_x) ** 2 + (nr - center_y) ** 2) <= radius_px_sq:
                            # Limite di costo
                            cost = grid_data[nr, nc]
                            if 0 <= cost < cost_threshold:
                                flood_mask[nr, nc] = True
                                queue.append((nr, nc))

        # Dopo il flood fill, prima della pubblicazione
        # =====================================================================
        # >>> DISTANCE TRANSFORM DAI MURI  <<<
        # =====================================================================

        # Distance transform: ogni pixel riceve la distanza in pixel dal muro più vicino
        dist_px = cv2.distanceTransform(flood_mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Converti soglie metriche in pixel
        d_min_px = self.MIN_DIST / resolution 
        d_max_px = self.MAX_DIST / resolution 

        # Anello: punti che sono nella banda [10cm, 12cm] dai muri E dentro il flood fill
        ring_mask = (
            (dist_px >= d_min_px) &
            (dist_px <= d_max_px)
        )

        # =====================================================================
        # >>> GENERAZIONE BASE DEBUG 2: MASCHERA FLOOD FILL BIANCO/NERO <<<
        # =====================================================================
        flood_debug_img = np.zeros((height, width, 3), dtype=np.uint8)
        flood_debug_img[flood_mask] = [255, 255, 255] # Bianco per i punti accettati

        # Se il flood fill è partito da un punto alternativo, disegniamo un pallino rosso di debug su entrambe le immagini
        if found_seed and (start_y != center_y):
            for img in [color_debug_img, flood_debug_img]:
                cv2.circle(img, (start_x, start_y), 2, (0, 0, 255), -1) # Pallino rosso sul seed alternativo

        # ===========================================================================
        # >>> GENERAZIONE BASE DEBUG 3: MASCHERA FLOOD FILL BIANCO/NERO + ROUTE <<<
        # ===========================================================================
        route_debug_img = color_debug_img.copy()
        route_debug_img[ring_mask] = [0, 255, 255]  # ciano

        # =====================================================================
        # >>> DISEGNO DEGLI ELEMENTI DI RIFERIMENTO SU TUTTE LE IMMAGINI <<<
        # =====================================================================
        lunghezza_assi = 30 
        spessore = 2

        # Applichiamo la grafica di riferimento in un ciclo su entrambe le immagini di debug
        for img in [color_debug_img, flood_debug_img, route_debug_img]:
            # Cerchio metrico (Giallo/Arancio tenue) ancorato al centro geometrico
            cv2.circle(img, (center_x, center_y), radius_px, (0, 165, 255), 1, lineType=cv2.LINE_AA)

            # Asse X: Rosso -> Avanti (-Y OpenCV)
            cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y - lunghezza_assi), (0, 0, 255), spessore, tipLength=0.2)

            # Asse Y: Verde -> Sinistra (-X OpenCV)
            cv2.arrowedLine(img, (center_x, center_y), (center_x - lunghezza_assi, center_y), (0, 255, 0), spessore, tipLength=0.2)

        # Origine (Centro): Usiamo il Bianco sulla mappa classica e il Ciano/Azzurro sul flood bianco per contrasto
        cv2.circle(color_debug_img, (center_x, center_y), 3, (255, 255, 255), -1)
        cv2.circle(flood_debug_img, (center_x, center_y), 3, (255, 255, 0), -1)
        cv2.circle(route_debug_img, (center_x, center_y), 3, (255, 255, 0), -1)

        # =====================================================================
        # >>> PUBBLICAZIONE DEI MESSAGGI ROS2 <<<
        # =====================================================================
        # 1. Pubblica Costmap Classica
        img_msg = self.bridge.cv2_to_imgmsg(color_debug_img, encoding='bgr8')
        img_msg.header = msg.header 
        self.image_pub.publish(img_msg)

        # 2. Pubblica Flood Mask
        flood_msg = self.bridge.cv2_to_imgmsg(flood_debug_img, encoding='bgr8')
        flood_msg.header = msg.header
        self.flood_pub.publish(flood_msg)

        # 3. Pubblica Route Mask
        route_msg = self.bridge.cv2_to_imgmsg(route_debug_img, encoding='bgr8')
        route_msg.header = msg.header
        self.route_pub.publish(route_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()