import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import math

class SimpleFusion(Node):
    def __init__(self):
        super().__init__('simple_fusion_node')
        self.maps = [None, None, None]
        
        # Sottoscrizioni ai 3 colori
        for i, color in enumerate(['magenta', 'red', 'green']):
            self.create_subscription(OccupancyGrid, f'/limo/map_paper_{color}', 
                                     lambda msg, idx=i: self.update_map(msg, idx), 10)
        
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/limo/vision_pose', 10)
        # 20Hz per un controllo fluido
        self.create_timer(0.05, self.timer_callback) 
        self.get_logger().info('SimpleFusion node avviato: Fusione cluster e gradiente attiva.')

    def update_map(self, msg, idx):
        self.maps[idx] = msg

    def timer_callback(self):
        valid_maps = [m for m in self.maps if m is not None]
        if not valid_maps: return

        # 1. Fusione delle mappe
        h, w = valid_maps[0].info.height, valid_maps[0].info.width
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in valid_maps:
            combined = np.maximum(combined, np.array(m.data).reshape((h, w)))

        # 2. Analisi dei cluster sulla riga h-10 (vicino) e h-15 (leggermente più lontano)
        scan_row = combined[h-10, :]
        mask = scan_row > 50
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        
        if mask[0]: starts = np.insert(starts, 0, 0)
        if mask[-1]: ends = np.append(ends, len(mask) - 1)

        best_yaw = 0.0
        best_center_px = w / 2.0
        found_valid = False
        min_yaw_error = 1.0 # Soglia di tolleranza allineamento

        # 3. Analisi di ogni cluster trovato
        for s, e in zip(starts, ends):
            if (e - s) > 3: # Filtro rumore
                segment = scan_row[s:e]
                coords = np.arange(s, e)
                center_px = np.sum(coords * segment) / np.sum(segment)
                
                # Gradiente (Yaw locale) confrontando riga h-10 e h-15
                upper_row = combined[h-15, max(0, s-2):min(w, e+2)]
                idx_up = np.where(upper_row > 50)[0]
                
                if len(idx_up) > 0:
                    center_up = np.mean(idx_up) + max(0, s-2)
                    yaw_cluster = math.atan2(center_px - center_up, 5) # 5 è la distanza tra le righe
                else:
                    yaw_cluster = 0.0

                # Selezione del cluster più parallelo al robot (yaw vicino a 0)
                if abs(yaw_cluster) < min_yaw_error:
                    min_yaw_error = abs(yaw_cluster)
                    best_yaw = yaw_cluster
                    best_center_px = center_px
                    found_valid = True

        # 4. Pubblicazione della posa (se abbiamo trovato un cluster valido)
        if found_valid:
            error_y = (best_center_px - (w / 2.0)) * valid_maps[0].info.resolution
            
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "base_footprint"
            msg.header.stamp = self.get_clock().now().to_msg()
            
            msg.pose.pose.position.y = -error_y
            msg.pose.pose.orientation.z = math.sin(best_yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(best_yaw / 2.0)
            
            # Covarianza ridotta: ci fidiamo del cluster scelto
            msg.pose.covariance[7] = 0.01  # Y
            msg.pose.covariance[35] = 0.05 # Yaw
            
            self.pose_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()