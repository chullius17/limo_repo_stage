#!/usr/bin/env python3

import heapq
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException

# Importing our custom service definition
from limo_interfaces.srv import GetSequencePlan 


def heuristic(p1, p2):
    """
    Octile distance: optimal heuristic for 8-connected grid transitions.
    """
    dy = abs(p1[0] - p2[0])
    dx = abs(p1[1] - p2[1])
    return (dy + dx) + (1.414 - 2.0) * min(dy, dx)


def get_neighbors(current, grid):
    """
    Finds valid 8-connected neighbor pixels inside grid boundaries.
    """
    y, x = current
    height, width = grid.shape
    neighbors = []
    
    directions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
    ]
    
    for dy, dx, move_cost in directions:
        ny, nx = y + dy, x + dx
        
        if 0 <= ny < height and 0 <= nx < width:
            cell_value = grid[ny, nx]
            
            # 100 = Route (preferred, standard cost)
            # 0 = Open space (accessible but heavily penalized)
            # Everything else (obstacles/unknown) is blocked
            if cell_value == 100:
                neighbors.append(((ny, nx), move_cost * 1.0))
            elif cell_value == 0:
                neighbors.append(((ny, nx), move_cost * 50.0))
                
    return neighbors


def astar_grid(grid, start, goal):
    """
    Computes A* path between two discrete pixel coordinate tuples.
    """
    if start == goal:
        return [start]

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor, move_cost in get_neighbors(current, grid):
            tentative_g = g_score[current] + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None


class SequencePathService(Node):

    def __init__(self):
        super().__init__('sequence_path_service_node')

        # --- TF BUFFER & LISTENER ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_costmap = None

        # --- COSTMAP SUBSCRIPTION ---
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/limo/debug/route_grid',
            self.map_callback,
            1
        )

        # --- CUSTOM SERVICE SERVER ---
        self.srv = self.create_service(
            GetSequencePlan,
            '/plan_sequence_path',
            self.handle_sequence_planning
        )

        self.get_logger().info("Service Server [/plan_sequence_path] con snap sulla mappa avviato.")

    def map_callback(self, msg: OccupancyGrid):
        """
        Maintains the most recent occupancy grid data structure in RAM.
        """
        self.latest_costmap = msg

    def handle_sequence_planning(self, request: GetSequencePlan.Request, response: GetSequencePlan.Response):
        """
        Handles sequential waypoint routing requests by shifting frames from Odom to local pixels,
        snapping each point to the nearest valid route pixel (value 100).
        """
        self.get_logger().info(f"Ricevuta richiesta per {len(request.goals)} obiettivi sequenziali.")

        if self.latest_costmap is None:
            self.get_logger().error("Costmap non disponibile. Calcolo annullato.")
            return response

        if len(request.goals) == 0:
            self.get_logger().warn("La lista degli obiettivi fornita è vuota.")
            return response

        # 1. LOOKUP CURRENT ROBOT ODOMETRY POSITION
        try:
            tf = self.tf_buffer.lookup_transform(
                "odom",
                "base_footprint",
                rclpy.time.Time()
            )
        except TransformException as e:
            self.get_logger().error(f"Impossibile ottenere TF odom -> base_footprint: {e}")
            return response

        robot_x = tf.transform.translation.x
        robot_y = tf.transform.translation.y
        q = tf.transform.rotation

        robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)

        # 2. GRID PARSING PREPARATION
        info = self.latest_costmap.info
        resolution = info.resolution
        width = info.width
        height = info.height
        grid_matrix = np.array(self.latest_costmap.data, dtype=np.int8).reshape((height, width))

        cx = width // 2
        cy = height // 2

        # Estraggo tutti i pixel della mappa che appartengono alla strada (valore 100)
        route_indices = np.argwhere(grid_matrix == 100)
        if route_indices.shape[0] == 0:
            self.get_logger().error("Nessun pixel di tipo 'strada' (100) trovato nella mappa!")
            return response

        # 3. BUILD WAYPOINTS LIST IN ROBOT FRAME METERS (Starting at Robot origin 0,0)
        waypoints_m = [(0.0, 0.0)]

        for goal_pose in request.goals:
            dx = goal_pose.pose.position.x - robot_x
            dy = goal_pose.pose.position.y - robot_y
            
            rx = cos_y * dx + sin_y * dy
            ry = -sin_y * dx + cos_y * dy
            waypoints_m.append((rx, ry))

        # 4. CONVERT METERS AND SNAP TO NEAREST ROUTE PIXEL (100)
        pixel_waypoints = []
        for rx, ry in waypoints_m:
            # Calcolo iniziale del pixel teorico corrispondente alle coordinate metriche
            ideal_x_px = int(cx - ry / resolution)
            ideal_y_px = int(cy - rx / resolution)
            
            # Verifico i confini minimi
            if not (0 <= ideal_x_px < width and 0 <= ideal_y_px < height):
                self.get_logger().error(f"Uno degli obiettivi ({rx}m, {ry}m) è fuori dai confini dei pixel.")
                return response

            # TROVA IL PIXEL DI STRADA (100) PIÙ VICINO
            # Calcolo la distanza euclidea tra il pixel teorico e tutti i punti stabili della pista
            distances_sq = (route_indices[:, 0] - ideal_y_px) ** 2 + (route_indices[:, 1] - ideal_x_px) ** 2
            min_idx = np.argmin(distances_sq)
            
            snapped_y, snapped_x = route_indices[min_idx]
            pixel_waypoints.append((int(snapped_y), int(snapped_x)))

        # 5. EXECUTE SEQUENTIAL PIECEWISE A* PLANNING
        full_pixel_path = []
        
        for i in range(len(pixel_waypoints) - 1):
            segment_start = pixel_waypoints[i]
            segment_goal = pixel_waypoints[i+1]
            
            segment_path = astar_grid(grid_matrix, segment_start, segment_goal)
            
            if segment_path is None:
                self.get_logger().error(f"A* fallito nel segmento intermedio da {segment_start} a {segment_goal}")
                return response
            
            if i > 0:
                full_pixel_path.extend(segment_path[1:])
            else:
                full_pixel_path.extend(segment_path)

        # 6. RECONVERT CONCATENATED PATH PIXELS BACK INTO ODOMETRY METERS MESSAGE
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for y_idx, x_idx in full_pixel_path:
            rx_m = (cy - y_idx) * resolution
            ry_m = (cx - x_idx) * resolution
            
            ox_m = robot_x + (cos_y * rx_m - sin_y * ry_m)
            oy_m = robot_y + (sin_y * rx_m + cos_y * ry_m)

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = ox_m
            pose.pose.position.y = oy_m
            pose.pose.position.z = 0.0
            
            path_msg.poses.append(pose)

        response.plan = path_msg
        self.get_logger().info(f"Pianificazione completata. Generati {len(full_pixel_path)} nodi totali.")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SequencePathService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()