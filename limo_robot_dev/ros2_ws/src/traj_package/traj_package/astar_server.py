#!/usr/bin/env python3

import heapq
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import time

from limo_interfaces.srv import GetSequencePlan

OBSTACLE_THRESHOLD = 90

def heuristic(p1, p2):
    dy = abs(p1[0] - p2[0])
    dx = abs(p1[1] - p2[1])
    return (dy + dx) + (1.414 - 2.0) * min(dy, dx)

def get_neighbors(current, grid):
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
            cell_value = int(grid[ny, nx])
            if cell_value < 0 or cell_value > OBSTACLE_THRESHOLD:
                continue  # -1 = unknown, >90 = obstacle
            neighbors.append(((ny, nx), move_cost * (1.0 + cell_value / 100.0)))

    return neighbors

def astar_grid(grid, start, goal):
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

    return None  # No path found


def odom_to_pixel(ox, oy, info):
    """Converte coordinate odom (metri) in pixel (row, col)."""
    col = int((ox - info.origin.position.x) / info.resolution)
    row = int((oy - info.origin.position.y) / info.resolution)
    # row=0 è il basso della mappa ROS, quindi non serve flipud qui:
    # la grid_matrix è già in ordine ROS (row 0 = y minimo)
    return (row, col)

def pixel_to_odom(row, col, info):
    """Converte pixel (row, col) in coordinate odom (metri), centro del pixel."""
    ox = info.origin.position.x + (col + 0.5) * info.resolution
    oy = info.origin.position.y + (row + 0.5) * info.resolution
    return (ox, oy)

def snap_to_valid(pixel, grid, max_radius_px=30):
    """
    Trova il pixel valido (costo <= OBSTACLE_THRESHOLD, != -1) più vicino
    a `pixel` entro un raggio massimo, con BFS sull'intorno.
    """
    row, col = pixel
    height, width = grid.shape

    if (0 <= row < height and 0 <= col < width and
            0 <= int(grid[row, col]) <= OBSTACLE_THRESHOLD):
        return pixel  # già valido

    visited = set()
    queue = [(0, row, col)]  # (dist_Manhattan, r, c)

    while queue:
        d, r, c = heapq.heappop(queue)
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if d > max_radius_px:
            break
        if 0 <= r < height and 0 <= c < width:
            val = int(grid[r, c])
            if 0 <= val <= OBSTACLE_THRESHOLD:
                return (r, c)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in visited:
                        heapq.heappush(queue, (max(abs(nr - row), abs(nc - col)), nr, nc))

    return None  # nessun pixel valido nel raggio


def nearest_neighbor_ordering(start_px, goal_pixels, grid):
    """
    Ordina i goal con greedy Nearest Neighbor a partire da start_px.
    Restituisce (ordered_indices, ordered_pixels, paths)
    dove paths[i] è il path A* da ordered_pixels[i-1] a ordered_pixels[i]
    (paths[0] è da start_px al primo goal).
    """
    remaining = list(enumerate(goal_pixels))  # (original_idx, pixel)
    current = start_px
    ordered_indices = []
    ordered_pixels = []
    paths = []

    while remaining:
        best_cost = float('inf')
        best_idx_in_remaining = None
        best_path = None

        for i, (orig_idx, gpx) in enumerate(remaining):
            path = astar_grid(grid, current, gpx)
            if path is None:
                continue
            # Costo del path = somma dei g_score simulata dalla lunghezza
            # (usiamo len come proxy; per precisione si potrebbe ricalcolare)
            cost = len(path)
            if cost < best_cost:
                best_cost = cost
                best_idx_in_remaining = i
                best_path = path

        if best_idx_in_remaining is None:
            # Goal irraggiungibili: li scartiamo loggando
            break

        orig_idx, gpx = remaining.pop(best_idx_in_remaining)
        ordered_indices.append(orig_idx)
        ordered_pixels.append(gpx)
        paths.append(best_path)
        current = gpx

    return ordered_indices, ordered_pixels, paths


class AStarRefService(Node):

    def __init__(self):
        super().__init__('astar_ref_service_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_costmap = None

        self.declare_parameter('topic_map_sub', '/limo/routes/coordinated_map')
        self.topic_map_sub = self.get_parameter('topic_map_sub').value

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.topic_map_sub,
            self.map_callback,
            map_qos
        )

        self.srv = self.create_service(
            GetSequencePlan,
            '/plan_sequence_path',
            self.handle_sequence_planning
        )
        self.get_logger().info('AStarRefService ready.')

    def map_callback(self, msg: OccupancyGrid):
        self.latest_costmap = msg

    def handle_sequence_planning(self, request: GetSequencePlan.Request, response: GetSequencePlan.Response):

        t0 = time.perf_counter()
        
        self.get_logger().info(f"Received request for {len(request.goals)} sequential objectives.")

        # If costmap not yet received, wait a short time to allow late publishers
        # (transient_local / latched) to deliver the message and avoid races.
        if self.latest_costmap is None:
            wait_time = 2.0  # seconds total to wait for the costmap
            poll_interval = 0.05
            waited = 0.0
            while self.latest_costmap is None and waited < wait_time:
                # allow ROS to process incoming messages
                try:
                    rclpy.spin_once(self, timeout_sec=poll_interval)
                except Exception:
                    # ignore spin issues inside callback
                    pass
                waited += poll_interval

        if self.latest_costmap is None:
            self.get_logger().error("Costmap not available.")
            return response
        if len(request.goals) == 0:
            self.get_logger().warn("Empty goal list.")
            return response

        # 1. ROBOT POSE dal TF
        try:
            tf = self.tf_buffer.lookup_transform("odom", "base_footprint", rclpy.time.Time())
        except TransformException as e:
            self.get_logger().error(f"TF odom->base_footprint failed: {e}")
            return response

        robot_x = tf.transform.translation.x
        robot_y = tf.transform.translation.y
        q = tf.transform.rotation
        robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # 2. GRID
        info = self.latest_costmap.info
        height, width = info.height, info.width
        grid_matrix = np.array(self.latest_costmap.data, dtype=np.int8).reshape((height, width))

        # 3. START PIXEL
        start_px = odom_to_pixel(robot_x, robot_y, info)
        start_px = snap_to_valid(start_px, grid_matrix)
        if start_px is None:
            self.get_logger().error("Start position not snappable to valid cell.")
            return response

        # 4. GOAL PIXELS — snap ciascuno al pixel valido più vicino
        goal_pixels = []
        valid_goal_indices = []
        snapped_goals = []

        for i, goal_pose in enumerate(request.goals):
            gx = goal_pose.pose.position.x
            gy = goal_pose.pose.position.y
            gpx = odom_to_pixel(gx, gy, info)
            snapped = snap_to_valid(gpx, grid_matrix)
            if snapped is None:
                self.get_logger().warn(f"Goal {i} unreachable (no valid snap). Skipping.")
                continue
            goal_pixels.append(snapped)
            valid_goal_indices.append(i)
            snapped_goals.append(snapped)

        if not goal_pixels:
            self.get_logger().error("No valid goals after snapping.")
            return response

        # 5. NEAREST NEIGHBOR ORDERING + A*
        ordered_indices, ordered_pixels, paths = nearest_neighbor_ordering(
            start_px, goal_pixels, grid_matrix
        )

        # 6. BUILD RESPONSE
        now = self.get_clock().now().to_msg()
        frame = self.latest_costmap.header.frame_id

        for seg_idx, (orig_idx, path_px) in enumerate(zip(ordered_indices, paths)):
            # --- Path ROS msg per questo segmento ---
            ros_path = Path()
            ros_path.header.stamp = now
            ros_path.header.frame_id = frame

            for (row, col) in path_px:
                ox, oy = pixel_to_odom(row, col, info)
                ps = PoseStamped()
                ps.header.stamp = now
                ps.header.frame_id = frame
                ps.pose.position.x = ox
                ps.pose.position.y = oy
                ps.pose.position.z = 0.0
                ps.pose.orientation.w = 1.0
                ros_path.poses.append(ps)

            response.paths.append(ros_path)

            # --- Goal ordinato corrispondente ---
            row, col = ordered_pixels[seg_idx]

            gx, gy = pixel_to_odom(row, col, info)

            projected_goal = PoseStamped()
            projected_goal.header.stamp = now
            projected_goal.header.frame_id = frame

            projected_goal.pose.position.x = gx
            projected_goal.pose.position.y = gy
            projected_goal.pose.position.z = 0.0
            projected_goal.pose.orientation.w = 1.0

            response.ordered_goals.append(projected_goal)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.get_logger().info(
            f"Plan complete: {len(response.ordered_goals)}/{len(request.goals)} goals reached."
        )

        self.get_logger().info(
            f"Planning execution time: {elapsed_ms:.2f} ms"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = AStarRefService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()