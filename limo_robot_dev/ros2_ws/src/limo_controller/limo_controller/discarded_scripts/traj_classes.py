import math
from collections import deque
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple


class TrajectorySeed:
    _coords = []
    _all_seeds = []
    _terminal_seeds = set()
    _kd_tree = None
    
    # Class-level counter to automatically generate unique path IDs for root seeds
    _path_id_counter = 0

    def __init__(
        self, x: float, y: float, yaw: float, spread_dir: str = "open", path_id: int = None
    ):
        """Initializes the seed and updates the spatial registry.

        spread_dir can be: 'open', 'up', or 'down'.
        path_id is inherited by children or auto-generated for root seeds.
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.spread_dir = spread_dir  # 'open', 'up', 'down'
        self._is_terminal = False

        # Assign path_id: inherit from parent or generate a new unique one if it's a root seed
        if path_id is None:
            self.path_id = TrajectorySeed._path_id_counter
            TrajectorySeed._path_id_counter += 1
        else:
            self.path_id = path_id

        # 'open' spawns up to 2 children (up and down).
        # 'up' and 'down' propagate their own branch (1 child at a time).
        self.max_children = 2 if spread_dir == "open" else 1
        self.children_count = 0

        TrajectorySeed._coords.append([x, y])
        TrajectorySeed._all_seeds.append(self)

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, value: bool):
        if self._is_terminal == value:
            return
        self._is_terminal = value
        if self._is_terminal:
            TrajectorySeed._terminal_seeds.add(self)
        else:
            TrajectorySeed._terminal_seeds.discard(self)

    def multiply(
        self,
        cond_func,
        dist: float,
        eps: float,
        delta: float = 0.5,
        num_rays: int = 50,
    ):
        """Raycasts an angular sector using numpy to find the first valid intersection with cond_func."""
        if self.children_count >= self.max_children:
            self.is_terminal = False
            return None

        # 1. Determine the angular boundaries based on spread_dir and current child being generated
        if self.spread_dir == "open":
            if self.children_count == 0:
                min_angle = self.yaw - delta
                max_angle = self.yaw + delta
                next_spread_dir = "up"
            else:
                min_angle = self.yaw + math.pi - delta
                max_angle = self.yaw + math.pi + delta
                next_spread_dir = "down"
        elif self.spread_dir == "up":
            min_angle = self.yaw - delta
            max_angle = self.yaw + delta
            next_spread_dir = "up"
        elif self.spread_dir == "down":
            min_angle = self.yaw - delta
            max_angle = self.yaw + delta
            next_spread_dir = "down"

        # 2. Quantize the angular sector using NumPy
        angles = np.linspace(min_angle, max_angle, num_rays)

        candidates_x = self.x + dist * np.cos(angles)
        candidates_y = self.y + dist * np.sin(angles)

        # 3. Evaluate cond_func on all candidates
        v_cond_func = np.vectorize(cond_func)
        cond_mask = v_cond_func(candidates_x, candidates_y)

        # Find indices where cond is True
        valid_indices = np.where(cond_mask)[0]

        if valid_indices.size == 0:
            self.is_terminal = True
            return None

        # 4. Proximity check via KD-Tree for valid candidates
        for idx in valid_indices:
            target_x = candidates_x[idx]
            target_y = candidates_y[idx]
            target_angle = angles[idx]

            if TrajectorySeed._kd_tree is not None:
                neighbors = TrajectorySeed._kd_tree.query_ball_point(
                    [target_x, target_y], eps
                )
                if neighbors:
                    continue  # Skip this ray, it's too close to existing seeds

            # Success: update parent state
            self.children_count += 1
            self.is_terminal = False

            # Spawn child inheriting the SAME path_id
            return TrajectorySeed(
                target_x, target_y, target_angle, spread_dir=next_spread_dir, path_id=self.path_id
            )

        self.is_terminal = True
        return None

    @classmethod
    def revive_all_seeds(
        cls,
        cond_func,
        dist: float,
        eps: float,
        delta: float = 0.5,
        num_rays: int = 50,
    ) -> int:
        """ROS optimized: Generational wave propagation handling numpy-quantized raycasting."""
        if not cls._terminal_seeds:
            return 0

        propagation_queue = deque(list(cls._terminal_seeds))
        total_new_seeds = 0

        while propagation_queue:
            generation_size = len(propagation_queue)
            new_seeds_in_this_generation = []

            for _ in range(generation_size):
                parent_seed = propagation_queue.popleft()

                new_seed = parent_seed.multiply(
                    cond_func, dist, eps, delta, num_rays
                )

                if new_seed:
                    total_new_seeds += 1
                    new_seeds_in_this_generation.append(new_seed)

                    if parent_seed.children_count < parent_seed.max_children:
                        propagation_queue.append(parent_seed)

            if new_seeds_in_this_generation:
                cls._kd_tree = KDTree(cls._coords)
                propagation_queue.extend(new_seeds_in_this_generation)

        return total_new_seeds
    
    @classmethod
    def clear_all_seeds(cls):
        cls._coords = []
        cls._all_seeds = []
        cls._terminal_seeds = set()
        cls._kd_tree = None
        cls._path_id_counter = 0


class LeftLane(TrajectorySeed):

    def __init__(
        self,
        x: float,
        y: float,
        yaw_input: float,
        robot_pos: Tuple[float, float, float],
        path_id: int = None
    ):
        """Initializes a LeftLane seed."""
        robot_x, robot_y, _ = robot_pos

        v_x = x - robot_x
        v_y = y - robot_y

        u_x = math.cos(yaw_input)
        u_y = math.sin(yaw_input)

        dot_product = v_x * u_x + v_y * u_y

        if dot_product > 0:
            final_yaw = yaw_input + math.pi
        else:
            final_yaw = yaw_input

        final_yaw = math.atan2(math.sin(final_yaw), math.cos(final_yaw))

        super().__init__(x, y, final_yaw, spread_dir="open", path_id=path_id)


class RightLane(TrajectorySeed):

    def __init__(
        self,
        x: float,
        y: float,
        yaw_input: float,
        robot_pos: Tuple[float, float, float],
        path_id: int = None
    ):
        """Initializes a RightLane seed."""
        robot_x, robot_y, _ = robot_pos

        v_x = x - robot_x
        v_y = y - robot_y

        u_x = math.cos(yaw_input)
        u_y = math.sin(yaw_input)

        dot_product = v_x * u_x + v_y * u_y

        if dot_product < 0:
            final_yaw = yaw_input + math.pi
        else:
            final_yaw = yaw_input

        final_yaw = math.atan2(math.sin(final_yaw), math.cos(final_yaw))

        super().__init__(x, y, final_yaw, spread_dir="open", path_id=path_id)


class UniqueLane(TrajectorySeed):

    def __init__(
        self,
        x: float,
        y: float,
        yaw_input: float,
        robot_pos: Tuple[float, float, float],
        path_id: int = None
    ):
        """Initializes a UniqueLane seed."""
        robot_x, robot_y, _ = robot_pos

        v_x = x - robot_x
        v_y = y - robot_y

        u_x = math.cos(yaw_input)
        u_y = math.sin(yaw_input)

        dot_product = v_x * u_x + v_y * u_y

        if dot_product < 0:
            final_yaw = yaw_input + math.pi
        else:
            final_yaw = yaw_input

        final_yaw = math.atan2(math.sin(final_yaw), math.cos(final_yaw))

        super().__init__(x, y, final_yaw, spread_dir="open", path_id=path_id)

