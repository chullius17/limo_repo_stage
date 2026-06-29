#!/usr/bin/env python3

import math

# ─────────────────────────────────────────────
#  PURSUIT POINT
# ─────────────────────────────────────────────

class PursuitPoint:
    """
    Punto cinematico che avanza lungo una lista di waypoint (x,y)
    a velocità lineare costante `speed` [m/s].
    Chiamare .advance(dt) ad ogni tick; leggere .x, .y e .done.
    """
    def __init__(self, waypoints: list, speed: float):
        # waypoints: [(x,y), ...]
        self.waypoints = waypoints
        self.speed     = speed          # m/s
        self.seg_idx   = 0             # segmento corrente
        self.t_seg     = 0.0           # parametro [0,1] sul segmento corrente
        self.done      = len(waypoints) < 2

        if not self.done:
            self.x, self.y = waypoints[0]
        else:
            self.x, self.y = waypoints[-1] if waypoints else (0.0, 0.0)

    def advance(self, dt: float):
        if self.done:
            return

        remaining = self.speed * dt

        while remaining > 0.0 and self.seg_idx < len(self.waypoints) - 1:
            x0, y0 = self.waypoints[self.seg_idx]
            x1, y1 = self.waypoints[self.seg_idx + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)

            if seg_len < 1e-6:
                self.seg_idx += 1
                self.t_seg = 0.0
                continue

            # distanza rimanente nel segmento corrente
            dist_in_seg = seg_len * (1.0 - self.t_seg)

            if remaining < dist_in_seg:
                self.t_seg += remaining / seg_len
                break
            else:
                remaining -= dist_in_seg
                self.seg_idx += 1
                self.t_seg = 0.0

        if self.seg_idx >= len(self.waypoints) - 1:
            self.seg_idx = len(self.waypoints) - 2
            self.t_seg   = 1.0
            self.done    = True

        x0, y0 = self.waypoints[self.seg_idx]
        x1, y1 = self.waypoints[self.seg_idx + 1]
        self.x = x0 + self.t_seg * (x1 - x0)
        self.y = y0 + self.t_seg * (y1 - y0)