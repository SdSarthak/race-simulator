import math

import numpy as np

from config import *

_EPS = 1e-10


def line_intersect(p1, p2, p3, p4):
    """Line segment intersection. Returns (x, y, t) or None.
    t is the parameter along (p1->p2).

    Reference implementation: `Track` uses a vectorised equivalent internally,
    and the two are kept in agreement by the test suite.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < _EPS:
        return None

    t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    u = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1), t)
    return None


# ── layouts ──────────────────────────────────────────────────
# Each layout returns (outer, inner, start_pos, start_angle).


def _layout_circuit():
    """Default circuit — large layout for a 1400x900 window.
    Road width ~120px so a 40-car field fits side by side."""

    outer = [
        (120,  60),   # 0  top-left
        (560,  60),   # 1  top straight mid
        (950,  60),   # 2  top-right entry
        (1150, 100),  # 3  turn 1 entry
        (1260, 240),  # 4  right upper
        (1280, 420),  # 5  right mid
        (1200, 510),  # 6  chicane out
        (1280, 600),  # 7  chicane back
        (1280, 750),  # 8  right lower
        (1150, 840),  # 9  bottom-right
        (820,  840),  # 10 bottom-mid-right
        (480,  840),  # 11 bottom-mid-left
        (200,  840),  # 12 bottom-left
        (80,   750),  # 13 left lower turn
        (60,   570),  # 14 left mid
        (60,   220),  # 15 left upper
        (80,   100),  # 16 top-left turn
    ]

    inner = [
        (240,  180),  # 0
        (550,  180),  # 1
        (890,  180),  # 2
        (1040, 215),  # 3
        (1130, 300),  # 4
        (1150, 420),  # 5
        (1085, 500),  # 6
        (1150, 585),  # 7
        (1150, 700),  # 8
        (1060, 720),  # 9
        (820,  720),  # 10
        (480,  720),  # 11
        (270,  720),  # 12
        (210,  660),  # 13
        (190,  540),  # 14
        (190,  280),  # 15
        (220,  210),  # 16
    ]

    # Start between CP 0 and CP 1, facing right
    return outer, inner, (420, 120), 0.0


def _layout_oval(n_points=20, road_width=120):
    """Procedural oval — a simple, smooth circuit used for quick experiments."""
    cx, cy = WIDTH / 2, HEIGHT / 2
    rx, ry = WIDTH / 2 - 80, HEIGHT / 2 - 80

    outer, inner = [], []
    for i in range(n_points):
        theta = 2 * math.pi * i / n_points
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        outer.append((cx + rx * cos_t, cy + ry * sin_t))
        inner.append((cx + (rx - road_width) * cos_t,
                      cy + (ry - road_width) * sin_t))

    start = ((outer[0][0] + inner[0][0]) / 2,
             (outer[0][1] + inner[0][1]) / 2)
    # Face along the direction of travel (towards checkpoint 1)
    nxt = ((outer[1][0] + inner[1][0]) / 2,
           (outer[1][1] + inner[1][1]) / 2)
    angle = math.degrees(math.atan2(nxt[1] - start[1], nxt[0] - start[0]))
    return outer, inner, start, angle


LAYOUTS = {
    "circuit": _layout_circuit,
    "oval": _layout_oval,
}


class Track:
    """Closed circuit made of two polygons: walls, checkpoints and ray casting."""

    def __init__(self, layout="circuit"):
        if layout not in LAYOUTS:
            raise ValueError(
                f"unknown layout {layout!r}; choose from {sorted(LAYOUTS)}")
        self.layout = layout
        self.walls = []         # [(p1, p2, color), ...]
        self.checkpoints = []   # [(inner_pt, outer_pt), ...]
        self.start_pos = (0, 0)
        self.start_angle = 0.0
        self._build()

    def _build(self):
        outer, inner, start_pos, start_angle = LAYOUTS[self.layout]()
        n = len(outer)
        if n != len(inner):
            raise ValueError("outer and inner boundaries must have equal length")

        # Store outer/inner for filled road rendering
        self.outer = outer
        self.inner = inner
        self.start_pos = start_pos
        self.start_angle = start_angle

        # Colour each quarter of the lap differently so progress is readable
        def wall_color(i):
            return WALL_COLORS[int(i * len(WALL_COLORS) / n) % len(WALL_COLORS)]

        for i in range(n):
            j = (i + 1) % n
            c = wall_color(i)
            self.walls.append((outer[i], outer[j], c))
            self.walls.append((inner[i], inner[j], c))

        # Checkpoints: line from inner[i] to outer[i]
        for i in range(n):
            self.checkpoints.append((inner[i], outer[i]))

        self._cache_wall_arrays()

    # ── vectorised geometry ──────────────────────────────────

    def _cache_wall_arrays(self):
        """Pack wall endpoints into arrays so ray casts hit numpy, not Python loops."""
        starts = np.array([w[0] for w in self.walls], dtype=np.float64)
        ends = np.array([w[1] for w in self.walls], dtype=np.float64)
        self._wx = starts[:, 0]
        self._wy = starts[:, 1]
        self._wdx = ends[:, 0] - starts[:, 0]
        self._wdy = ends[:, 1] - starts[:, 1]

    def _batch_hits(self, starts, ends):
        """Crossing parameters for K segments against every wall.

        Returns a (K, W) array holding t along each segment where it crosses
        wall w, and `inf` where it misses. Parallel segments are never hits, so
        the denominator is clamped rather than guarded — that keeps the whole
        thing branch-free and out of numpy's error machinery.
        """
        starts = np.asarray(starts, dtype=np.float64).reshape(-1, 2)
        ends = np.asarray(ends, dtype=np.float64).reshape(-1, 2)
        dx = (ends[:, 0] - starts[:, 0])[:, None]
        dy = (ends[:, 1] - starts[:, 1])[:, None]
        rx = starts[:, 0][:, None] - self._wx[None, :]
        ry = starts[:, 1][:, None] - self._wy[None, :]

        denom = dx * self._wdy[None, :] - dy * self._wdx[None, :]
        ok = np.abs(denom) > _EPS
        safe = np.where(ok, denom, 1.0)

        t = (self._wdx[None, :] * ry - self._wdy[None, :] * rx) / safe
        u = (dx * ry - dy * rx) / safe

        valid = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        return np.where(valid, t, np.inf)

    def cast_ray(self, origin, angle_deg):
        """Cast ray from origin at angle. Returns (distance, hit_point)."""
        dists, hits = self.cast_rays(origin, (angle_deg,))
        return float(dists[0]), (float(hits[0][0]), float(hits[0][1]))

    def cast_rays(self, origin, angles_deg):
        """Cast a whole fan at once. Returns (distances, hit_points array).

        All rays share an origin, so the wall tests collapse into two (R x W)
        array operations instead of R x W python-level intersections.
        """
        ang = np.radians(np.asarray(angles_deg, dtype=np.float64))
        dx = np.cos(ang) * MAX_RAY_LENGTH
        dy = np.sin(ang) * MAX_RAY_LENGTH

        x1, y1 = origin
        rx = x1 - self._wx                      # (W,)
        ry = y1 - self._wy

        denom = np.outer(dx, self._wdy) - np.outer(dy, self._wdx)   # (R, W)
        ok = np.abs(denom) > _EPS
        safe = np.where(ok, denom, 1.0)

        t = (self._wdx * ry - self._wdy * rx)[None, :] / safe       # (R, W)
        u = (np.outer(dx, ry) - np.outer(dy, rx)) / safe

        valid = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        t_min = np.where(valid, t, np.inf).min(axis=1)
        t_hit = np.where(np.isfinite(t_min), t_min, 1.0)

        hits = np.stack([x1 + t_hit * dx, y1 + t_hit * dy], axis=1)
        return t_hit * MAX_RAY_LENGTH, hits

    def check_collision(self, corners):
        """Check if any edge of a closed polygon (e.g. a car's 4 corners) hits a wall."""
        pts = np.asarray(corners, dtype=np.float64)
        return bool(np.isfinite(self._batch_hits(pts, np.roll(pts, -1, axis=0))).any())

    def check_checkpoint(self, prev_pos, curr_pos, cp_idx):
        """Check if movement from prev to curr crosses checkpoint cp_idx."""
        a, b = self.checkpoints[cp_idx % len(self.checkpoints)]
        return line_intersect(prev_pos, curr_pos, a, b) is not None

    # ── helpers ──────────────────────────────────────────────

    def checkpoint_midpoint(self, idx):
        a, b = self.checkpoints[idx % len(self.checkpoints)]
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    def checkpoint_width(self, idx):
        a, b = self.checkpoints[idx % len(self.checkpoints)]
        return math.hypot(b[0] - a[0], b[1] - a[1])
