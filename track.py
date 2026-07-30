import math
import pygame
from config import *


def line_intersect(p1, p2, p3, p4):
    """Line segment intersection. Returns (x, y, t) or None.
    t is the parameter along (p1->p2)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-10:
        return None

    t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    u = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1), t)
    return None


class Track:
    def __init__(self):
        self.walls = []         # [(p1, p2, color), ...]
        self.checkpoints = []   # [(p1, p2), ...]
        self.start_pos = (0, 0)
        self.start_angle = 0.0
        self._build()

    def _build(self):
        """Build the default circuit — large layout for 1400×900 window.
        Road width ~120px so 40 cars fit comfortably side by side."""

        # Outer boundary (clockwise, fits within 1400×900 with ~60px margin)
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

        # Inner boundary (clockwise, ~120px inside outer — wide road)
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

        n = len(outer)

        # Color assignment per section
        def wall_color(i):
            if i <= 2:
                return YELLOW     # top straight
            if i <= 7:
                return GREEN      # right side + chicane
            if i <= 12:
                return BLUE       # bottom
            return RED            # left side

        # Store outer/inner for filled road rendering
        self.outer = outer
        self.inner = inner

        # Build wall segments
        for i in range(n):
            j = (i + 1) % n
            c = wall_color(i)
            self.walls.append((outer[i], outer[j], c))
            self.walls.append((inner[i], inner[j], c))

        # Checkpoints: line from inner[i] to outer[i]
        for i in range(n):
            self.checkpoints.append((inner[i], outer[i]))

        # Start between CP 0 and CP 1, facing right
        self.start_pos = (420, 120)
        self.start_angle = 0.0

    def cast_ray(self, origin, angle_deg):
        """Cast ray from origin at angle. Returns (distance, hit_point)."""
        rad = math.radians(angle_deg)
        end = (origin[0] + math.cos(rad) * MAX_RAY_LENGTH,
               origin[1] + math.sin(rad) * MAX_RAY_LENGTH)

        closest = MAX_RAY_LENGTH
        hit = end

        for w_start, w_end, _ in self.walls:
            result = line_intersect(origin, end, w_start, w_end)
            if result:
                dist = result[2] * MAX_RAY_LENGTH
                if dist < closest:
                    closest = dist
                    hit = (result[0], result[1])

        return closest, hit

    def check_collision(self, corners):
        """Check if any edge of a rectangle (4 corners) hits a wall."""
        for i in range(4):
            j = (i + 1) % 4
            for ws, we, _ in self.walls:
                if line_intersect(corners[i], corners[j], ws, we):
                    return True
        return False

    def check_checkpoint(self, prev_pos, curr_pos, cp_idx):
        """Check if movement from prev to curr crosses checkpoint cp_idx."""
        a, b = self.checkpoints[cp_idx]
        return line_intersect(prev_pos, curr_pos, a, b) is not None

    def draw(self, surface):
        """Draw track walls and start/finish line."""
        for ws, we, color in self.walls:
            pygame.draw.line(surface, color, ws, we, 3)

        # Start/finish line
        if self.checkpoints:
            a, b = self.checkpoints[0]
            pygame.draw.line(surface, RED, a, b, 2)

    def draw_next_checkpoint(self, surface, idx):
        """Draw the next checkpoint as a faint white line."""
        if 0 <= idx < len(self.checkpoints):
            a, b = self.checkpoints[idx]
            pygame.draw.line(surface, (60, 60, 60), a, b, 1)
