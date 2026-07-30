import math
import numpy as np
import pygame
from config import *


class Car:
    def __init__(self, start_pos, start_angle, car_id=0):
        self.start_pos = start_pos
        self.start_angle = start_angle
        self.car_id = car_id
        self.color = CAR_COLORS[car_id % len(CAR_COLORS)]
        self.reset()

    def reset(self):
        self.x, self.y = self.start_pos
        self.angle = self.start_angle
        self.speed = 0.0
        self.alive = True

        self.ray_dists = [1.0] * NUM_RAYS
        self.ray_hits = [(self.x, self.y)] * NUM_RAYS

        # Checkpoint / lap tracking
        self.next_cp = 1          # start past CP 0 (the finish line)
        self.lap = 0
        self.total_cps = 0

        # Timing
        self.lap_time = 0.0
        self.best_time = float('inf')
        self.total_time = 0.0

        self.prev_pos = self.start_pos
        self.steps = 0

        # Angular velocity tracking (for state encoding)
        self._prev_angle = self.start_angle
        self.angular_velocity = 0.0   # degrees/step, normalized in get_state

        # Trail for visual effect
        self.trail = []

    # ── physics ──────────────────────────────────────────────

    def update(self, steering, throttle, dt=1.0):
        if not self.alive:
            return
        self.prev_pos = (self.x, self.y)

        # Steering scales with speed so you can't spin in place
        speed_fac = max(self.speed / MAX_SPEED, 0.1)
        self.angle += steering * TURN_RATE * speed_fac * dt

        # Throttle / brake
        if throttle > 0:
            self.speed += throttle * ACCELERATION * dt
        else:
            self.speed += throttle * BRAKE_FORCE * dt

        self.speed -= FRICTION * self.speed * dt
        self.speed = max(0.0, min(self.speed, MAX_SPEED))

        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed * dt
        self.y += math.sin(rad) * self.speed * dt

        # Track angular velocity so the model knows it's rotating
        raw_delta = self.angle - self._prev_angle
        # Unwrap to [-180, 180]
        while raw_delta > 180:  raw_delta -= 360
        while raw_delta < -180: raw_delta += 360
        self.angular_velocity = raw_delta          # degrees/step
        self._prev_angle = self.angle

        self.steps += 1
        self.lap_time += dt / FPS
        self.total_time += dt / FPS

        # Trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > 30:
            self.trail.pop(0)

    # ── sensors ──────────────────────────────────────────────

    def cast_rays(self, track):
        origin = (self.x, self.y)
        start = self.angle - RAY_SPREAD / 2
        step = RAY_SPREAD / (NUM_RAYS - 1) if NUM_RAYS > 1 else 0

        for i in range(NUM_RAYS):
            d, hp = track.cast_ray(origin, start + i * step)
            self.ray_dists[i] = d / MAX_RAY_LENGTH
            self.ray_hits[i] = hp

    # ── collision / checkpoints ──────────────────────────────

    def get_corners(self):
        rad = math.radians(self.angle)
        c, s = math.cos(rad), math.sin(rad)
        hw, hh = CAR_LENGTH / 2, CAR_WIDTH / 2
        return [
            (self.x + c * hw - s * hh, self.y + s * hw + c * hh),
            (self.x + c * hw + s * hh, self.y + s * hw - c * hh),
            (self.x - c * hw + s * hh, self.y - s * hw - c * hh),
            (self.x - c * hw - s * hh, self.y - s * hw + c * hh),
        ]

    def check_collision(self, track):
        if not self.alive:
            return False
        if track.check_collision(self.get_corners()):
            # Teleport back to midpoint of last checkpoint, face next one
            last_cp_idx = (self.next_cp - 1) % len(track.checkpoints)
            cp_a, cp_b = track.checkpoints[last_cp_idx]
            self.x = (cp_a[0] + cp_b[0]) / 2
            self.y = (cp_a[1] + cp_b[1]) / 2
            # Face toward midpoint of next checkpoint
            ncp_a, ncp_b = track.checkpoints[self.next_cp]
            nx = (ncp_a[0] + ncp_b[0]) / 2
            ny = (ncp_a[1] + ncp_b[1]) / 2
            self.angle = math.degrees(math.atan2(ny - self.y, nx - self.x))
            self.speed = 0.0
            self.prev_pos = (self.x, self.y)
            return True
        return False

    def check_checkpoints(self, track):
        """Returns checkpoint reward earned this step."""
        if not self.alive:
            return 0.0

        curr = (self.x, self.y)
        if not track.check_checkpoint(self.prev_pos, curr, self.next_cp):
            return 0.0

        self.total_cps += 1
        reward = CHECKPOINT_REWARD

        # Speed bonus at checkpoint — rewards carrying speed through corners
        reward += (self.speed / MAX_SPEED) * CP_SPEED_BONUS_SCALE

        self.next_cp = (self.next_cp + 1) % len(track.checkpoints)

        # Crossed CP 0 → lap complete (next_cp wraps back to 1)
        if self.next_cp == 1:
            self.lap += 1
            reward += LAP_REWARD
            if self.lap_time < self.best_time:
                self.best_time = self.lap_time
            self.lap_time = 0.0

        return reward

    def get_shaping_reward(self, track):
        """Per-step reward shaping to improve racing line and wall avoidance.

        Returns a small float added to the base reward each step.
        Kept separate so it's easy to tune without touching checkpoint logic.
        """
        if not self.alive:
            return 0.0

        shaping = 0.0

        # 1. Wall proximity penalty — proportional to how close the nearest wall is.
        #    min_ray == 0 → very close; min_ray == 1 → max range.
        #    We penalise being within 30% of max range.
        min_ray = min(self.ray_dists)
        if min_ray < 0.3:
            shaping += WALL_PROXIMITY_PENALTY * (1.0 - min_ray / 0.3)

        # 2. Lookahead alignment bonus — reward heading toward the checkpoint
        #    *after* the next one, not just the immediate one. This nudges the model
        #    to start positioning for upcoming turns (partial racing-line incentive).
        n_cps = len(track.checkpoints)
        ahead_idx = (self.next_cp + 1) % n_cps
        cp_a, cp_b = track.checkpoints[ahead_idx]
        mid = ((cp_a[0] + cp_b[0]) / 2, (cp_a[1] + cp_b[1]) / 2)
        dx, dy = mid[0] - self.x, mid[1] - self.y
        target = math.degrees(math.atan2(dy, dx))
        diff = abs(target - self.angle)
        while diff > 180: diff -= 360
        diff = abs(diff)
        # diff == 0° → full bonus; diff == 90° → 0 bonus; > 90° → 0
        if diff < 90:
            shaping += LOOKAHEAD_ALIGN_REWARD * (1.0 - diff / 90.0)

        return shaping

    # ── state for RL ─────────────────────────────────────────

    def get_state(self, track):
        # ── 1. LiDAR raycasts (NUM_RAYS floats, each 0-1) ──────
        state = list(self.ray_dists)

        # ── 2. Kinematics ──────────────────────────────────────
        state.append(self.speed / MAX_SPEED)               # normalized speed
        state.append(self.angular_velocity / (TURN_RATE * MAX_SPEED / MAX_SPEED))  # norm angular vel

        # ── 3. Lookahead: angles to next LOOKAHEAD_CPS checkpoints ──
        # Encoding multiple checkpoint angles gives the model a "plan" rather
        # than just reacting to immediate wall distances.
        n_cps = len(track.checkpoints)
        for k in range(LOOKAHEAD_CPS):
            cp_idx = (self.next_cp + k) % n_cps
            cp_a, cp_b = track.checkpoints[cp_idx]
            mid = ((cp_a[0] + cp_b[0]) / 2, (cp_a[1] + cp_b[1]) / 2)
            dx, dy = mid[0] - self.x, mid[1] - self.y
            target = math.degrees(math.atan2(dy, dx))
            diff = target - self.angle
            while diff > 180:  diff -= 360
            while diff < -180: diff += 360
            state.append(diff / 180.0)                     # 1 float each

        # ── 4. Proximity to nearest wall (from min ray dist) ───
        min_ray = min(self.ray_dists)
        state.append(min_ray)                              # 1 float — also useful for reward shaping

        # Total state size = NUM_RAYS + 2 + LOOKAHEAD_CPS + 1
        return np.array(state, dtype=np.float32)

    # ── drawing ──────────────────────────────────────────────

    def draw(self, surface, draw_rays=True):
        if not self.alive:
            return

        # Trail (exhaust dots)
        for i, (tx, ty) in enumerate(self.trail):
            alpha = int(180 * i / max(len(self.trail), 1))
            r = min(255, self.color[0])
            g = min(255, self.color[1] // 3)
            pygame.draw.circle(surface, (r, g, 0), (int(tx), int(ty)),
                               max(1, 2 * i // len(self.trail) + 1))

        # Sensor rays
        if draw_rays:
            for dist, hit in zip(self.ray_dists, self.ray_hits):
                r = int(255 * (1 - dist))
                g = int(255 * dist)
                pygame.draw.line(surface, (r, g, 0),
                                 (int(self.x), int(self.y)),
                                 (int(hit[0]), int(hit[1])), 1)
                pygame.draw.circle(surface, (r, g, 0),
                                   (int(hit[0]), int(hit[1])), 3)

        # Car body
        corners = [(int(c[0]), int(c[1])) for c in self.get_corners()]
        pygame.draw.polygon(surface, self.color, corners)
        pygame.draw.polygon(surface, WHITE, corners, 1)
