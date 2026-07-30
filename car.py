import math

import numpy as np

from config import *

# Ray angles relative to the car's heading, computed once.
RAY_OFFSETS = (np.linspace(-RAY_SPREAD / 2, RAY_SPREAD / 2, NUM_RAYS)
               if NUM_RAYS > 1 else np.zeros(1))


class Car:
    """A single agent-controlled car: physics, LiDAR sensing and lap telemetry.

    The car owns everything the learner needs to score it: checkpoint counts,
    lap/sector times, speed at each checkpoint and the number of wall contacts.
    """

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
        self.death_reason = None

        self.ray_dists = [1.0] * NUM_RAYS
        self.ray_hits = [(self.x, self.y)] * NUM_RAYS

        # Checkpoint / lap tracking
        self.next_cp = 1          # start past CP 0 (the finish line)
        self.lap = 0
        self.total_cps = 0
        self.num_cps = 0          # filled in on first track interaction

        # Timing
        self.lap_time = 0.0
        self.best_time = float('inf')
        self.total_time = 0.0

        self.prev_pos = self.start_pos
        self.steps = 0

        # Angular velocity tracking (for state encoding)
        self._prev_angle = self.start_angle
        self.angular_velocity = 0.0   # degrees/step, normalized in get_state

        # Telemetry consumed by the phase-2 fitness function
        self.cp_speeds = []       # speed carried through each checkpoint
        self.cp_splits = []       # steps taken between consecutive checkpoints
        self.wall_hits = 0
        self.sector_wall_hits = 0  # hits since the last checkpoint
        self.total_reward = 0.0
        self._last_cp_step = 0
        self._idle_steps = 0

        # Trail for visual effect
        self.trail = []

    # ── physics ──────────────────────────────────────────────

    def update(self, steering, throttle, dt=1.0):
        if not self.alive:
            return
        self.prev_pos = (self.x, self.y)

        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))

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
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

    def step(self, track, steering, throttle, dt=1.0):
        """Advance one tick and return the reward earned.

        This is the single entry point used by both the headless trainer and
        the renderer, so physics, sensing and scoring can never drift apart.
        """
        if not self.alive:
            return 0.0

        self.update(steering, throttle, dt)

        reward = 0.0
        if self.check_collision(track):
            reward += CRASH_PENALTY
        # Sense after any crash teleport, so the state fed to the policy on the
        # next step describes where the car actually is.
        self.cast_rays(track)
        reward += self.check_checkpoints(track)
        reward += self.get_shaping_reward(track)

        # Speed / idling signals
        reward += SPEED_REWARD * (self.speed / MAX_SPEED)
        if self.speed < IDLE_SPEED:
            reward += IDLE_PENALTY
            self._idle_steps += 1
        else:
            self._idle_steps = 0

        self._check_retirement()
        self.total_reward += reward
        return reward

    def _check_retirement(self):
        """Retire cars that are stuck, stalled or hopeless so a generation ends."""
        if not self.alive:
            return
        if self.sector_wall_hits >= SECTOR_WALL_HITS:
            self.alive = False
            self.death_reason = "wall_hits"
        elif self._idle_steps >= IDLE_LIMIT_STEPS:
            self.alive = False
            self.death_reason = "idle"
        elif self.steps - self._last_cp_step >= STALL_LIMIT_STEPS:
            self.alive = False
            self.death_reason = "no_progress"

    # ── sensors ──────────────────────────────────────────────

    def cast_rays(self, track):
        dists, hits = track.cast_rays((self.x, self.y), self.angle + RAY_OFFSETS)
        self.ray_dists = (dists / MAX_RAY_LENGTH).tolist()
        self.ray_hits = [tuple(p) for p in hits.tolist()]

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
            self.wall_hits += 1
            self.sector_wall_hits += 1
            # Teleport back to midpoint of last checkpoint, face next one
            n_cps = len(track.checkpoints)
            last_cp_idx = (self.next_cp - 1) % n_cps
            cp_a, cp_b = track.checkpoints[last_cp_idx]
            self.x = (cp_a[0] + cp_b[0]) / 2
            self.y = (cp_a[1] + cp_b[1]) / 2
            # Face toward midpoint of next checkpoint
            ncp_a, ncp_b = track.checkpoints[self.next_cp % n_cps]
            nx = (ncp_a[0] + ncp_b[0]) / 2
            ny = (ncp_a[1] + ncp_b[1]) / 2
            self.angle = math.degrees(math.atan2(ny - self.y, nx - self.x))
            self._prev_angle = self.angle
            self.angular_velocity = 0.0
            self.speed = 0.0
            self.prev_pos = (self.x, self.y)
            self.trail.clear()
            return True
        return False

    def check_checkpoints(self, track):
        """Returns checkpoint reward earned this step."""
        if not self.alive:
            return 0.0

        n_cps = len(track.checkpoints)
        self.num_cps = n_cps

        curr = (self.x, self.y)
        if not track.check_checkpoint(self.prev_pos, curr, self.next_cp):
            return 0.0

        self.total_cps += 1
        reward = CHECKPOINT_REWARD

        # Speed bonus at checkpoint — rewards carrying speed through corners
        reward += (self.speed / MAX_SPEED) * CP_SPEED_BONUS_SCALE

        # Telemetry: speed carried through, and how long that sector took
        self.cp_speeds.append(self.speed)
        self.cp_splits.append(max(1, self.steps - self._last_cp_step))
        self._last_cp_step = self.steps
        self.sector_wall_hits = 0      # progress earns a clean slate

        self.next_cp = (self.next_cp + 1) % n_cps

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
        if min_ray < WALL_PROXIMITY_BAND:
            shaping += WALL_PROXIMITY_PENALTY * (1.0 - min_ray / WALL_PROXIMITY_BAND)

        # 2. Lookahead alignment bonus — reward heading toward the checkpoint
        #    *after* the next one, not just the immediate one. This nudges the model
        #    to start positioning for upcoming turns (partial racing-line incentive).
        ahead_idx = (self.next_cp + 1) % len(track.checkpoints)
        diff = abs(self.angle_to_checkpoint(track, ahead_idx))
        # diff == 0° → full bonus; diff == 90° → 0 bonus; > 90° → 0
        if diff < 90:
            shaping += LOOKAHEAD_ALIGN_REWARD * (1.0 - diff / 90.0)

        return shaping

    # ── geometry helpers ─────────────────────────────────────

    def angle_to_checkpoint(self, track, cp_idx):
        """Signed heading error, in degrees within [-180, 180], to a checkpoint."""
        cp_a, cp_b = track.checkpoints[cp_idx % len(track.checkpoints)]
        mid_x = (cp_a[0] + cp_b[0]) / 2
        mid_y = (cp_a[1] + cp_b[1]) / 2
        target = math.degrees(math.atan2(mid_y - self.y, mid_x - self.x))
        diff = target - self.angle
        while diff > 180:  diff -= 360
        while diff < -180: diff += 360
        return diff

    # ── state for RL ─────────────────────────────────────────

    def get_state(self, track):
        # ── 1. LiDAR raycasts (NUM_RAYS floats, each 0-1) ──────
        state = list(self.ray_dists)

        # ── 2. Kinematics ──────────────────────────────────────
        state.append(self.speed / MAX_SPEED)               # normalized speed
        state.append(self.angular_velocity / TURN_RATE)    # normalized angular vel

        # ── 3. Lookahead: angles to next LOOKAHEAD_CPS checkpoints ──
        # Encoding multiple checkpoint angles gives the model a "plan" rather
        # than just reacting to immediate wall distances.
        for k in range(LOOKAHEAD_CPS):
            diff = self.angle_to_checkpoint(track, self.next_cp + k)
            state.append(diff / 180.0)                     # 1 float each

        # ── 4. Proximity to nearest wall (from min ray dist) ───
        state.append(min(self.ray_dists))                  # also used for shaping

        # Total state size = NUM_RAYS + 2 + LOOKAHEAD_CPS + 1
        return np.array(state, dtype=np.float32)

    # ── reporting ────────────────────────────────────────────

    def telemetry(self):
        """Flat dict of everything worth logging about this car."""
        return {
            "car_id": self.car_id,
            "alive": self.alive,
            "death_reason": self.death_reason,
            "laps": self.lap,
            "checkpoints": self.total_cps,
            "steps": self.steps,
            "wall_hits": self.wall_hits,
            "best_lap_time": self.best_time,
            "total_time": self.total_time,
            "total_reward": self.total_reward,
            "avg_cp_speed": (sum(self.cp_speeds) / len(self.cp_speeds)
                             if self.cp_speeds else 0.0),
            "avg_split_steps": (sum(self.cp_splits) / len(self.cp_splits)
                                if self.cp_splits else 0.0),
        }
