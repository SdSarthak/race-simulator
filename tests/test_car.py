import math

import numpy as np
import pytest

from config import (
    STATE_DIM, NUM_RAYS, MAX_SPEED, ACCELERATION, FRICTION, IDLE_LIMIT_STEPS,
    MAX_WALL_HITS, STALL_LIMIT_STEPS, CHECKPOINT_REWARD, LAP_REWARD,
    CRASH_PENALTY, TOTAL_LAPS,
)
from car import Car


def _place_on_checkpoint(car, track, cp_idx, offset=6.0):
    """Put the car just before checkpoint `cp_idx`, aimed across it."""
    a, b = track.checkpoints[cp_idx]
    mid = track.checkpoint_midpoint(cp_idx)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    px, py = -dy / length, dx / length
    car.prev_pos = (mid[0] - px * offset, mid[1] - py * offset)
    car.x, car.y = mid[0] + px * offset, mid[1] + py * offset
    car.next_cp = cp_idx


# ── physics ──────────────────────────────────────────────────

def test_throttle_accelerates_and_moves_the_car(car, track):
    x0 = car.x
    car.update(0.0, 1.0)
    assert car.speed == pytest.approx(ACCELERATION * (1 - FRICTION))
    assert car.x > x0


def test_speed_is_clamped_to_max(car, track):
    for _ in range(500):
        car.update(0.0, 1.0)
    assert car.speed <= MAX_SPEED
    assert car.speed == pytest.approx(MAX_SPEED, rel=0.05)


def test_speed_never_goes_negative(car, track):
    for _ in range(50):
        car.update(0.0, -1.0)
    assert car.speed == 0.0


def test_actions_outside_the_valid_range_are_clamped(track):
    wild = Car(track.start_pos, track.start_angle)
    tame = Car(track.start_pos, track.start_angle)
    for _ in range(10):
        wild.update(9.0, 9.0)
        tame.update(1.0, 1.0)
    assert wild.speed == pytest.approx(tame.speed)
    assert wild.angle == pytest.approx(tame.angle)


def test_steering_barely_turns_a_stationary_car(car):
    car.update(1.0, 0.0)
    # speed factor floors at 0.1, so a full-lock turn from a standstill is small
    assert 0 < abs(car.angle - car.start_angle) < 1.0


def test_angular_velocity_is_unwrapped(car):
    car.speed = MAX_SPEED
    car.angle = 179.0
    car._prev_angle = 179.0
    car.update(1.0, 0.0)
    assert abs(car.angular_velocity) < 180.0


def test_trail_is_bounded(car):
    for _ in range(200):
        car.update(0.0, 1.0)
    assert 0 < len(car.trail) <= 30


# ── state encoding ───────────────────────────────────────────

def test_state_shape_and_ranges(car, track):
    car.cast_rays(track)
    state = car.get_state(track)
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32
    assert np.all(state[:NUM_RAYS] >= 0) and np.all(state[:NUM_RAYS] <= 1)
    assert -1.0 <= state[NUM_RAYS] <= 1.0                 # normalised speed
    assert np.all(np.abs(state[NUM_RAYS + 2:-1]) <= 1.0)  # heading errors
    assert 0.0 <= state[-1] <= 1.0                        # nearest wall


def test_heading_error_is_zero_when_pointed_at_the_checkpoint(car, track):
    mid = track.checkpoint_midpoint(3)
    car.next_cp = 3
    car.angle = math.degrees(math.atan2(mid[1] - car.y, mid[0] - car.x))
    assert car.angle_to_checkpoint(track, 3) == pytest.approx(0.0, abs=1e-9)


def test_heading_error_is_signed_and_wrapped(car, track):
    mid = track.checkpoint_midpoint(3)
    car.angle = math.degrees(math.atan2(mid[1] - car.y, mid[0] - car.x)) + 190.0
    err = car.angle_to_checkpoint(track, 3)
    assert -180.0 <= err <= 180.0
    # target - heading is -190 degrees, which wraps to +170
    assert err == pytest.approx(170.0, abs=1e-6)


# ── checkpoints and laps ─────────────────────────────────────

def test_crossing_a_checkpoint_scores_and_advances(car, track):
    _place_on_checkpoint(car, track, 2)
    car.speed = MAX_SPEED / 2
    reward = car.check_checkpoints(track)
    assert reward >= CHECKPOINT_REWARD
    assert car.total_cps == 1
    assert car.next_cp == 3
    assert car.cp_speeds == [MAX_SPEED / 2]
    assert len(car.cp_splits) == 1


def test_completing_a_lap_awards_the_lap_bonus_and_records_the_time(car, track):
    car.lap_time = 4.0
    _place_on_checkpoint(car, track, 0)
    reward = car.check_checkpoints(track)
    assert reward >= LAP_REWARD
    assert car.lap == 1
    assert car.next_cp == 1
    assert car.best_time == pytest.approx(4.0)
    assert car.lap_time == 0.0


def test_not_crossing_anything_scores_nothing(car, track):
    car.prev_pos = (car.x, car.y)
    assert car.check_checkpoints(track) == 0.0
    assert car.total_cps == 0


# ── collisions ───────────────────────────────────────────────

def test_hitting_a_wall_teleports_to_the_last_checkpoint(car, track):
    car.next_cp = 4
    wall_start, wall_end, _ = track.walls[0]
    car.x = (wall_start[0] + wall_end[0]) / 2
    car.y = (wall_start[1] + wall_end[1]) / 2
    car.speed = 10.0

    assert car.check_collision(track) is True
    assert car.wall_hits == 1
    assert car.speed == 0.0
    assert (car.x, car.y) == pytest.approx(track.checkpoint_midpoint(3))
    # and it now faces the checkpoint it was heading for
    assert car.angle_to_checkpoint(track, 4) == pytest.approx(0.0, abs=1e-9)


def test_clean_running_reports_no_collision(car, track):
    assert car.check_collision(track) is False
    assert car.wall_hits == 0


# ── retirement rules ─────────────────────────────────────────

def test_idling_retires_the_car(car, track):
    for _ in range(IDLE_LIMIT_STEPS + 1):
        car.step(track, 0.0, 0.0)
    assert car.alive is False
    assert car.death_reason == "idle"


def test_repeated_wall_hits_retire_the_car(car, track):
    car.wall_hits = MAX_WALL_HITS
    car._check_retirement()
    assert car.alive is False
    assert car.death_reason == "wall_hits"


def test_failing_to_reach_a_checkpoint_retires_the_car(car, track):
    car.steps = STALL_LIMIT_STEPS + 1
    car._check_retirement()
    assert car.alive is False
    assert car.death_reason == "no_progress"


def test_a_retired_car_ignores_further_input(car, track):
    car.alive = False
    before = (car.x, car.y, car.speed)
    assert car.step(track, 1.0, 1.0) == 0.0
    assert (car.x, car.y, car.speed) == before


# ── step() ───────────────────────────────────────────────────

def test_step_accumulates_reward_and_advances_time(car, track):
    total = sum(car.step(track, 0.0, 1.0) for _ in range(20))
    assert car.total_reward == pytest.approx(total)
    assert car.steps == 20
    assert car.total_time > 0


def test_step_penalises_crashes(car, track):
    car.next_cp = 4
    wall_start, wall_end, _ = track.walls[0]
    car.x = (wall_start[0] + wall_end[0]) / 2
    car.y = (wall_start[1] + wall_end[1]) / 2
    reward = car.step(track, 0.0, 0.0)
    assert reward <= CRASH_PENALTY + 1.0
    assert car.wall_hits == 1


def test_telemetry_reports_the_scoring_fields(car, track):
    car.step(track, 0.0, 1.0)
    tel = car.telemetry()
    for key in ("laps", "checkpoints", "wall_hits", "total_reward",
                "best_lap_time", "avg_cp_speed", "avg_split_steps"):
        assert key in tel
    assert tel["laps"] == 0
    assert tel["best_lap_time"] == float("inf")


def test_reset_clears_progress(car, track):
    for _ in range(30):
        car.step(track, 0.2, 1.0)
    car.reset()
    assert car.steps == 0
    assert car.total_cps == 0
    assert car.total_reward == 0.0
    assert car.wall_hits == 0
    assert car.cp_speeds == [] and car.cp_splits == []
    assert car.alive is True
    assert (car.x, car.y) == car.start_pos
    assert car.lap < TOTAL_LAPS
