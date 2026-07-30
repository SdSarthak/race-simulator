import math
import random

import numpy as np
import pytest

from config import MAX_RAY_LENGTH
from track import Track, line_intersect, LAYOUTS


# ── line_intersect (the reference implementation) ────────────

def test_crossing_segments_intersect_at_midpoint():
    hit = line_intersect((0, 0), (10, 0), (5, -5), (5, 5))
    assert hit is not None
    x, y, t = hit
    assert (x, y) == pytest.approx((5.0, 0.0))
    assert t == pytest.approx(0.5)


def test_parallel_segments_do_not_intersect():
    assert line_intersect((0, 0), (10, 0), (0, 1), (10, 1)) is None


def test_segments_that_only_cross_when_extended_do_not_intersect():
    assert line_intersect((0, 0), (1, 0), (5, -5), (5, 5)) is None


# ── track construction ───────────────────────────────────────

@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_layouts_build_consistently(layout):
    t = Track(layout)
    assert len(t.outer) == len(t.inner)
    assert len(t.checkpoints) == len(t.outer)
    # every boundary edge, inner and outer, becomes a wall
    assert len(t.walls) == 2 * len(t.outer)
    assert all(len(w) == 3 for w in t.walls)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError):
        Track("figure-of-eight")


def test_start_position_is_inside_the_road(track):
    """A car parked at the start should see walls on both sides, not on its nose."""
    dists, _ = track.cast_rays(track.start_pos, [0, 90, 180, 270])
    assert all(0 < d <= MAX_RAY_LENGTH for d in dists)
    assert dists[1] < MAX_RAY_LENGTH      # wall to the right
    assert dists[3] < MAX_RAY_LENGTH      # wall to the left


# ── ray casting ──────────────────────────────────────────────

def _reference_cast(track, origin, angle_deg):
    """Slow, obvious ray cast used to pin the vectorised implementation."""
    rad = math.radians(angle_deg)
    end = (origin[0] + math.cos(rad) * MAX_RAY_LENGTH,
           origin[1] + math.sin(rad) * MAX_RAY_LENGTH)
    closest = float(MAX_RAY_LENGTH)
    for ws, we, _ in track.walls:
        hit = line_intersect(origin, end, ws, we)
        if hit is not None:
            closest = min(closest, hit[2] * MAX_RAY_LENGTH)
    return closest


def test_vectorised_ray_cast_matches_reference(track):
    rng = random.Random(1234)
    for _ in range(60):
        origin = (rng.uniform(100, 1300), rng.uniform(100, 800))
        angle = rng.uniform(0, 360)
        dist, _ = track.cast_ray(origin, angle)
        assert dist == pytest.approx(_reference_cast(track, origin, angle), abs=1e-6)


def test_ray_fan_matches_individual_casts(track):
    angles = [0, 37, 90, 181, 300]
    dists, hits = track.cast_rays(track.start_pos, angles)
    assert dists.shape == (len(angles),)
    assert hits.shape == (len(angles), 2)
    for i, a in enumerate(angles):
        single_d, single_hit = track.cast_ray(track.start_pos, a)
        assert dists[i] == pytest.approx(single_d, abs=1e-9)
        assert hits[i] == pytest.approx(np.array(single_hit), abs=1e-9)


def test_hit_point_lies_on_the_ray(track):
    origin = track.start_pos
    dist, hit = track.cast_ray(origin, 90.0)
    assert hit[0] == pytest.approx(origin[0], abs=1e-6)
    assert hit[1] == pytest.approx(origin[1] + dist, abs=1e-6)


def test_ray_from_open_space_returns_max_range():
    """Far outside the circuit, a ray pointing away hits nothing."""
    t = Track()
    dist, _ = t.cast_ray((-5000, -5000), 225.0)
    assert dist == pytest.approx(MAX_RAY_LENGTH)


# ── collision ────────────────────────────────────────────────

def test_polygon_crossing_a_wall_collides(track):
    wall_start, wall_end, _ = track.walls[0]
    mid = ((wall_start[0] + wall_end[0]) / 2, (wall_start[1] + wall_end[1]) / 2)
    box = [(mid[0] - 5, mid[1] - 5), (mid[0] + 5, mid[1] - 5),
           (mid[0] + 5, mid[1] + 5), (mid[0] - 5, mid[1] + 5)]
    assert track.check_collision(box) is True


def test_polygon_in_open_road_does_not_collide(track):
    cx, cy = track.checkpoint_midpoint(1)
    box = [(cx - 4, cy - 4), (cx + 4, cy - 4), (cx + 4, cy + 4), (cx - 4, cy + 4)]
    assert track.check_collision(box) is False


# ── checkpoints ──────────────────────────────────────────────

def test_movement_across_a_checkpoint_is_detected(track):
    a, b = track.checkpoints[1]
    mid = track.checkpoint_midpoint(1)
    # step across the checkpoint line perpendicular to it
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    px, py = -dy / length, dx / length
    before = (mid[0] - px * 5, mid[1] - py * 5)
    after = (mid[0] + px * 5, mid[1] + py * 5)
    assert track.check_checkpoint(before, after, 1) is True
    assert track.check_checkpoint(before, before, 1) is False


def test_checkpoint_index_wraps(track):
    n = len(track.checkpoints)
    assert (track.checkpoint_midpoint(n) == track.checkpoint_midpoint(0))
    assert track.checkpoint_width(n + 1) == pytest.approx(track.checkpoint_width(1))
