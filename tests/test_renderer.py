"""Renderer tests that need no window.

Anything that draws needs a live GL context, so only the pure helpers are
exercised here; the module is skipped entirely when raylib is not installed
(headless boxes and CI).
"""

import pytest

pytest.importorskip("pyray", reason="raylib is optional")

import renderer as R  # noqa: E402


def test_hsl_to_rgb_matches_known_colours():
    assert R._hsl_to_rgb(0.0, 1.0, 0.5) == pytest.approx((1.0, 0.0, 0.0))
    assert R._hsl_to_rgb(1 / 3, 1.0, 0.5) == pytest.approx((0.0, 1.0, 0.0))
    assert R._hsl_to_rgb(2 / 3, 1.0, 0.5) == pytest.approx((0.0, 0.0, 1.0))
    assert R._hsl_to_rgb(0.0, 0.0, 0.25) == pytest.approx((0.25, 0.25, 0.25))


def test_population_colours_are_distinct_and_in_range():
    colours = [R._pop_color(i, 6, True) for i in range(6)]
    assert len({(c.r, c.g, c.b) for c in colours}) == 6
    for c in colours:
        assert all(0 <= v <= 255 for v in (c.r, c.g, c.b, c.a))


def test_retired_cars_are_drawn_dim_and_translucent():
    dead = R._pop_color(0, 6, False)
    assert dead.a < 255
    assert dead.r == dead.g == dead.b


def test_wall_colours_pass_through_the_track_palette():
    from config import YELLOW
    col = R._wall_color(YELLOW)
    assert (col.r, col.g, col.b) == YELLOW


def test_the_drawing_entry_points_exist():
    for name in ("draw_track", "draw_cars", "draw_sensor_bar",
                 "draw_status", "draw_footer", "train_rendered", "replay"):
        assert callable(getattr(R, name))
