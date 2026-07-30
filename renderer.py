"""Raylib visualiser.

Draws the same `Simulation` objects the headless trainer uses, so what you see
is exactly what is being scored. Importing this module requires `raylib`;
`main.py` only imports it when a window is actually wanted.
"""

import math

import pyray as rl
from pyray import (
    Vector2, Color,
    begin_drawing, end_drawing, clear_background,
    draw_line_ex, draw_circle, draw_rectangle, draw_text, draw_triangle,
    init_window, close_window, window_should_close,
    set_target_fps, is_key_pressed,
    KEY_ESCAPE, KEY_V, KEY_R, KEY_F,
)

from config import WIDTH, HEIGHT, FPS, NUM_RAYS, FAST_STEPS


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _c(r, g, b, a=255):
    return Color(r, g, b, a)


BG_DARK   = _c(10, 18, 10)
ROAD_FILL = _c(28, 32, 36)
WHITE_C   = _c(255, 255, 255)
GRAY_C    = _c(80,  80,  80)
DIM_GRAY  = _c(40,  40,  40)
_RED_C    = _c(255,  60,  60)
_YEL_C    = _c(255, 220,   0)


def _wall_color(tup):
    r, g, b = tup[:3]
    return _c(int(r), int(g), int(b))


def _hsl_to_rgb(h, s, l):
    if s == 0:
        return l, l, l
    def _h2r(p, q, t):
        t = t % 1.0
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return _h2r(p, q, h+1/3), _h2r(p, q, h), _h2r(p, q, h-1/3)


def _pop_color(idx, n, alive):
    if not alive:
        return _c(60, 60, 60, 90)
    hue = (idx / max(n, 1)) * 300.0
    r, g, b = _hsl_to_rgb(hue / 360.0, 1.0, 0.60)
    return _c(int(r * 255), int(g * 255), int(b * 255))


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _draw_quad(v0, v1, v2, v3, color):
    # Both windings — draw each triangle twice (CW + CCW) to handle any orientation
    draw_triangle(v0, v1, v2, color)
    draw_triangle(v0, v2, v3, color)
    draw_triangle(v2, v1, v0, color)
    draw_triangle(v3, v2, v0, color)


def _dashed_line(x1, y1, x2, y2, dash=14, gap=10, color=None, thick=1.0):
    if color is None:
        color = _c(255, 255, 255, 18)
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash if drawing else gap
        end = min(pos + seg, length)
        if drawing:
            sx = x1 + ux * pos;  sy = y1 + uy * pos
            ex = x1 + ux * end;  ey = y1 + uy * end
            if thick > 1:
                draw_line_ex(Vector2(sx, sy), Vector2(ex, ey), thick, color)
            else:
                rl.draw_line(int(sx), int(sy), int(ex), int(ey), color)
        pos = end
        drawing = not drawing


# ══════════════════════════════════════════════════════════════════════════════
# TRACK RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _draw_road_surface(track):
    """Fill road between consecutive outer/inner pairs as quads."""
    outer, inner = track.outer, track.inner
    n = len(outer)
    for i in range(n):
        j = (i + 1) % n
        _draw_quad(
            Vector2(*outer[i]), Vector2(*outer[j]),
            Vector2(*inner[j]), Vector2(*inner[i]),
            ROAD_FILL
        )


def draw_track(track, best_car_next_cp=-1):
    # 1. filled asphalt
    _draw_road_surface(track)

    # 2. wall segments — glow + crisp edge line + endpoint dots
    for (x1, y1), (x2, y2), col_tup in track.walls:
        col = _wall_color(col_tup)
        glow = _c(max(0, col.r // 5), max(0, col.g // 5),
                  max(0, col.b // 5), 130)
        draw_line_ex(Vector2(x1, y1), Vector2(x2, y2), 10.0, glow)
        draw_line_ex(Vector2(x1, y1), Vector2(x2, y2),  2.5, col)
        draw_rectangle(int(x1) - 3, int(y1) - 3, 6, 6, col)
        draw_rectangle(int(x2) - 3, int(y2) - 3, 6, 6, col)

    # 3. checkpoints
    for i, ((ix, iy), (ox, oy)) in enumerate(track.checkpoints):
        if i == 0:
            draw_line_ex(Vector2(ix, iy), Vector2(ox, oy), 3.0, _RED_C)
            draw_text("S/F",
                      int((ix + ox) // 2) + 4,
                      int((iy + oy) // 2) - 14,
                      11, _c(255, 255, 255, 160))
        else:
            is_next = (i == best_car_next_cp)
            col   = _c(255, 220, 0, 200 if is_next else 45)
            thick = 2.5 if is_next else 1.0
            _dashed_line(ix, iy, ox, oy, dash=7, gap=5, color=col, thick=thick)
            draw_text(str(i),
                      int((ix + ox) // 2) + 4,
                      int((iy + oy) // 2) + 2,
                      11, _c(255, 220, 0, 240 if is_next else 70))


# ══════════════════════════════════════════════════════════════════════════════
# CAR + RAY RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _draw_rays(car, alpha_scale=1.0):
    ox, oy = car.x, car.y
    for dist, (hx, hy) in zip(car.ray_dists, car.ray_hits):
        wall  = 1.0 - dist
        ray_a = int((0.10 + wall * 0.55) * 255 * alpha_scale)
        draw_line_ex(
            Vector2(ox, oy), Vector2(hx, hy),
            1.0, _c(255, 150, 0, max(12, min(255, ray_a)))
        )
        if dist < 0.99:
            dot_a = int((0.45 + wall * 0.55) * 255 * alpha_scale)
            draw_circle(int(hx), int(hy), 3,
                        _c(255, 80, 0, max(60, min(255, dot_a))))


def _draw_car_body(car, color, is_best=False):
    raw = car.get_corners()
    tl, tr = Vector2(*raw[0]), Vector2(*raw[1])
    br, bl = Vector2(*raw[2]), Vector2(*raw[3])

    if is_best:
        cx_f = sum(p[0] for p in raw) / 4
        cy_f = sum(p[1] for p in raw) / 4
        def _exp(v, s=1.22):
            return Vector2(cx_f + (v.x - cx_f) * s,
                           cy_f + (v.y - cy_f) * s)
        gc = _c(max(0, color.r // 2), max(0, color.g // 2),
                max(0, color.b // 2), 100)
        _draw_quad(_exp(tl), _exp(tr), _exp(br), _exp(bl), gc)

    _draw_quad(tl, tr, br, bl, color)

    edge_c = _c(255, 220, 80, 200) if is_best else _c(255, 255, 255, 70)
    draw_line_ex(tl, tr, 1.0, edge_c)
    draw_line_ex(tr, br, 1.0, edge_c)
    draw_line_ex(br, bl, 1.0, edge_c)
    draw_line_ex(bl, tl, 1.0, edge_c)

    if is_best:
        rad = math.radians(car.angle)
        draw_circle(int(car.x + math.cos(rad) * 9),
                    int(car.y + math.sin(rad) * 9), 3, _c(255, 220, 0))
        draw_circle(int(car.x - math.cos(rad) * 8),
                    int(car.y - math.sin(rad) * 8), 3, _c(255, 100, 0, 160))


def draw_cars(cars, best_idx):
    n = len(cars)

    # 1. dead / crashed cars — dim
    for i, car in enumerate(cars):
        if not car.alive:
            _draw_car_body(car, _pop_color(i, n, False), False)

    # 2. alive non-best — faint rays + body
    for i, car in enumerate(cars):
        if not car.alive or i == best_idx:
            continue
        _draw_rays(car, alpha_scale=0.30)
        _draw_car_body(car, _pop_color(i, n, True), False)

    # 3. best car on top
    if 0 <= best_idx < n and cars[best_idx].alive:
        _draw_rays(cars[best_idx], alpha_scale=1.0)
        _draw_car_body(cars[best_idx], _RED_C, True)


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR BAR  (bottom-right, one column per LiDAR ray)
# ══════════════════════════════════════════════════════════════════════════════

def draw_sensor_bar(best_car):
    if best_car is None or not best_car.alive:
        return
    n  = NUM_RAYS
    bx = WIDTH - 10 - n * 22
    by = HEIGHT - 46
    draw_rectangle(bx - 6, by - 16, n * 22 + 12, 40, _c(0, 0, 0, 153))
    for i, norm in enumerate(best_car.ray_dists):
        bh = int(norm * 22)
        x  = bx + i * 22
        draw_rectangle(x, by - 6, 16, 22, _c(17, 17, 17))
        rr, gg, bb = _hsl_to_rgb(norm * 60 / 360.0, 1.0, 0.5)
        if bh > 0:
            draw_rectangle(x, by - 6 + (22 - bh), 16, bh,
                           _c(int(rr * 255), int(gg * 255), int(bb * 255)))
        draw_text(str(i + 1), x + 5, by + 18, 9, _c(68, 68, 68))


# ══════════════════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════════════════

def draw_footer(text):
    draw_text(text, WIDTH // 2 - len(text) * 3, HEIGHT - 18, 11, _c(90, 90, 90))


def draw_status(lines, x=12, y=12):
    width = 8 * max((len(t) for t in lines), default=0) + 20
    draw_rectangle(x - 6, y - 6, width, 16 * len(lines) + 12, _c(0, 0, 0, 150))
    for i, text in enumerate(lines):
        draw_text(text, x, y + i * 16, 13, WHITE_C)


def _draw_phase2_banner():
    bx = WIDTH  // 2 - 240
    by = HEIGHT // 2 - 52
    draw_rectangle(bx, by, 480, 96, _c(0, 0, 0, 215))
    draw_text("PHASE 2 UNLOCKED!", bx + 20, by + 10, 36, _YEL_C)
    draw_text("Route complete - now optimising for speed",
              bx + 20, by + 54, 16, WHITE_C)


# ══════════════════════════════════════════════════════════════════════════════
# RENDERED TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_rendered(trainer, generations, fast=True, show_rays=True):
    """Run the trainer with a live window. Returns the stats collected."""
    sim = trainer.sim
    init_window(WIDTH, HEIGHT, "Race Simulator - training")
    set_target_fps(FPS)

    render = True
    phase2_flash = 0
    last = None
    final_gen = trainer.generation + generations

    try:
        for _ in range(generations):
            trainer.begin_generation()

            while not sim.generation_done:
                if window_should_close() or is_key_pressed(KEY_ESCAPE):
                    trainer.save()
                    return trainer.history
                if is_key_pressed(KEY_V):
                    render = not render
                if is_key_pressed(KEY_F):
                    fast = not fast
                if is_key_pressed(KEY_R):
                    break

                for _ in range(FAST_STEPS if fast else 1):
                    if sim.tick():
                        break

                if render:
                    best_idx = sim.best_index()
                    best_car = sim.cars[best_idx] if sim.cars[best_idx].alive else None
                    begin_drawing()
                    clear_background(BG_DARK)
                    draw_track(sim.track, best_car.next_cp if best_car else -1)
                    draw_cars(sim.cars, best_idx if show_rays else -1)
                    draw_sensor_bar(best_car)
                    status = [
                        f"Gen {trainer.generation + 1}/{final_gen}"
                        f"   Phase {trainer.phase}",
                        f"Step {sim.step_count}/{sim.max_steps}"
                        f"   Alive {sum(1 for c in sim.cars if c.alive)}/{len(sim.cars)}",
                        f"Best CP {max((c.total_cps for c in sim.cars), default=0)}"
                        f"   Laps {max((c.lap for c in sim.cars), default=0)}",
                    ]
                    if last is not None:
                        status.append(f"Last fitness {last.best_fitness:,.0f}")
                    draw_status(status)
                    if phase2_flash > 0:
                        _draw_phase2_banner()
                        phase2_flash -= 1
                    draw_footer(f"{'FAST' if fast else '1x'} | "
                                "F=speed  V=render  R=skip gen  ESC=save and quit")
                    end_drawing()

            last = trainer.finish_generation()
            if trainer.phase_changed:
                phase2_flash = FPS * 3

        trainer.save()
        return trainer.history
    finally:
        close_window()


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def replay(simulation, title="Race Simulator - replay"):
    """Watch a trained policy drive. `simulation` should hold a single car."""
    init_window(WIDTH, HEIGHT, title)
    set_target_fps(FPS)
    try:
        while not window_should_close():
            if is_key_pressed(KEY_ESCAPE):
                break
            if is_key_pressed(KEY_R) or simulation.generation_done:
                simulation.reset()

            simulation.tick()

            best_idx = simulation.best_index()
            car = simulation.cars[best_idx]
            begin_drawing()
            clear_background(BG_DARK)
            draw_track(simulation.track, car.next_cp)
            draw_cars(simulation.cars, best_idx)
            draw_sensor_bar(car if car.alive else None)
            draw_status([
                f"Lap {car.lap}/{simulation.total_laps}   CP {car.next_cp}",
                f"Speed {car.speed:5.1f}   Best lap "
                + (f"{car.best_time:.2f}s" if car.best_time < float('inf') else "--"),
                f"Wall hits {car.wall_hits}",
            ])
            draw_footer("R=restart  ESC=exit")
            end_drawing()
    finally:
        close_window()
