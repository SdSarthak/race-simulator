#!/usr/bin/env python3
"""
main_rl.py — AI Race Simulator with Raylib.
Large polygon track, filled road surface, 11-ray LiDAR on every car.
Collisions teleport to last checkpoint. No HUD scoreboard.

  python main_rl.py           # train
  python main_rl.py replay    # watch saved model
"""

# ── pygame shim — must come BEFORE importing track.py / car.py ───────────────
# track.py and car.py both `import pygame` at module level for draw() methods.
# Stub it out so physics runs with no Pygame display.
import sys, types as _types
_pg = _types.ModuleType("pygame")
_pg.draw = _types.SimpleNamespace(
    line=lambda *a, **k: None,
    circle=lambda *a, **k: None,
    polygon=lambda *a, **k: None,
    rect=lambda *a, **k: None,
)
sys.modules.setdefault("pygame", _pg)
# ─────────────────────────────────────────────────────────────────────────────

import math, os

import pyray as rl
from pyray import (
    Vector2, Color,
    begin_drawing, end_drawing, clear_background,
    draw_line_ex, draw_circle, draw_rectangle, draw_text, draw_triangle,
    init_window, close_window, window_should_close,
    set_target_fps, is_key_pressed,
    KEY_ESCAPE, KEY_V, KEY_R, KEY_F,
)

FAST_STEPS = 500   # physics ticks per rendered frame in fast mode

from config import (
    WIDTH, HEIGHT, FPS,
    NUM_RAYS,
    POP_SIZE, NUM_GENERATIONS, MAX_STEPS_GEN, TOTAL_LAPS,
    STATE_DIM, CAR_COLORS,
    BEST_MODEL, MODEL_DIR,
)
from track import Track
from car import Car
from ai import GeneticAgent, PPOAgent


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
_GRN_C    = _c(  0, 221, 100)
_ORA_C    = _c(255, 102,   0)

_WALL_COLOR_MAP = {
    (255, 255,  60): _c(255, 255,  60),
    ( 50, 255, 100): _c( 50, 255, 100),
    ( 80, 130, 255): _c( 80, 130, 255),
    (255,  60,  60): _c(255,  60,  60),
}

def _wall_color(tup):
    return _WALL_COLOR_MAP.get(tup, WHITE_C)

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
    outer = track.outer
    inner = track.inner
    n = len(outer)
    for i in range(n):
        j = (i + 1) % n
        _draw_quad(
            Vector2(*outer[i]), Vector2(*outer[j]),
            Vector2(*inner[j]), Vector2(*inner[i]),
            ROAD_FILL
        )


def draw_track_rl(track, best_car_next_cp=-1):
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

def _draw_rays_rl(car, alpha_scale=1.0):
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


def _draw_car_body_rl(car, color, is_best=False):
    raw = car.get_corners()
    tl = Vector2(*raw[0])
    tr = Vector2(*raw[1])
    br = Vector2(*raw[2])
    bl = Vector2(*raw[3])

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
        fx = car.x + math.cos(rad) * 9
        fy = car.y + math.sin(rad) * 9
        draw_circle(int(fx), int(fy), 3, _c(255, 220, 0))
        bx = car.x - math.cos(rad) * 8
        by_v = car.y - math.sin(rad) * 8
        draw_circle(int(bx), int(by_v), 3, _c(255, 100, 0, 160))


def draw_cars_rl(cars, best_idx):
    n = len(cars)

    # 1. dead / crashed cars — dim
    for i, car in enumerate(cars):
        if car.alive:
            continue
        _draw_car_body_rl(car, _pop_color(i, n, False), False)

    # 2. alive non-best — faint rays + body
    for i, car in enumerate(cars):
        if not car.alive or i == best_idx:
            continue
        _draw_rays_rl(car, alpha_scale=0.30)
        _draw_car_body_rl(car, _pop_color(i, n, True), False)

    # 3. best car on top
    if 0 <= best_idx < n and cars[best_idx].alive:
        _draw_rays_rl(cars[best_idx], alpha_scale=1.0)
        _draw_car_body_rl(cars[best_idx], _RED_C, True)


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR BAR  (bottom-right, one column per LiDAR ray)
# ══════════════════════════════════════════════════════════════════════════════

def draw_sensor_bar_rl(best_car):
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
        hue = norm * 60 / 360.0
        rr, gg, bb = _hsl_to_rgb(hue, 1.0, 0.5)
        if bh > 0:
            draw_rectangle(x, by - 6 + (22 - bh), 16, bh,
                           _c(int(rr * 255), int(gg * 255), int(bb * 255)))
        draw_text(str(i + 1), x + 5, by + 18, 9, _c(68, 68, 68))


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER + PHASE-2 BANNER
# ══════════════════════════════════════════════════════════════════════════════

def draw_footer_rl(text):
    w = len(text) * 6
    draw_text(text, WIDTH // 2 - w // 2, HEIGHT - 18, 11, _c(60, 60, 60))


def _draw_phase2_banner():
    bx = WIDTH  // 2 - 240
    by = HEIGHT // 2 - 52
    draw_rectangle(bx, by, 480, 96, _c(0, 0, 0, 215))
    draw_text("PHASE 2 UNLOCKED!", bx + 20, by + 10, 36, _YEL_C)
    draw_text("Route complete — now optimising for speed",
              bx + 20, by + 54, 16, WHITE_C)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_genetic_rl():
    init_window(WIDTH, HEIGHT, "Speed Racer — AI Training (Raylib)")
    set_target_fps(FPS)

    track = Track()
    agent = GeneticAgent(STATE_DIM, 2)

    if os.path.exists(BEST_MODEL):
        try:
            agent.load(BEST_MODEL)
            print(f"Resumed from {BEST_MODEL}")
        except Exception:
            print("Fresh start (model mismatch).")

    phase            = 1
    phase2_flash     = 0
    global_best_time = float('inf')
    best_fit_ever    = float('-inf')
    render           = True
    fast             = True     # start in 500× fast mode; F toggles

    for gen in range(NUM_GENERATIONS):

        # Stagger spawn: 4 columns × 10 rows on the start straight.
        cp0_a, cp0_b = track.checkpoints[0]
        cp1_a, cp1_b = track.checkpoints[1]
        m0x = (cp0_a[0]+cp0_b[0])/2;  m0y = (cp0_a[1]+cp0_b[1])/2
        m1x = (cp1_a[0]+cp1_b[0])/2;  m1y = (cp1_a[1]+cp1_b[1])/2
        adx = m1x-m0x; ady = m1y-m0y
        alen = math.hypot(adx, ady)
        adx /= alen; ady /= alen
        apx = -ady;   apy = adx
        road_w = math.hypot(cp0_b[0]-cp0_a[0], cp0_b[1]-cp0_a[1])
        cols, col_gap, row_gap = 4, road_w * 0.6 / 3, 30
        spawn_angle = math.degrees(math.atan2(ady, adx))
        cars = []
        for i in range(POP_SIZE):
            col = i % cols
            row = i // cols
            lat = (col - (cols-1)/2.0) * col_gap
            lon = (row + 1) * row_gap
            sx = m0x + adx*lon + apx*lat
            sy = m0y + ady*lon + apy*lat
            car = Car((sx, sy), spawn_angle, i % len(CAR_COLORS))
            cars.append(car)
        for car in cars:
            car.cast_rays(track)

        step = 0
        while step < MAX_STEPS_GEN:
            if window_should_close():
                agent.save(BEST_MODEL)
                close_window()
                return

            if is_key_pressed(KEY_ESCAPE):
                agent.save(BEST_MODEL)
                close_window()
                return
            if is_key_pressed(KEY_V):
                render = not render
            if is_key_pressed(KEY_F):
                fast = not fast
            if is_key_pressed(KEY_R):
                break

            # run FAST_STEPS ticks per frame in fast mode, else 1
            ticks = FAST_STEPS if fast else 1
            gen_done = False
            for _ in range(ticks):
                all_done = True
                for i, car in enumerate(cars):
                    if not car.alive or car.lap >= TOTAL_LAPS:
                        continue
                    all_done = False
                    state  = car.get_state(track)
                    action = agent.get_action(i, state)
                    car.update(action[0], action[1])
                    car.cast_rays(track)
                    car.check_collision(track)
                    car.check_checkpoints(track)
                    if car.best_time < global_best_time:
                        global_best_time = car.best_time
                step += 1
                if all_done or step >= MAX_STEPS_GEN:
                    gen_done = True
                    break

            if render:
                alive_list = [c for c in cars if c.alive]
                best_idx   = max(range(len(cars)),
                                 key=lambda i: (cars[i].alive,
                                                cars[i].total_cps,
                                                cars[i].speed))
                best_car   = cars[best_idx] if cars[best_idx].alive else None
                next_cp    = best_car.next_cp if best_car else -1

                begin_drawing()
                clear_background(BG_DARK)
                draw_track_rl(track, next_cp)
                draw_cars_rl(cars, best_idx)
                draw_sensor_bar_rl(best_car)

                if phase2_flash > 0:
                    _draw_phase2_banner()
                    phase2_flash -= 1

                mode_lbl = f"{'500x FAST' if fast else '1x'}"
                draw_footer_rl(
                    f"Gen {gen+1}/{NUM_GENERATIONS} | Phase {phase} | "
                    f"{mode_lbl} | F=speed | V=render | ESC=quit | R=skip"
                )
                end_drawing()

            if gen_done:
                break

        agent.phase = phase
        best_fit, _ = agent.evolve(cars)

        if best_fit > best_fit_ever:
            best_fit_ever = best_fit
            agent.save(BEST_MODEL)

        if (gen + 1) % 50 == 0:
            os.makedirs(MODEL_DIR, exist_ok=True)
            agent.save(f"{MODEL_DIR}/gen_{gen+1}.pt")

        if phase == 1 and any(c.lap >= TOTAL_LAPS for c in cars):
            phase        = 2
            phase2_flash = FPS * 4
            agent.phase  = 2
            print(f"\n{'='*55}\n  PHASE 2 UNLOCKED gen {gen+1}\n{'='*55}\n")

        finished  = sum(1 for c in cars if c.lap >= TOTAL_LAPS)
        best_cps  = max(c.total_cps for c in cars)
        bt_str    = f"{global_best_time:.2f}s" if global_best_time < float('inf') else "--"
        alive_end = sum(1 for c in cars if c.alive)
        print(f"Gen {gen+1:>4}/{NUM_GENERATIONS} | Ph{phase} | "
              f"Done {finished:>2}/{POP_SIZE} | BestCP {best_cps:>3} | "
              f"BestTime {bt_str:>8} | Fit {best_fit:>8.0f} | Alive {alive_end:>2}")

    agent.save(BEST_MODEL)
    print("Training complete!")
    close_window()


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def run_replay_rl():
    if not os.path.exists(BEST_MODEL):
        print(f"No model at {BEST_MODEL}")
        return

    init_window(WIDTH, HEIGHT, "Speed Racer — Replay (Raylib)")
    set_target_fps(FPS)

    track = Track()
    agent = PPOAgent(STATE_DIM, 2)
    agent.load(BEST_MODEL)

    car = Car(track.start_pos, track.start_angle)
    car.cast_rays(track)
    gbest = float('inf')

    while not window_should_close():
        if is_key_pressed(KEY_ESCAPE):
            break
        if is_key_pressed(KEY_R):
            car.reset()
            car.cast_rays(track)

        if car.alive and car.lap < TOTAL_LAPS:
            action, *_ = agent.net.act(car.get_state(track), deterministic=True)
            car.update(action[0], action[1])
            car.cast_rays(track)
            car.check_collision(track)
            car.check_checkpoints(track)
            if car.best_time < gbest:
                gbest = car.best_time
        else:
            car.reset()
            car.cast_rays(track)

        begin_drawing()
        clear_background(BG_DARK)
        draw_track_rl(track, car.next_cp)
        draw_cars_rl([car], 0)
        draw_sensor_bar_rl(car if car.alive else None)
        draw_footer_rl("R: Reset | ESC: Exit")
        end_drawing()

    close_window()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "train"
    if mode == "replay":
        run_replay_rl()
    else:
        run_genetic_rl()
