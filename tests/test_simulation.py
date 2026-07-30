import glob
import os

import numpy as np
import pytest
import torch

from config import STATE_DIM, ACTION_DIM
from ai import GeneticAgent, PolicyAgent
from simulation import Simulation, Trainer, GenerationStats, build, set_seed
from track import Track


def _sim(pop=4, laps=2, steps=40, seed=0):
    track, agent, sim = build(pop_size=pop, total_laps=laps,
                              max_steps=steps, seed=seed)
    return track, agent, sim


# ── seeding ──────────────────────────────────────────────────

def test_set_seed_is_a_no_op_without_a_seed():
    assert set_seed(None) is None
    assert set_seed(5) == 5


# ── spawning ─────────────────────────────────────────────────

def test_the_field_spawns_on_the_road_and_ready_to_drive():
    track, _, sim = _sim(pop=8)
    assert len(sim.cars) == 8
    assert all(car.alive for car in sim.cars)
    for car in sim.cars:
        assert not track.check_collision(car.get_corners())
        assert len(car.ray_dists) == len(car.ray_hits)
        assert min(car.ray_dists) > 0.0


def test_cars_are_staggered_rather_than_stacked():
    _, _, sim = _sim(pop=8)
    positions = {(round(c.x, 3), round(c.y, 3)) for c in sim.cars}
    assert len(positions) == 8


def test_cars_start_pointing_down_the_track():
    track, _, sim = _sim(pop=4)
    for car in sim.cars:
        assert abs(car.angle_to_checkpoint(track, 1)) < 90.0


# ── stepping ─────────────────────────────────────────────────

def test_a_tick_advances_the_clock():
    _, _, sim = _sim(steps=10)
    assert sim.tick() is False
    assert sim.step_count == 1
    assert all(car.steps == 1 for car in sim.cars if car.alive)


def test_a_generation_stops_at_the_step_limit():
    _, _, sim = _sim(steps=25)
    steps = sim.run()
    assert sim.generation_done is True
    assert steps <= 25
    assert sim.tick() is True          # further ticks are inert


def test_a_generation_ends_early_once_every_car_is_done():
    _, _, sim = _sim(pop=4, laps=0, steps=100)
    sim.run()
    assert sim.step_count == 1         # nothing left to simulate


def test_the_same_seed_reproduces_the_same_race():
    results = []
    for _ in range(2):
        _, _, sim = _sim(pop=6, steps=60, seed=99)
        sim.run()
        results.append([(c.total_cps, round(c.x, 6), round(c.y, 6))
                        for c in sim.cars])
    assert results[0] == results[1]


def test_different_seeds_produce_different_races():
    runs = []
    for seed in (1, 2):
        _, _, sim = _sim(pop=6, steps=60, seed=seed)
        sim.run()
        runs.append([round(c.x, 6) for c in sim.cars])
    assert runs[0] != runs[1]


def test_reset_returns_a_clean_field():
    _, _, sim = _sim(pop=4, steps=30)
    sim.run()
    sim.reset()
    assert sim.step_count == 0
    assert sim.generation_done is False
    assert all(c.steps == 0 and c.alive for c in sim.cars)


def test_an_agent_without_batching_still_drives_the_field():
    torch.manual_seed(0)
    sim = Simulation(Track(), PolicyAgent(STATE_DIM, ACTION_DIM),
                     pop_size=3, total_laps=1, max_steps=15)
    sim.run()
    assert sim.step_count == 15
    assert all(c.steps > 0 for c in sim.cars)


# ── leader tracking ──────────────────────────────────────────

def test_the_leader_is_the_car_that_has_gone_furthest():
    _, _, sim = _sim(pop=5, steps=5)
    sim.cars[2].total_cps = 9
    assert sim.best_index() == 2
    assert sim.best_car() is sim.cars[2]


def test_best_lap_time_is_infinite_until_a_lap_is_set():
    _, _, sim = _sim(pop=3, steps=5)
    assert sim.best_lap_time() == float("inf")
    sim.cars[1].best_time = 7.5
    assert sim.best_lap_time() == 7.5


# ── trainer ──────────────────────────────────────────────────

def test_a_generation_produces_stats_and_a_saved_model(tmp_path):
    _, agent, sim = _sim(pop=4, steps=30, seed=3)
    model = str(tmp_path / "best.pt")
    trainer = Trainer(agent, sim, model_path=model,
                      log_dir=str(tmp_path / "logs"), quiet=True,
                      checkpoint_every=0)
    stats = trainer.run_generation()

    assert isinstance(stats, GenerationStats)
    assert stats.generation == 1
    assert stats.steps <= 30
    assert os.path.exists(model)
    assert "Gen" in stats.summary()

    logs = glob.glob(str(tmp_path / "logs" / "*.csv"))
    assert len(logs) == 1
    lines = open(logs[0], encoding="utf-8").read().strip().splitlines()
    assert lines[0].startswith("generation,phase,steps")
    assert len(lines) == 2


def test_consecutive_generations_append_to_one_log(tmp_path):
    _, agent, sim = _sim(pop=4, steps=20, seed=4)
    trainer = Trainer(agent, sim, model_path=str(tmp_path / "m.pt"),
                      log_dir=str(tmp_path / "logs"), quiet=True,
                      checkpoint_every=0)
    trainer.train(generations=3)
    assert trainer.generation == 3
    assert len(trainer.history) == 3
    logs = glob.glob(str(tmp_path / "logs" / "*.csv"))
    assert len(open(logs[0], encoding="utf-8").read().strip().splitlines()) == 4


def test_logging_can_be_switched_off(tmp_path):
    _, agent, sim = _sim(pop=4, steps=20)
    trainer = Trainer(agent, sim, model_path=str(tmp_path / "m.pt"),
                      log_dir=None, quiet=True, checkpoint_every=0)
    trainer.run_generation()
    assert not glob.glob(str(tmp_path / "*.csv"))


def test_finishing_the_race_unlocks_phase_two(tmp_path):
    # total_laps=0 means every car counts as finished immediately
    _, agent, sim = _sim(pop=4, laps=0, steps=10)
    trainer = Trainer(agent, sim, model_path=str(tmp_path / "m.pt"),
                      log_dir=None, quiet=True, checkpoint_every=0)
    stats = trainer.run_generation()
    assert stats.phase == 1            # the generation itself ran in phase 1
    assert trainer.phase == 2
    assert trainer.phase_changed is True
    assert agent.phase == 2

    trainer.begin_generation()
    assert trainer.phase_changed is False


def test_numbered_snapshots_are_written_on_schedule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, agent, sim = _sim(pop=4, steps=15)
    trainer = Trainer(agent, sim, model_path=str(tmp_path / "best.pt"),
                      log_dir=None, quiet=True, checkpoint_every=2)
    trainer.train(generations=2)
    assert os.path.exists(os.path.join("models", "gen_2.pt"))


def test_training_state_survives_a_restart(tmp_path):
    model = str(tmp_path / "resume.pt")
    _, agent, sim = _sim(pop=4, laps=0, steps=10)
    trainer = Trainer(agent, sim, model_path=model, log_dir=None,
                      quiet=True, checkpoint_every=0)
    trainer.run_generation()           # flips to phase 2 and saves
    trainer.save()

    _, agent2, sim2 = _sim(pop=4, laps=0, steps=10)
    resumed = Trainer(agent2, sim2, model_path=model, log_dir=None,
                      quiet=True, checkpoint_every=0)
    meta = resumed.load()
    assert meta["generation"] == 1
    assert resumed.phase == 2
    assert agent2.phase == 2


def test_a_broken_log_directory_does_not_stop_training(tmp_path):
    blocker = tmp_path / "logs"
    blocker.write_text("not a directory", encoding="utf-8")
    _, agent, sim = _sim(pop=4, steps=15)
    trainer = Trainer(agent, sim, model_path=str(tmp_path / "m.pt"),
                      log_dir=str(blocker), quiet=True, checkpoint_every=0)
    stats = trainer.run_generation()
    assert stats.generation == 1
    assert trainer.log_dir is None     # disabled itself and carried on


# ── stats ────────────────────────────────────────────────────

def test_stats_rows_are_flat_and_csv_friendly(tmp_path):
    _, agent, sim = _sim(pop=4, steps=15)
    trainer = Trainer(agent, sim, model_path=str(tmp_path / "m.pt"),
                      log_dir=None, quiet=True, checkpoint_every=0)
    row = trainer.run_generation().as_row()
    assert set(row) >= {"generation", "phase", "best_fitness", "wall_hits"}
    assert all(not isinstance(v, (list, dict)) for v in row.values())
