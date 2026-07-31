"""Failure modes: non-finite controls, degenerate populations, broken files.

Everything here is a regression test for something that used to produce a wrong
result, a silently corrupted network, or an unhandled traceback.
"""

import math
import os

import numpy as np
import pytest
import torch

from config import STATE_DIM, ACTION_DIM, MAX_SPEED, TOTAL_LAPS
from ai import (
    GeneticAgent, PPOAgent, PolicyAgent, PolicyNet,
    save_checkpoint, load_checkpoint,
)
from car import Car
from simulation import Simulation, Trainer, build
from track import Track


class FakeCar:
    """Just the fields the fitness function reads."""

    def __init__(self, lap=0, total_cps=0, speed=0.0, best_time=float("inf"),
                 cp_speeds=None, cp_splits=None, wall_hits=0, num_cps=17):
        self.lap = lap
        self.total_cps = total_cps
        self.speed = speed
        self.best_time = best_time
        self.cp_speeds = cp_speeds or []
        self.cp_splits = cp_splits or []
        self.wall_hits = wall_hits
        self.num_cps = num_cps


def _sig(net):
    return tuple(float(p.detach().sum()) for p in net.parameters())


# ── non-finite control inputs ────────────────────────────────

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_control_is_treated_as_no_input(track, bad):
    car = Car(track.start_pos, track.start_angle)
    reference = Car(track.start_pos, track.start_angle)
    car.update(bad, bad)
    reference.update(0.0, 0.0)
    assert (car.x, car.y, car.speed, car.angle) == pytest.approx(
        (reference.x, reference.y, reference.speed, reference.angle))


def test_a_nan_action_cannot_corrupt_the_state_vector(track):
    car = Car(track.start_pos, track.start_angle)
    for _ in range(20):
        car.step(track, float("nan"), float("nan"))
    state = car.get_state(track)
    assert np.all(np.isfinite(state))
    assert math.isfinite(car.x) and math.isfinite(car.y)
    assert math.isfinite(car.total_reward)


def test_an_out_of_range_control_still_saturates(track):
    car = Car(track.start_pos, track.start_angle)
    clean = Car(track.start_pos, track.start_angle)
    car.update(50.0, 50.0)
    clean.update(1.0, 1.0)
    assert car.speed == pytest.approx(clean.speed)
    assert car.angle == pytest.approx(clean.angle)


# ── NaN fitness never wins ───────────────────────────────────

class _NaNCar(FakeCar):
    """Scores NaN in either phase."""

    def __init__(self):
        super().__init__(total_cps=float("nan"), lap=0)
        self.best_time = float("nan")


def test_a_nan_scoring_car_is_never_crowned_best():
    torch.manual_seed(20)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=4)
    winner = _sig(agent.nets[2])
    cars = [_NaNCar(), _NaNCar(), FakeCar(total_cps=9), _NaNCar()]
    best, avg = agent.evolve(cars)
    assert math.isfinite(best)
    assert _sig(agent.best_net) == winner


def test_a_field_that_all_scores_nan_still_evolves_without_crashing():
    torch.manual_seed(21)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=4)
    best, avg = agent.evolve([_NaNCar() for _ in range(4)])
    assert best == -float("inf")
    assert len(agent.nets) == 4


# ── small populations still search ───────────────────────────

@pytest.mark.parametrize("pop", [2, 3, 5])
def test_a_small_population_is_not_frozen_by_its_elite_quota(pop):
    torch.manual_seed(22)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=pop)
    start = {_sig(n) for n in agent.nets}
    for _ in range(4):
        agent.evolve([FakeCar(total_cps=i) for i in range(pop)])
    novel = {_sig(n) for n in agent.nets} - start
    assert novel, f"pop_size={pop} never produced a new network"


def test_the_elite_never_swallows_the_whole_population():
    torch.manual_seed(23)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    best_before = _sig(agent.nets[1])
    agent.evolve([FakeCar(total_cps=0), FakeCar(total_cps=5)])
    # the winner survives untouched, the other slot is a fresh child
    assert _sig(agent.best_net) == best_before
    assert _sig(agent.nets[0]) == best_before


# ── phase-2 fitness follows the configured race distance ─────

def test_phase_two_scores_a_short_race_out_of_its_own_lap_target():
    one_lap = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=2, total_laps=1)
    default = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    one_lap.phase = default.phase = 2
    finisher = FakeCar(lap=1, total_cps=17, best_time=5.0, num_cps=17)

    # a car that finished the race it was actually asked to run scores full
    # marks on the lap and progress terms
    assert one_lap._fitness(finisher) > default._fitness(finisher)
    assert one_lap._fitness(finisher) == pytest.approx(
        1000.0 + 500.0 + max(0.0, 1000.0 - 5.0 * 60.0))


def test_the_trainer_tells_the_agent_the_lap_target(tmp_path):
    _, agent, sim = build(pop_size=4, total_laps=1, max_steps=5, seed=0)
    assert agent.total_laps == TOTAL_LAPS      # untouched before wiring
    Trainer(agent, sim, model_path=str(tmp_path / "m.pt"),
            log_dir=None, quiet=True, checkpoint_every=0)
    assert agent.total_laps == 1


def test_a_zero_lap_target_cannot_divide_by_zero():
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=2, total_laps=0)
    agent.phase = 2
    assert agent.total_laps == 1
    assert math.isfinite(agent._fitness(FakeCar(lap=1, total_cps=17)))


# ── PPO numerical safety ─────────────────────────────────────

def test_a_single_transition_update_does_not_destroy_the_network():
    """torch.std() of one sample is NaN; normalising by it used to nuke every weight."""
    torch.manual_seed(24)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=1)
    losses = agent.update(
        np.zeros((1, STATE_DIM), dtype=np.float32),
        np.zeros((1, ACTION_DIM), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )
    assert all(np.isfinite(v) for v in losses.values())
    assert all(torch.isfinite(p).all() for p in agent.net.parameters())
    # and the network still produces usable actions afterwards
    actions = agent.get_actions(np.zeros((1, STATE_DIM), dtype=np.float32))
    assert np.all(np.isfinite(actions))


def test_identical_advantages_normalise_to_zero_rather_than_exploding():
    torch.manual_seed(25)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    n = 8
    losses = agent.update(
        np.zeros((n, STATE_DIM), dtype=np.float32),
        np.zeros((n, ACTION_DIM), dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.full(n, 3.0, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    )
    assert all(np.isfinite(v) for v in losses.values())
    assert all(torch.isfinite(p).all() for p in agent.net.parameters())


def test_a_non_finite_batch_is_skipped_instead_of_poisoning_the_weights():
    torch.manual_seed(26)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    n = 8
    returns = np.full(n, np.inf, dtype=np.float32)
    agent.update(
        np.zeros((n, STATE_DIM), dtype=np.float32),
        np.zeros((n, ACTION_DIM), dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.linspace(-1.0, 1.0, n).astype(np.float32),
        returns,
    )
    assert agent.skipped_updates > 0
    assert all(torch.isfinite(p).all() for p in agent.net.parameters())


def test_a_ppo_generation_with_one_step_is_survivable(tmp_path):
    from simulation import PPOTrainer
    torch.manual_seed(27)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    sim = Simulation(Track(), agent, pop_size=2, total_laps=1, max_steps=1)
    trainer = PPOTrainer(agent, sim, model_path=str(tmp_path / "p.pt"),
                         log_dir=None, quiet=True, checkpoint_every=0)
    trainer.run_generation()
    assert all(torch.isfinite(p).all() for p in agent.net.parameters())


# ── checkpoint loading ───────────────────────────────────────

def test_a_missing_checkpoint_names_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_checkpoint(str(tmp_path / "absent.pt"), STATE_DIM, ACTION_DIM)


def test_a_truncated_file_is_reported_not_traced(tmp_path):
    path = tmp_path / "junk.pt"
    path.write_bytes(b"this is definitely not a torch checkpoint")
    with pytest.raises(ValueError, match="not a readable checkpoint"):
        load_checkpoint(str(path), STATE_DIM, ACTION_DIM)


def test_a_pickle_of_something_else_is_rejected(tmp_path):
    path = str(tmp_path / "wrong-payload.pt")
    torch.save({"state_dict": {"note": "no tensors here"}}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, STATE_DIM, ACTION_DIM)


def test_checkpoints_load_without_executing_arbitrary_pickle(tmp_path):
    """A checkpoint is tensors plus scalars, so a weights-only load must suffice."""
    torch.manual_seed(28)
    path = str(tmp_path / "safe.pt")
    save_checkpoint(PolicyNet(STATE_DIM, ACTION_DIM).state_dict(), path,
                    STATE_DIM, ACTION_DIM, generation=4, phase=2, algo="ppo")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["generation"] == 4 and raw["algo"] == "ppo"

    state, meta = load_checkpoint(path, STATE_DIM, ACTION_DIM)
    assert meta["phase"] == 2
    PolicyAgent(STATE_DIM, ACTION_DIM).net.load_state_dict(state)


# ── the saved model tracks the best of the phase being trained ──

class _ScriptedTrainer(Trainer):
    """Feeds `finish_generation` a fixed fitness so the bookkeeping is testable."""

    def __init__(self, *args, script=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.script = list(script)
        self.saved = []

    def _learn(self, cars):
        fitness = self.script.pop(0)
        return fitness, fitness / 2.0, {}

    def save(self, path=None):
        self.saved.append((self.generation, self.phase))
        super().save(path)


def _scripted(tmp_path, script, name="best.pt"):
    _, agent, sim = build(pop_size=4, total_laps=3, max_steps=5, seed=0)
    return _ScriptedTrainer(agent, sim, model_path=str(tmp_path / name),
                            log_dir=None, quiet=True, checkpoint_every=0,
                            script=script)


def test_phase_two_improvements_are_still_saved(tmp_path):
    """Phase 1 scores in the tens of thousands, phase 2 around a thousand.

    A single high-water mark carried across the switch rejected every phase-2
    improvement, so the saved model froze the moment the phase flipped.
    """
    trainer = _scripted(tmp_path, [31000.0, 2600.0, 2700.0, 2500.0, 2900.0])
    trainer.run_generation()                    # phase 1
    assert trainer.saved == [(1, 1)]
    trainer.phase = 2                           # route learned
    for _ in range(4):
        trainer.run_generation()
    # 2600 (first of the phase), 2700 and 2900 improve; 2500 does not
    assert [gen for gen, phase in trainer.saved if phase == 2] == [2, 3, 5]


def test_a_worse_generation_does_not_overwrite_the_best(tmp_path):
    trainer = _scripted(tmp_path, [500.0, 100.0, 400.0])
    for _ in range(3):
        trainer.run_generation()
    assert trainer.saved == [(1, 1)]
    assert trainer.best_fitness_ever == pytest.approx(500.0)


def test_resuming_does_not_clobber_a_better_saved_model(tmp_path):
    model = tmp_path / "resume.pt"
    first = _scripted(tmp_path, [900.0], name="resume.pt")
    first.run_generation()
    assert first.saved == [(1, 1)]

    _, agent, sim = build(pop_size=4, total_laps=3, max_steps=5, seed=1)
    second = _ScriptedTrainer(agent, sim, model_path=str(model), log_dir=None,
                              quiet=True, checkpoint_every=0, script=[100.0])
    second.load()
    assert second.best_fitness_ever == pytest.approx(900.0)
    second.run_generation()
    assert second.saved == []          # the weaker generation was not written


# ── checkpoint writes are atomic ─────────────────────────────

def test_a_failed_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    import ai
    torch.manual_seed(29)
    path = str(tmp_path / "best.pt")
    good = PolicyNet(STATE_DIM, ACTION_DIM)
    save_checkpoint(good.state_dict(), path, STATE_DIM, ACTION_DIM, generation=5)

    real_save = torch.save

    def explode(payload, target, *args, **kwargs):
        real_save(payload, target, *args, **kwargs)
        raise KeyboardInterrupt("interrupted mid-write")

    monkeypatch.setattr(ai.torch, "save", explode)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(PolicyNet(STATE_DIM, ACTION_DIM).state_dict(), path,
                        STATE_DIM, ACTION_DIM, generation=6)
    monkeypatch.undo()

    state, meta = load_checkpoint(path, STATE_DIM, ACTION_DIM)
    assert meta["generation"] == 5              # the old checkpoint survived
    assert not list(tmp_path.glob("*.tmp"))     # and no debris was left behind


# ── determinism of a whole training run ──────────────────────

def _fitness_trace(seed, generations=3):
    _, agent, sim = build(pop_size=4, total_laps=1, max_steps=25, seed=seed)
    trainer = Trainer(agent, sim, model_path=os.devnull, log_dir=None,
                      quiet=True, checkpoint_every=0)
    trainer.save = lambda *a, **k: None        # keep the run off disk
    return [(round(s.best_fitness, 6), round(s.avg_fitness, 6))
            for s in trainer.train(generations)]


def test_a_seeded_training_run_is_reproducible():
    """Covers the evolution RNG (tournament draws, crossover masks, mutation)."""
    assert _fitness_trace(31) == _fitness_trace(31)


def test_different_seeds_train_differently():
    assert _fitness_trace(31) != _fitness_trace(32)


# ── field identity ───────────────────────────────────────────

def test_every_car_in_the_field_reports_a_unique_id():
    _, _, sim = build(pop_size=12, max_steps=1, seed=0)
    ids = [c.telemetry()["car_id"] for c in sim.cars]
    assert ids == list(range(12))
    assert len({c.color for c in sim.cars}) > 1     # colours still cycle
