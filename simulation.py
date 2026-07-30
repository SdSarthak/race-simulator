"""Headless simulation and training loop.

Nothing in this module imports a rendering library, so training runs on a
server, in CI or inside tests. `main.py` drives it directly for headless runs;
`renderer.py` drives the very same objects and draws between ticks.
"""

import csv
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from config import (
    POP_SIZE, TOTAL_LAPS, MAX_STEPS_GEN, NUM_GENERATIONS,
    STATE_DIM, ACTION_DIM, CAR_COLORS,
    BEST_MODEL, MODEL_DIR, LOG_DIR, CHECKPOINT_EVERY,
)
from track import Track
from car import Car


def set_seed(seed):
    """Seed python, numpy and torch. Returns the seed (None is a no-op)."""
    if seed is None:
        return None
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    return seed


@dataclass
class GenerationStats:
    """One row of the training log."""
    generation: int
    phase: int
    steps: int
    best_fitness: float
    avg_fitness: float
    best_reward: float
    best_checkpoints: int
    laps_completed: int
    finished_cars: int
    alive_cars: int
    best_lap_time: float
    wall_hits: int
    policy_loss: float = 0.0     # PPO only; zero for the genetic trainer
    value_loss: float = 0.0

    def as_row(self):
        return asdict(self)

    def summary(self):
        bt = (f"{self.best_lap_time:.2f}s"
              if self.best_lap_time < float("inf") else "--")
        return (f"Gen {self.generation:>4} | Ph{self.phase} | "
                f"Done {self.finished_cars:>2} | BestCP {self.best_checkpoints:>3} | "
                f"BestTime {bt:>8} | Fit {self.best_fitness:>9.0f} | "
                f"Alive {self.alive_cars:>2} | Hits {self.wall_hits:>3}")


class Simulation:
    """A population of cars driving one generation on a track."""

    def __init__(self, track=None, agent=None, pop_size=POP_SIZE,
                 total_laps=TOTAL_LAPS, max_steps=MAX_STEPS_GEN):
        self.track = track if track is not None else Track()
        self.agent = agent
        self.pop_size = pop_size
        self.total_laps = total_laps
        self.max_steps = max_steps
        self.cars = []
        self.step_count = 0
        self.generation_done = True
        self.observer = None      # optional per-tick hook (see PPOTrainer)
        self.reset()

    # ── spawning ─────────────────────────────────────────────

    def _grid_positions(self):
        """Staggered starting grid on the straight between CP 0 and CP 1."""
        m0 = self.track.checkpoint_midpoint(0)
        m1 = self.track.checkpoint_midpoint(1)

        adx, ady = m1[0] - m0[0], m1[1] - m0[1]
        alen = max(1e-9, (adx * adx + ady * ady) ** 0.5)
        adx, ady = adx / alen, ady / alen
        apx, apy = -ady, adx                       # lateral (across the road)

        road_w = self.track.checkpoint_width(0)
        cols = 4
        col_gap = road_w * 0.6 / max(1, cols - 1)
        row_gap = 30
        angle = np.degrees(np.arctan2(ady, adx))

        positions = []
        for i in range(self.pop_size):
            col, row = i % cols, i // cols
            lat = (col - (cols - 1) / 2.0) * col_gap
            lon = (row + 1) * row_gap
            positions.append((m0[0] + adx * lon + apx * lat,
                              m0[1] + ady * lon + apy * lat))
        return positions, float(angle)

    def reset(self):
        """Spawn a fresh field and prime every car's sensors."""
        positions, angle = self._grid_positions()
        self.cars = [Car(pos, angle, i % len(CAR_COLORS))
                     for i, pos in enumerate(positions)]
        for car in self.cars:
            car.cast_rays(self.track)
        self.step_count = 0
        self.generation_done = False
        self.last_states = np.zeros((len(self.cars), STATE_DIM), dtype=np.float32)
        self.last_actions = np.zeros((len(self.cars), ACTION_DIM), dtype=np.float32)
        self.last_rewards = np.zeros(len(self.cars), dtype=np.float32)
        self.last_active = []
        return self.cars

    # ── stepping ─────────────────────────────────────────────

    def _car_active(self, car):
        return car.alive and car.lap < self.total_laps

    def tick(self):
        """Advance every active car by one physics step.

        Returns True once the generation is over (all cars done or out of time).
        """
        if self.generation_done:
            return True

        n = len(self.cars)
        active = [i for i, car in enumerate(self.cars) if self._car_active(car)]
        states = np.zeros((n, STATE_DIM), dtype=np.float32)
        actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)

        if active:
            batched = getattr(self.agent, "get_actions", None)
            if batched is not None:
                # One matrix pass drives the whole field; idle rows stay zeroed.
                for i in active:
                    states[i] = self.cars[i].get_state(self.track)
                actions = np.asarray(batched(states), dtype=np.float32)
            else:
                for i in active:
                    states[i] = self.cars[i].get_state(self.track)
                    actions[i] = self.agent.get_action(i, states[i])

            for i in active:
                rewards[i] = self.cars[i].step(
                    self.track, float(actions[i][0]), float(actions[i][1]))

        # Kept so learners that need transitions (PPO) can read them back.
        self.last_states = states
        self.last_actions = actions
        self.last_rewards = rewards
        self.last_active = active

        any_active = bool(active)
        self.step_count += 1
        if not any_active or self.step_count >= self.max_steps:
            self.generation_done = True
        if self.observer is not None:
            self.observer(self)
        return self.generation_done

    def run(self):
        """Run the generation to completion. Returns the number of steps taken."""
        while not self.tick():
            pass
        return self.step_count

    # ── inspection ───────────────────────────────────────────

    def best_index(self):
        """Index of the car currently leading (alive > progress > speed)."""
        if not self.cars:
            return -1
        return max(range(len(self.cars)),
                   key=lambda i: (self.cars[i].alive,
                                  self.cars[i].total_cps,
                                  self.cars[i].speed))

    def best_car(self):
        idx = self.best_index()
        return self.cars[idx] if idx >= 0 else None

    def best_lap_time(self):
        times = [c.best_time for c in self.cars if c.best_time < float("inf")]
        return min(times) if times else float("inf")


class Trainer:
    """Drives generations, evolves the population and persists progress."""

    def __init__(self, agent, simulation, model_path=BEST_MODEL,
                 log_dir=LOG_DIR, quiet=False, checkpoint_every=CHECKPOINT_EVERY):
        self.agent = agent
        self.sim = simulation
        self.model_path = model_path
        self.log_dir = log_dir
        self.quiet = quiet
        self.checkpoint_every = checkpoint_every

        self.generation = getattr(agent, "generation", 0)
        self.phase = getattr(agent, "phase", 1)
        self.best_fitness_ever = -float("inf")
        self.global_best_time = float("inf")
        self.phase_changed = False
        self.history = []
        self._log_path = None

    # ── logging ──────────────────────────────────────────────

    def _log(self, stats):
        self.history.append(stats)
        if not self.quiet:
            print(stats.summary(), flush=True)
        if self.log_dir is None:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            if self._log_path is None:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                self._log_path = os.path.join(self.log_dir, f"training-{stamp}.csv")
            new_file = not os.path.exists(self._log_path)
            with open(self._log_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(stats.as_row()))
                if new_file:
                    writer.writeheader()
                writer.writerow(stats.as_row())
        except OSError as exc:  # logging must never kill a training run
            if not self.quiet:
                print(f"warning: could not write training log ({exc})")
            self.log_dir = None

    # ── generation lifecycle ─────────────────────────────────

    def begin_generation(self):
        self.agent.phase = self.phase
        self.phase_changed = False
        return self.sim.reset()

    def _learn(self, cars):
        """Turn a finished generation into a new policy.

        Returns (best_fitness, avg_fitness, extra_stats).
        """
        best_fit, avg_fit = self.agent.evolve(cars)
        return best_fit, avg_fit, {}

    def finish_generation(self):
        """Score the field, learn from it, persist, and return the stats."""
        cars = self.sim.cars
        self.generation += 1
        self.agent.generation = self.generation

        best_fit, avg_fit, extra = self._learn(cars)

        best_time = self.sim.best_lap_time()
        self.global_best_time = min(self.global_best_time, best_time)
        finished = sum(1 for c in cars if c.lap >= self.sim.total_laps)

        stats = GenerationStats(
            generation=self.generation,
            phase=self.phase,
            steps=self.sim.step_count,
            best_fitness=best_fit,
            avg_fitness=avg_fit,
            best_reward=max((c.total_reward for c in cars), default=0.0),
            best_checkpoints=max((c.total_cps for c in cars), default=0),
            laps_completed=max((c.lap for c in cars), default=0),
            finished_cars=finished,
            alive_cars=sum(1 for c in cars if c.alive),
            best_lap_time=self.global_best_time,
            wall_hits=sum(c.wall_hits for c in cars),
            policy_loss=extra.get("policy_loss", 0.0),
            value_loss=extra.get("value_loss", 0.0),
        )

        if best_fit > self.best_fitness_ever:
            self.best_fitness_ever = best_fit
            self.save()

        if (self.checkpoint_every and
                self.generation % self.checkpoint_every == 0):
            os.makedirs(MODEL_DIR, exist_ok=True)
            self.save(os.path.join(MODEL_DIR, f"gen_{self.generation}.pt"))

        # Phase 1 teaches the route; phase 2 optimises lap time.
        if self.phase == 1 and finished > 0:
            self.phase = 2
            self.agent.phase = 2
            self.phase_changed = True
            if not self.quiet:
                print(f"\n{'=' * 55}\n  PHASE 2 UNLOCKED at generation "
                      f"{self.generation}\n{'=' * 55}\n", flush=True)

        self._log(stats)
        return stats

    def run_generation(self):
        """Headless convenience: spawn, simulate to the end, evolve."""
        self.begin_generation()
        self.sim.run()
        return self.finish_generation()

    def train(self, generations=NUM_GENERATIONS):
        for _ in range(generations):
            self.run_generation()
        self.save()
        return self.history

    # ── persistence ──────────────────────────────────────────

    def save(self, path=None):
        self.agent.save(path or self.model_path,
                        generation=self.generation, phase=self.phase)

    def load(self, path=None):
        """Restore weights and, when present, generation/phase metadata."""
        meta = self.agent.load(path or self.model_path)
        self.generation = meta.get("generation", self.generation)
        self.phase = meta.get("phase", self.phase)
        self.agent.generation = self.generation
        self.agent.phase = self.phase
        return meta


class PPOTrainer(Trainer):
    """Trainer variant that learns by gradient descent instead of evolution.

    Every car is a parallel environment for one shared network. Transitions are
    recorded through the simulation's per-tick observer hook, so the rendered
    and headless loops both feed the learner without knowing about it.
    """

    def __init__(self, agent, simulation, **kwargs):
        super().__init__(agent, simulation, **kwargs)
        self._buffers = []
        self._active_before = []
        simulation.observer = self._record

    # ── rollout collection ───────────────────────────────────

    def begin_generation(self):
        cars = super().begin_generation()
        self._buffers = [{"states": [], "actions": [], "log_probs": [],
                          "values": [], "rewards": [], "dones": []}
                         for _ in cars]
        return cars

    def _record(self, sim):
        log_probs = getattr(self.agent, "last_log_probs", None)
        values = getattr(self.agent, "last_values", None)
        if log_probs is None or values is None:
            return
        for i in sim.last_active:
            done = not sim._car_active(sim.cars[i])
            buf = self._buffers[i]
            buf["states"].append(sim.last_states[i])
            buf["actions"].append(sim.last_actions[i])
            buf["log_probs"].append(float(log_probs[i]))
            buf["values"].append(float(values[i]))
            buf["rewards"].append(float(sim.last_rewards[i]))
            buf["dones"].append(1.0 if done else 0.0)

    # ── learning ─────────────────────────────────────────────

    def _learn(self, cars):
        from ai import compute_gae

        states, actions, log_probs, advantages, returns = [], [], [], [], []
        for i, buf in enumerate(self._buffers):
            if not buf["rewards"]:
                continue
            # A car still running when the generation was cut short gets a
            # bootstrapped value; one that died or finished ends at zero.
            if buf["dones"][-1] >= 1.0:
                last_value = 0.0
            else:
                last_value = float(self.agent.value_of(
                    cars[i].get_state(self.sim.track)[None, :])[0])

            adv, ret = compute_gae(buf["rewards"], buf["values"],
                                   buf["dones"], last_value)
            states.extend(buf["states"])
            actions.extend(buf["actions"])
            log_probs.extend(buf["log_probs"])
            advantages.extend(adv)
            returns.extend(ret)

        losses = self.agent.update(states, actions, log_probs,
                                   advantages, returns)
        rewards = [c.total_reward for c in cars]
        best = max(rewards, default=0.0)
        avg = sum(rewards) / len(rewards) if rewards else 0.0
        return best, avg, losses


def build(agent_cls=None, layout="circuit", pop_size=POP_SIZE,
          total_laps=TOTAL_LAPS, max_steps=MAX_STEPS_GEN, seed=None):
    """Create a (track, agent, simulation) triple with consistent dimensions."""
    set_seed(seed)
    from ai import GeneticAgent  # imported lazily to keep torch off cheap paths

    agent_cls = agent_cls or GeneticAgent
    track = Track(layout)
    agent = agent_cls(STATE_DIM, ACTION_DIM, pop_size=pop_size)
    sim = Simulation(track, agent, pop_size=pop_size,
                     total_laps=total_laps, max_steps=max_steps)
    return track, agent, sim
