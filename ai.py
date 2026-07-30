import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from config import *


# ── shared network ───────────────────────────────────────────

class PolicyNet(nn.Module):
    """Actor-critic MLP. The genetic trainer evolves it; PPO gradient-trains it."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shared = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(HIDDEN_SIZE // 2, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(HIDDEN_SIZE // 2, 1)

    def forward(self, x):
        h = self.shared(x)
        return torch.tanh(self.actor_mean(h)), self.critic(h)

    def act(self, state, deterministic=False):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, val = self.forward(s)
            if deterministic:
                return mean.squeeze(0).numpy(), None, val.item()
            std = torch.exp(self.actor_log_std)
            dist = Normal(mean, std)
            a = torch.clamp(dist.sample(), -1.0, 1.0)
            lp = dist.log_prob(a).sum(-1)
            return a.squeeze(0).numpy(), lp.item(), val.item()

    def evaluate(self, states, actions):
        means, vals = self.forward(states)
        std = torch.exp(self.actor_log_std)
        dist = Normal(means, std)
        lps = dist.log_prob(actions).sum(-1)
        ent = dist.entropy().sum(-1)
        return lps, vals.squeeze(-1), ent


# ── batched population inference ─────────────────────────────

class BatchedPolicy:
    """A numpy snapshot of a whole population, evaluated in one shot.

    The genetic trainer holds the weights fixed for an entire generation, so
    per-car `torch` forward passes are pure overhead: this stacks every net's
    weights and drives the whole field with three matrix products per step.
    Output matches `PolicyNet.act(..., deterministic=True)`.
    """

    def __init__(self, nets):
        if not nets:
            raise ValueError("BatchedPolicy needs at least one network")

        def stack(getter):
            return np.stack([getter(n).detach().numpy().astype(np.float32)
                             for n in nets])

        self.w1 = stack(lambda n: n.shared[0].weight)     # (P, H, S)
        self.b1 = stack(lambda n: n.shared[0].bias)       # (P, H)
        self.w2 = stack(lambda n: n.shared[2].weight)     # (P, H2, H)
        self.b2 = stack(lambda n: n.shared[2].bias)
        self.wm = stack(lambda n: n.actor_mean.weight)    # (P, A, H2)
        self.bm = stack(lambda n: n.actor_mean.bias)
        self.pop_size = len(nets)

    def actions(self, states):
        """states: (P, state_dim) -> (P, action_dim) in [-1, 1]."""
        x = np.asarray(states, dtype=np.float32)
        if x.shape[0] != self.pop_size:
            raise ValueError(
                f"expected {self.pop_size} states, got {x.shape[0]}")
        h = np.maximum(np.einsum("phs,ps->ph", self.w1, x) + self.b1, 0.0)
        h = np.maximum(np.einsum("pkh,ph->pk", self.w2, h) + self.b2, 0.0)
        return np.tanh(np.einsum("pah,ph->pa", self.wm, h) + self.bm)


# ── checkpoint helpers ───────────────────────────────────────

def save_checkpoint(state_dict, path, state_dim, action_dim, **meta):
    """Write weights plus metadata so a run can be resumed exactly."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "state_dict": state_dict,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    payload.update(meta)
    torch.save(payload, path)
    return path


def load_checkpoint(path, state_dim=None, action_dim=None):
    """Read a checkpoint, tolerating the older bare-state_dict format.

    Returns (state_dict, metadata).
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        meta = {k: v for k, v in obj.items() if k != "state_dict"}
    else:                                    # legacy: raw state_dict
        state, meta = obj, {}

    saved_dim = meta.get("state_dim")
    if state_dim is not None and saved_dim is not None and saved_dim != state_dim:
        raise ValueError(
            f"checkpoint {path} was trained with state_dim={saved_dim}, "
            f"but the current config gives {state_dim}. Retrain or restore "
            "the matching config.")
    if state_dim is not None and saved_dim is None:
        # Legacy file: infer the input width from the first layer.
        first = state.get("shared.0.weight")
        if first is not None and first.shape[1] != state_dim:
            raise ValueError(
                f"checkpoint {path} expects state_dim={first.shape[1]}, "
                f"current config gives {state_dim}.")
    return state, meta


# ── genetic agent ────────────────────────────────────────────

class GeneticAgent:
    """Population-based neuroevolution with elitism + tournament selection + island restart.

    Phase 1 - route completion: fitness = laps x 10000 + checkpoints x 100 + speed
    Phase 2 - balanced multi-signal fitness (lap, time, speed, splits - all ~0-1000 scale)
    The trainer switches phase once any car completes TOTAL_LAPS.

    Plateau-breaking mechanisms:
    - Tournament selection (TOURN_K=3) instead of pure top-N elitism
    - Adaptive mutation std bump after STAG_LIM gens without improvement
    - Island restart: randomise bottom ISLAND_FRAC of population after ISLAND_STAG gens
    """

    def __init__(self, state_dim, action_dim, pop_size=POP_SIZE):
        if pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.pop_size   = pop_size
        self.nets = [PolicyNet(state_dim, action_dim) for _ in range(pop_size)]
        self.best_net   = copy.deepcopy(self.nets[0])
        self.generation = 0
        self.phase      = 1   # 1 = learn route, 2 = optimise speed
        self._stag      = 0
        self._last_best = -float('inf')
        self._batched   = None

    # ── interface ─────────────────────────────────────────────

    def get_action(self, car_idx: int, state):
        return self.nets[car_idx % self.pop_size].act(state, deterministic=True)[0]

    def get_actions(self, states):
        """Drive the whole population from a (pop_size, state_dim) array."""
        if self._batched is None:
            self._batched = BatchedPolicy(self.nets)
        return self._batched.actions(states)

    def _tournament(self, fitnesses: list) -> int:
        """Return one index via tournament selection (pick best of TOURN_K random candidates)."""
        candidates = random.sample(range(self.pop_size), min(TOURN_K, self.pop_size))
        return max(candidates, key=lambda i: fitnesses[i])

    def evolve(self, cars) -> tuple:
        """Score, select, reproduce.  Returns (best_fitness, avg_fitness)."""
        if len(cars) != self.pop_size:
            raise ValueError(f"expected {self.pop_size} cars, got {len(cars)}")

        self.generation += 1
        fitnesses = [self._fitness(c) for c in cars]
        order     = sorted(range(self.pop_size), key=lambda i: fitnesses[i], reverse=True)

        elite_n   = min(self.pop_size, max(2, int(self.pop_size * ELITE_FRACTION)))
        elite_idx = order[:elite_n]

        self.best_net = copy.deepcopy(self.nets[elite_idx[0]])

        # ── stagnation tracking + adaptive std ────────────────
        base_std = MUTATION_STD_P1 if self.phase == 1 else MUTATION_STD_P2
        top_fit  = fitnesses[order[0]]
        if top_fit > self._last_best + 0.1:
            self._last_best = top_fit
            self._stag = 0
        else:
            self._stag += 1
        std = base_std + (STAG_BUMP if self._stag >= STAG_LIM else 0.0)

        # ── island restart: inject fresh genes when deeply stagnant ──
        if self._stag >= ISLAND_STAG:
            regen_n = max(1, int(self.pop_size * ISLAND_FRAC))
            for i in order[self.pop_size - regen_n:]:
                self.nets[i] = PolicyNet(self.state_dim, self.action_dim)
            self._stag = 0
            self._last_best = top_fit

        # ── elites survive unchanged ──────────────────────────
        new_nets = [copy.deepcopy(self.nets[i]) for i in elite_idx]

        # ── fill rest via tournament selection ────────────────
        while len(new_nets) < self.pop_size:
            if random.random() < CROSSOVER_PROB:
                pa = self._tournament(fitnesses)
                pb = self._tournament(fitnesses)
                for _ in range(8):               # bounded retry; small pops may repeat
                    if pb != pa:
                        break
                    pb = self._tournament(fitnesses)
                child = self._crossover(self.nets[pa], self.nets[pb])
            else:
                child = copy.deepcopy(self.nets[self._tournament(fitnesses)])
            self._mutate(child, std)
            new_nets.append(child)

        self.nets = new_nets
        self._batched = None          # weights changed; drop the cached snapshot
        return top_fit, sum(fitnesses) / self.pop_size

    # ── internals ─────────────────────────────────────────────

    def _fitness(self, car) -> float:
        if self.phase == 1:
            return (car.lap * 10_000 +
                    car.total_cps * 100 +
                    car.speed * 5)

        # Phase 2: balanced multi-signal fitness.
        # All terms scaled to ~0-1000 so no single term dominates.
        score = 0.0

        # 1. Lap completion (capped so it doesn't drown other signals)
        score += min(car.lap, TOTAL_LAPS) * (1000.0 / max(TOTAL_LAPS, 1))

        # 2. Checkpoint progress as fraction of total possible
        cps_per_lap = getattr(car, "num_cps", 0) or 1
        total_possible = max(1, TOTAL_LAPS * cps_per_lap)
        score += min(1.0, car.total_cps / total_possible) * 500

        # 3. Best lap time (lower = better): 2s -> 880, 5s -> 700
        if car.best_time < float("inf"):
            score += max(0.0, 1000.0 - car.best_time * 60.0)

        # 4. Average speed carried through checkpoints (0 -> 0, MAX_SPEED -> 300)
        if car.cp_speeds:
            avg_spd = sum(car.cp_speeds) / len(car.cp_speeds)
            score += (avg_spd / MAX_SPEED) * 300

        # 5. Sector splits: short sectors score, inconsistent ones are penalised
        if car.cp_splits:
            avg_split = sum(car.cp_splits) / len(car.cp_splits)
            score += 300.0 * max(0.0, 1.0 - avg_split / SPLIT_REF_STEPS)
            if len(car.cp_splits) > 1:
                variance = sum((s - avg_split) ** 2 for s in car.cp_splits) / len(car.cp_splits)
                score -= math.sqrt(variance) * SPLIT_CONSISTENCY_WEIGHT

        # 6. Wall contacts are expensive
        score -= car.wall_hits * WALL_HIT_FITNESS_PENALTY

        return score

    def _mutate(self, net, std: float):
        with torch.no_grad():
            for p in net.parameters():
                mask  = torch.rand_like(p) < MUTATION_RATE
                noise = torch.randn_like(p) * std
                p.add_(mask.float() * noise)

    def _crossover(self, net_a, net_b):
        child = copy.deepcopy(net_a)
        with torch.no_grad():
            for pa, pb, pc in zip(net_a.parameters(),
                                   net_b.parameters(),
                                   child.parameters()):
                mask = torch.rand_like(pa) < 0.5
                pc.copy_(torch.where(mask, pa, pb))
        return child

    # ── persistence ───────────────────────────────────────────

    def save(self, path: str, **meta):
        meta.setdefault("generation", self.generation)
        meta.setdefault("phase", self.phase)
        return save_checkpoint(self.best_net.state_dict(), path,
                               self.state_dim, self.action_dim, **meta)

    def load(self, path: str):
        """Seed the whole population from a checkpoint. Returns its metadata."""
        state, meta = load_checkpoint(path, self.state_dim, self.action_dim)
        for net in self.nets:
            net.load_state_dict(copy.deepcopy(state))
        self.best_net.load_state_dict(state)
        self._batched = None
        self.generation = meta.get("generation", self.generation)
        self.phase = meta.get("phase", self.phase)
        return meta


# ── single-network agent (replay / evaluation) ───────────────

class PolicyAgent:
    """One network driving one car — used for replay and scored evaluation."""

    def __init__(self, state_dim, action_dim, pop_size=1, deterministic=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pop_size = pop_size
        self.deterministic = deterministic
        self.net = PolicyNet(state_dim, action_dim)
        self.generation = 0
        self.phase = 1

    def get_action(self, car_idx, state):
        return self.net.act(state, deterministic=self.deterministic)[0]

    def save(self, path: str, **meta):
        meta.setdefault("generation", self.generation)
        meta.setdefault("phase", self.phase)
        return save_checkpoint(self.net.state_dict(), path,
                               self.state_dim, self.action_dim, **meta)

    def load(self, path: str):
        state, meta = load_checkpoint(path, self.state_dim, self.action_dim)
        self.net.load_state_dict(state)
        self.generation = meta.get("generation", self.generation)
        self.phase = meta.get("phase", self.phase)
        return meta
