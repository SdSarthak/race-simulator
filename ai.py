import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from config import *


# ── shared network ───────────────────────────────────────────

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
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
            s = torch.FloatTensor(state).unsqueeze(0)
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


# ── genetic agent ────────────────────────────────────────────

class GeneticAgent:
    """Population-based neuroevolution with elitism + tournament selection + island restart.

    Phase 1 – route completion: fitness = laps × 10000 + checkpoints × 100 + speed
    Phase 2 – balanced multi-signal fitness (lap, time, speed, splits — all ~0-1000 scale)
    Auto-switches when any car completes TOTAL_LAPS in phase 1.

    Plateau-breaking mechanisms:
    - Tournament selection (TOURN_K=3) instead of pure top-N elitism
    - Adaptive mutation std bump after STAG_LIM gens without improvement
    - Island restart: randomise bottom ISLAND_FRAC of population after ISLAND_STAG gens
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.nets = [PolicyNet(state_dim, action_dim) for _ in range(POP_SIZE)]
        self.best_net   = copy.deepcopy(self.nets[0])
        self.generation = 0
        self.phase      = 1   # 1 = learn route, 2 = optimise speed
        self._stag      = 0
        self._last_best = -float('inf')

    # ── interface ─────────────────────────────────────────────

    def get_action(self, car_idx: int, state):
        return self.nets[car_idx].act(state, deterministic=True)[0]

    def _tournament(self, fitnesses: list) -> int:
        """Return one index via tournament selection (pick best of TOURN_K random candidates)."""
        candidates = random.sample(range(POP_SIZE), min(TOURN_K, POP_SIZE))
        return max(candidates, key=lambda i: fitnesses[i])

    def evolve(self, cars) -> tuple:
        """Score, select, reproduce.  Returns (best_fitness, avg_fitness)."""
        self.generation += 1
        fitnesses = [self._fitness(c) for c in cars]
        order     = sorted(range(POP_SIZE), key=lambda i: fitnesses[i], reverse=True)

        elite_n   = max(2, int(POP_SIZE * ELITE_FRACTION))
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
            regen_n = max(1, int(POP_SIZE * ISLAND_FRAC))
            for i in order[POP_SIZE - regen_n:]:
                self.nets[i] = PolicyNet(self.state_dim, self.action_dim)
            self._stag = 0
            self._last_best = top_fit

        # ── elites survive unchanged ──────────────────────────
        new_nets = [copy.deepcopy(self.nets[i]) for i in elite_idx]

        # ── fill rest via tournament selection ────────────────
        while len(new_nets) < POP_SIZE:
            if random.random() < CROSSOVER_PROB:
                pa = self._tournament(fitnesses)
                pb = self._tournament(fitnesses)
                while pb == pa:
                    pb = self._tournament(fitnesses)
                child = self._crossover(self.nets[pa], self.nets[pb])
            else:
                child = copy.deepcopy(self.nets[self._tournament(fitnesses)])
            self._mutate(child, std)
            new_nets.append(child)

        self.nets = new_nets
        return top_fit, sum(fitnesses) / POP_SIZE

    # ── internals ─────────────────────────────────────────────

    def _fitness(self, car) -> float:
        if self.phase == 1:
            return (car.lap * 10_000 +
                    car.total_cps * 100 +
                    car.speed * 5)

        # Phase 2: balanced multi-signal fitness.
        # All terms scaled to ~0-1000 so no single term dominates.
        import math
        score = 0.0

        # 1. Lap completion (capped so it doesn't drown other signals)
        score += min(car.lap, TOTAL_LAPS) * (1000.0 / TOTAL_LAPS)

        # 2. Checkpoint progress as fraction of total possible
        total_possible = TOTAL_LAPS * (getattr(car, '_num_cps', 7))
        score += (car.total_cps / max(total_possible, 1)) * 500

        # 3. Best lap time (lower = better), ~2s→880, ~5s→700
        if car.best_time < float("inf"):
            score += max(0.0, 1000.0 - car.best_time * 60.0)

        # 4. Average speed at checkpoints (0→0, MAX_SPEED→300)
        if hasattr(car, 'cp_speeds') and car.cp_speeds:
            from config import MAX_SPEED
            avg_spd = sum(car.cp_speeds) / len(car.cp_speeds)
            score += (avg_spd / MAX_SPEED) * 300

        # 5. Sector split speed and consistency
        if hasattr(car, 'cp_splits') and car.cp_splits:
            avg_split = sum(car.cp_splits) / len(car.cp_splits)
            score += max(0.0, 300.0 - avg_split) * (300.0 / 220.0)
            if len(car.cp_splits) > 1:
                variance = sum((s - avg_split) ** 2 for s in car.cp_splits) / len(car.cp_splits)
                score -= math.sqrt(variance) * 0.3

        # 6. Wall hit penalty
        if hasattr(car, 'wall_hits'):
            score -= car.wall_hits * 10

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

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.best_net.state_dict(), path)

    def load(self, path: str):
        sd = torch.load(path, map_location="cpu")
        for net in self.nets:
            net.load_state_dict(copy.deepcopy(sd))
        self.best_net.load_state_dict(sd)


# ── PPO agent (kept for replay only) ─────────────────────────

class PPOAgent:
    """Single-network agent used only for deterministic replay."""

    def __init__(self, state_dim, action_dim):
        self.net = PolicyNet(state_dim, action_dim)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
