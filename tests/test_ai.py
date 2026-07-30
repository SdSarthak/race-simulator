import copy

import numpy as np
import pytest
import torch

from config import STATE_DIM, ACTION_DIM, TOTAL_LAPS
from ai import (
    PolicyNet, BatchedPolicy, GeneticAgent, PolicyAgent,
    save_checkpoint, load_checkpoint,
)


class FakeCar:
    """Minimal stand-in with just the fields the fitness function reads."""

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


def _params(net):
    return [p.detach().clone() for p in net.parameters()]


def _same(net_a, net_b):
    return all(torch.equal(a, b) for a, b in zip(_params(net_a), _params(net_b)))


# ── network ──────────────────────────────────────────────────

def test_deterministic_action_is_bounded_and_repeatable():
    torch.manual_seed(0)
    net = PolicyNet(STATE_DIM, ACTION_DIM)
    state = np.random.RandomState(0).rand(STATE_DIM).astype(np.float32)
    a1, logp, value = net.act(state, deterministic=True)
    a2, _, _ = net.act(state, deterministic=True)
    assert a1.shape == (ACTION_DIM,)
    assert np.all(np.abs(a1) <= 1.0)
    assert logp is None and isinstance(value, float)
    assert np.array_equal(a1, a2)


def test_sampled_actions_stay_in_range_and_report_log_prob():
    torch.manual_seed(1)
    net = PolicyNet(STATE_DIM, ACTION_DIM)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    action, logp, _ = net.act(state, deterministic=False)
    assert np.all(np.abs(action) <= 1.0)
    assert isinstance(logp, float)


def test_evaluate_returns_log_probs_values_and_entropy():
    torch.manual_seed(2)
    net = PolicyNet(STATE_DIM, ACTION_DIM)
    states = torch.zeros(5, STATE_DIM)
    actions = torch.zeros(5, ACTION_DIM)
    logps, values, entropy = net.evaluate(states, actions)
    assert logps.shape == (5,) and values.shape == (5,) and entropy.shape == (5,)


# ── batched inference ────────────────────────────────────────

def test_batched_policy_matches_per_net_forward():
    torch.manual_seed(3)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=6)
    states = np.random.RandomState(1).rand(6, STATE_DIM).astype(np.float32)

    batched = agent.get_actions(states)
    single = np.stack([agent.get_action(i, states[i]) for i in range(6)])

    assert batched.shape == (6, ACTION_DIM)
    assert np.allclose(batched, single, atol=1e-5)


def test_batched_policy_rejects_a_mismatched_batch():
    torch.manual_seed(4)
    policy = BatchedPolicy([PolicyNet(STATE_DIM, ACTION_DIM) for _ in range(3)])
    with pytest.raises(ValueError):
        policy.actions(np.zeros((2, STATE_DIM), dtype=np.float32))


def test_batched_snapshot_is_refreshed_after_evolution():
    torch.manual_seed(5)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=4)
    states = np.zeros((4, STATE_DIM), dtype=np.float32)
    before = agent.get_actions(states).copy()
    agent.evolve([FakeCar(total_cps=i) for i in range(4)])
    after = agent.get_actions(states)
    # the elite is preserved, so at least one row must still match a fresh
    # per-net forward — what matters is that the cache was rebuilt
    fresh = np.stack([agent.get_action(i, states[i]) for i in range(4)])
    assert np.allclose(after, fresh, atol=1e-5)
    assert before.shape == after.shape


# ── genetic agent ────────────────────────────────────────────

def test_population_size_must_be_sane():
    with pytest.raises(ValueError):
        GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=1)


def test_evolve_rejects_a_mismatched_field():
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=4)
    with pytest.raises(ValueError):
        agent.evolve([FakeCar()])


def test_evolve_keeps_population_size_and_reports_fitness():
    torch.manual_seed(6)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=8)
    cars = [FakeCar(total_cps=i) for i in range(8)]
    best, avg = agent.evolve(cars)
    assert len(agent.nets) == 8
    assert best == pytest.approx(700.0)          # 7 checkpoints x 100
    assert avg < best
    assert agent.generation == 1


def test_the_fittest_network_is_carried_over_untouched():
    torch.manual_seed(7)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=6)
    winner = copy.deepcopy(agent.nets[3])
    cars = [FakeCar(total_cps=1) for _ in range(6)]
    cars[3] = FakeCar(total_cps=99)
    agent.evolve(cars)
    assert _same(agent.best_net, winner)
    assert any(_same(net, winner) for net in agent.nets)


def test_children_differ_from_their_parents():
    torch.manual_seed(8)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=6)
    originals = [copy.deepcopy(n) for n in agent.nets]
    agent.evolve([FakeCar(total_cps=i) for i in range(6)])
    changed = sum(0 if any(_same(new, old) for old in originals) else 1
                  for new in agent.nets)
    assert changed > 0


def test_crossover_takes_every_weight_from_one_parent_or_the_other():
    torch.manual_seed(9)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    a, b = agent.nets
    with torch.no_grad():
        for p in a.parameters():
            p.fill_(1.0)
        for p in b.parameters():
            p.fill_(-1.0)
    child = agent._crossover(a, b)
    for p in child.parameters():
        assert torch.all((p == 1.0) | (p == -1.0))


def test_stagnation_triggers_an_island_restart():
    torch.manual_seed(10)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=6)
    cars = [FakeCar(total_cps=1) for _ in range(6)]
    for _ in range(3):
        agent.evolve(cars)
    stagnant_before = agent._stag
    assert stagnant_before > 0
    agent._stag = 10 ** 6          # deep stagnation
    agent.evolve(cars)
    assert agent._stag == 0        # restart resets the counter


# ── fitness ──────────────────────────────────────────────────

def test_phase_one_fitness_ranks_progress_first():
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    lapper = FakeCar(lap=1, total_cps=17)
    crawler = FakeCar(lap=0, total_cps=16, speed=20.0)
    assert agent._fitness(lapper) > agent._fitness(crawler)


def test_phase_two_prefers_the_faster_cleaner_car():
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    agent.phase = 2
    fast = FakeCar(lap=TOTAL_LAPS, total_cps=51, best_time=4.0,
                   cp_speeds=[18.0] * 5, cp_splits=[20] * 5, wall_hits=0)
    slow = FakeCar(lap=TOTAL_LAPS, total_cps=51, best_time=9.0,
                   cp_speeds=[8.0] * 5, cp_splits=[60] * 5, wall_hits=6)
    assert agent._fitness(fast) > agent._fitness(slow)


def test_phase_two_penalises_wall_contacts():
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    agent.phase = 2
    clean = FakeCar(lap=1, total_cps=17, best_time=6.0, wall_hits=0)
    scruffy = FakeCar(lap=1, total_cps=17, best_time=6.0, wall_hits=5)
    assert agent._fitness(clean) > agent._fitness(scruffy)


def test_phase_two_handles_a_car_that_never_scored():
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    agent.phase = 2
    assert agent._fitness(FakeCar()) == pytest.approx(0.0)


# ── persistence ──────────────────────────────────────────────

def test_checkpoint_round_trip_preserves_weights_and_metadata(tmp_path):
    torch.manual_seed(11)
    agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=4)
    agent.generation, agent.phase = 12, 2
    path = str(tmp_path / "best.pt")
    agent.save(path)

    restored = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=4)
    meta = restored.load(path)

    assert meta["generation"] == 12 and meta["phase"] == 2
    assert restored.generation == 12 and restored.phase == 2
    assert _same(restored.best_net, agent.best_net)
    assert all(_same(net, agent.best_net) for net in restored.nets)


def test_legacy_bare_state_dicts_still_load(tmp_path):
    torch.manual_seed(12)
    net = PolicyNet(STATE_DIM, ACTION_DIM)
    path = str(tmp_path / "legacy.pt")
    torch.save(net.state_dict(), path)          # old format: no metadata

    agent = PolicyAgent(STATE_DIM, ACTION_DIM)
    meta = agent.load(path)
    assert meta == {}
    assert _same(agent.net, net)


def test_a_checkpoint_from_another_state_size_is_refused(tmp_path):
    torch.manual_seed(13)
    other = PolicyNet(STATE_DIM + 3, ACTION_DIM)
    path = str(tmp_path / "wrong.pt")
    save_checkpoint(other.state_dict(), path, STATE_DIM + 3, ACTION_DIM)

    with pytest.raises(ValueError, match="state_dim"):
        load_checkpoint(path, STATE_DIM, ACTION_DIM)


def test_a_legacy_checkpoint_of_the_wrong_width_is_refused(tmp_path):
    torch.manual_seed(14)
    other = PolicyNet(STATE_DIM + 3, ACTION_DIM)
    path = str(tmp_path / "legacy-wrong.pt")
    torch.save(other.state_dict(), path)

    with pytest.raises(ValueError, match="state_dim"):
        load_checkpoint(path, STATE_DIM, ACTION_DIM)


def test_policy_agent_drives_from_a_saved_genetic_checkpoint(tmp_path):
    torch.manual_seed(15)
    trainer_agent = GeneticAgent(STATE_DIM, ACTION_DIM, pop_size=3)
    path = str(tmp_path / "best.pt")
    trainer_agent.save(path)

    driver = PolicyAgent(STATE_DIM, ACTION_DIM)
    driver.load(path)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    assert np.allclose(driver.get_action(0, state),
                       trainer_agent.best_net.act(state, deterministic=True)[0])


def test_stochastic_policy_agent_varies_its_actions():
    torch.manual_seed(16)
    agent = PolicyAgent(STATE_DIM, ACTION_DIM, deterministic=False)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    actions = np.stack([agent.get_action(0, state) for _ in range(5)])
    assert actions.std() > 0
