import numpy as np
import pytest
import torch

from config import STATE_DIM, ACTION_DIM, GAMMA, GAE_LAMBDA
from ai import PPOAgent, compute_gae
from simulation import Simulation, PPOTrainer, set_seed
from track import Track


def _params(agent):
    return [p.detach().clone() for p in agent.net.parameters()]


def _changed(before, after):
    return any(not torch.equal(a, b) for a, b in zip(before, after))


# ── GAE ──────────────────────────────────────────────────────

def test_advantage_of_a_perfectly_predicted_trajectory_is_zero():
    # values already equal the discounted returns, so there is nothing to learn
    rewards = [0.0, 0.0, 1.0]
    values = [GAMMA ** 2, GAMMA, 1.0]
    adv, ret = compute_gae(rewards, values, [0.0, 0.0, 1.0], last_value=0.0)
    assert np.allclose(adv, 0.0, atol=1e-9)
    assert np.allclose(ret, values, atol=1e-9)


def test_an_unexpected_reward_produces_positive_advantage():
    adv, _ = compute_gae([1.0], [0.0], [1.0], last_value=0.0)
    assert adv[0] == pytest.approx(1.0)


def test_terminal_steps_do_not_bootstrap():
    ended, _ = compute_gae([0.0], [0.0], [1.0], last_value=100.0)
    cut_short, _ = compute_gae([0.0], [0.0], [0.0], last_value=100.0)
    assert ended[0] == pytest.approx(0.0)
    assert cut_short[0] == pytest.approx(GAMMA * 100.0)


def test_advantages_decay_backwards_through_time():
    adv, _ = compute_gae([0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0], last_value=0.0)
    assert adv[2] == pytest.approx(1.0)
    assert adv[1] == pytest.approx(GAMMA * GAE_LAMBDA)
    assert adv[0] == pytest.approx((GAMMA * GAE_LAMBDA) ** 2)


def test_mismatched_trajectory_lengths_are_rejected():
    with pytest.raises(ValueError):
        compute_gae([1.0, 2.0], [0.0], [0.0], last_value=0.0)


# ── agent ────────────────────────────────────────────────────

def test_sampled_actions_are_bounded_and_carry_log_probs():
    set_seed(0)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=5)
    actions = agent.get_actions(np.zeros((5, STATE_DIM), dtype=np.float32))
    assert actions.shape == (5, ACTION_DIM)
    assert np.all(np.abs(actions) <= 1.0)
    assert agent.last_log_probs.shape == (5,)
    assert agent.last_values.shape == (5,)


def test_sampling_explores_but_the_deterministic_mode_does_not():
    set_seed(1)
    states = np.zeros((4, STATE_DIM), dtype=np.float32)
    stochastic = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=4)
    assert stochastic.get_actions(states).std() > 0

    greedy = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=4, deterministic=True)
    assert np.array_equal(greedy.get_actions(states), greedy.get_actions(states))


def test_an_update_moves_the_weights():
    set_seed(2)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=4)
    n = 40
    states = np.random.RandomState(0).rand(n, STATE_DIM).astype(np.float32)
    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    log_probs = np.zeros(n, dtype=np.float32)
    advantages = np.random.RandomState(1).randn(n).astype(np.float32)
    returns = np.random.RandomState(2).randn(n).astype(np.float32)

    before = _params(agent)
    losses = agent.update(states, actions, log_probs, advantages, returns)
    assert _changed(before, _params(agent))
    assert set(losses) == {"policy_loss", "value_loss", "entropy"}
    assert all(np.isfinite(v) for v in losses.values())


def test_an_empty_update_is_harmless():
    set_seed(3)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=2)
    before = _params(agent)
    losses = agent.update([], [], [], [], [])
    assert losses["policy_loss"] == 0.0
    assert not _changed(before, _params(agent))


def test_the_critic_learns_a_constant_return():
    """Value loss must fall when every state shares the same return."""
    set_seed(4)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=4)
    n = 64
    states = np.random.RandomState(3).rand(n, STATE_DIM).astype(np.float32)
    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    log_probs = np.zeros(n, dtype=np.float32)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.full(n, 5.0, dtype=np.float32)

    first = agent.update(states, actions, log_probs, advantages, returns)
    for _ in range(15):
        last = agent.update(states, actions, log_probs, advantages, returns)
    assert last["value_loss"] < first["value_loss"]


def test_checkpoints_round_trip(tmp_path):
    set_seed(5)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=3)
    path = str(tmp_path / "ppo.pt")
    agent.generation = 7
    agent.save(path)

    restored = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=3)
    meta = restored.load(path)
    assert meta["algo"] == "ppo"
    assert restored.generation == 7
    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    assert np.allclose(restored.value_of(state), agent.value_of(state))


# ── trainer ──────────────────────────────────────────────────

def _ppo_trainer(tmp_path, pop=4, steps=40):
    set_seed(6)
    agent = PPOAgent(STATE_DIM, ACTION_DIM, pop_size=pop)
    sim = Simulation(Track(), agent, pop_size=pop, total_laps=1, max_steps=steps)
    return agent, sim, PPOTrainer(agent, sim, model_path=str(tmp_path / "ppo.pt"),
                                  log_dir=None, quiet=True, checkpoint_every=0)


def test_a_generation_collects_transitions_and_updates(tmp_path):
    agent, sim, trainer = _ppo_trainer(tmp_path)
    before = _params(agent)
    stats = trainer.run_generation()

    collected = sum(len(b["rewards"]) for b in trainer._buffers)
    assert collected > 0
    assert collected <= sim.max_steps * len(sim.cars)
    assert _changed(before, _params(agent))
    assert stats.generation == 1
    assert np.isfinite(stats.policy_loss) and np.isfinite(stats.value_loss)


def test_reported_fitness_is_the_collected_reward(tmp_path):
    _, sim, trainer = _ppo_trainer(tmp_path)
    stats = trainer.run_generation()
    assert stats.best_fitness == pytest.approx(
        max(c.total_reward for c in sim.cars))


def test_buffers_are_cleared_between_generations(tmp_path):
    _, _, trainer = _ppo_trainer(tmp_path, steps=20)
    trainer.run_generation()
    first = sum(len(b["rewards"]) for b in trainer._buffers)
    trainer.run_generation()
    second = sum(len(b["rewards"]) for b in trainer._buffers)
    assert first > 0 and second > 0
    assert second <= first * 2      # not accumulating across generations


def test_recorded_steps_match_the_simulation(tmp_path):
    _, sim, trainer = _ppo_trainer(tmp_path, pop=3, steps=12)
    trainer.begin_generation()
    sim.tick()
    sim.tick()
    assert all(len(b["rewards"]) == 2 for b in trainer._buffers)
    assert all(len(b["states"]) == len(b["actions"]) == len(b["values"])
               for b in trainer._buffers)
