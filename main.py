#!/usr/bin/env python3
"""Race Simulator command line.

    python main.py train                  # train with the live visualiser
    python main.py train --headless       # train with no window (servers, CI)
    python main.py replay                 # watch the saved policy drive
    python main.py evaluate --episodes 5  # score the saved policy, print a table

Run `python main.py <command> --help` for the full option list.
"""

import argparse
import os
import sys

from config import (
    NUM_GENERATIONS, POP_SIZE, TOTAL_LAPS, MAX_STEPS_GEN,
    STATE_DIM, ACTION_DIM, BEST_MODEL, LOG_DIR, SEED, CHECKPOINT_EVERY,
)
from track import LAYOUTS


def _common_args(parser):
    parser.add_argument("--model", default=BEST_MODEL,
                        help=f"checkpoint path (default: {BEST_MODEL})")
    parser.add_argument("--layout", default="circuit", choices=sorted(LAYOUTS),
                        help="track layout to drive")
    parser.add_argument("--laps", type=int, default=TOTAL_LAPS,
                        help="laps that count as a finished run")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="random seed for reproducible runs")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="race-simulator",
        description="Neuroevolution race simulator: LiDAR cars learn a circuit.")
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="evolve a population of drivers")
    _common_args(train)
    train.add_argument("--algo", choices=("ga", "ppo"), default="ga",
                       help="ga: neuroevolution (default). "
                            "ppo: gradient-based policy optimisation")
    train.add_argument("--generations", type=int, default=NUM_GENERATIONS)
    train.add_argument("--pop", type=int, default=POP_SIZE,
                       help="population size (cars per generation)")
    train.add_argument("--max-steps", type=int, default=MAX_STEPS_GEN,
                       help="physics steps before a generation is cut short")
    train.add_argument("--headless", action="store_true",
                       help="train with no window (much faster)")
    train.add_argument("--no-resume", action="store_true",
                       help="ignore any existing checkpoint and start fresh")
    train.add_argument("--slow", action="store_true",
                       help="start the visualiser at 1x instead of fast-forward")
    train.add_argument("--log-dir", default=LOG_DIR,
                       help="directory for per-generation CSV logs ('' to disable)")
    train.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                       help="generations between numbered snapshots (0 to disable)")
    train.add_argument("--quiet", action="store_true")

    rep = sub.add_parser("replay", help="watch the saved policy drive")
    _common_args(rep)

    ev = sub.add_parser("evaluate", help="score the saved policy headlessly")
    _common_args(ev)
    ev.add_argument("--episodes", type=int, default=1)
    ev.add_argument("--max-steps", type=int, default=MAX_STEPS_GEN)
    ev.add_argument("--stochastic", action="store_true",
                    help="sample actions from the policy instead of taking the "
                         "mean (the mean is deterministic, so every episode "
                         "would otherwise be identical)")

    return parser


# ── commands ─────────────────────────────────────────────────

def cmd_train(args):
    from ai import GeneticAgent, PPOAgent
    from simulation import Simulation, Trainer, PPOTrainer, set_seed
    from track import Track

    set_seed(args.seed)
    track = Track(args.layout)
    agent_cls = PPOAgent if args.algo == "ppo" else GeneticAgent
    trainer_cls = PPOTrainer if args.algo == "ppo" else Trainer

    agent = agent_cls(STATE_DIM, ACTION_DIM, pop_size=args.pop)
    sim = Simulation(track, agent, pop_size=args.pop,
                     total_laps=args.laps, max_steps=args.max_steps)
    trainer = trainer_cls(agent, sim, model_path=args.model,
                          log_dir=args.log_dir or None, quiet=args.quiet,
                          checkpoint_every=args.checkpoint_every)

    if not args.no_resume and os.path.exists(args.model):
        try:
            meta = trainer.load(args.model)
            print(f"resumed {args.model} "
                  f"(generation {meta.get('generation', '?')}, "
                  f"phase {meta.get('phase', 1)})")
        except (ValueError, RuntimeError, KeyError) as exc:
            print(f"starting fresh - could not resume {args.model}: {exc}")

    if args.headless:
        try:
            trainer.train(args.generations)
        except KeyboardInterrupt:
            print("\ninterrupted - saving current best")
            trainer.save()
    else:
        try:
            from renderer import train_rendered
        except ImportError as exc:
            print(f"raylib is unavailable ({exc}); falling back to --headless")
            trainer.train(args.generations)
        else:
            train_rendered(trainer, args.generations, fast=not args.slow)

    best = trainer.best_fitness_ever
    print(f"done - {trainer.generation} generations, "
          f"best fitness {best:,.0f}, model at {args.model}")
    return 0


def _load_policy(args, deterministic=True):
    from ai import PolicyAgent
    agent = PolicyAgent(STATE_DIM, ACTION_DIM, deterministic=deterministic)
    agent.load(args.model)
    return agent


def cmd_replay(args):
    from simulation import Simulation, set_seed
    from track import Track

    if not os.path.exists(args.model):
        print(f"no model at {args.model} - train one first "
              f"(python main.py train --headless)")
        return 1

    set_seed(args.seed)
    agent = _load_policy(args)
    sim = Simulation(Track(args.layout), agent, pop_size=1,
                     total_laps=args.laps, max_steps=MAX_STEPS_GEN)
    try:
        from renderer import replay
    except ImportError as exc:
        print(f"raylib is required for replay ({exc}); "
              f"use `python main.py evaluate` instead")
        return 1
    replay(sim)
    return 0


def cmd_evaluate(args):
    from simulation import Simulation, set_seed
    from track import Track

    if not os.path.exists(args.model):
        print(f"no model at {args.model} - train one first "
              f"(python main.py train --headless)")
        return 1

    set_seed(args.seed)
    agent = _load_policy(args, deterministic=not args.stochastic)
    sim = Simulation(Track(args.layout), agent, pop_size=1,
                     total_laps=args.laps, max_steps=args.max_steps)

    print(f"{'ep':>3} {'laps':>5} {'cps':>5} {'steps':>6} "
          f"{'hits':>5} {'best lap':>9} {'reward':>9}")
    laps, rewards = [], []
    for ep in range(args.episodes):
        sim.reset()
        sim.run()
        car = sim.cars[0]
        best = f"{car.best_time:.2f}s" if car.best_time < float("inf") else "--"
        print(f"{ep + 1:>3} {car.lap:>5} {car.total_cps:>5} {car.steps:>6} "
              f"{car.wall_hits:>5} {best:>9} {car.total_reward:>9.1f}")
        laps.append(car.lap)
        rewards.append(car.total_reward)

    print(f"\nmean laps {sum(laps) / len(laps):.2f} | "
          f"mean reward {sum(rewards) / len(rewards):.1f}")
    return 0


COMMANDS = {
    "train": cmd_train,
    "replay": cmd_replay,
    "evaluate": cmd_evaluate,
}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
