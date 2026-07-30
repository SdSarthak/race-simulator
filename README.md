# Race Simulator

A 2D racing simulator where cars learn to drive a circuit from scratch. Each car
sees the world through an 11-beam LiDAR fan, feeds that into a small neural
network, and gets steering and throttle back. Nobody writes the driving logic —
a genetic algorithm evolves a population of 40 networks until they can complete
laps, then switches objective and optimises for lap time.

There is no dataset and no pretrained model to download: the environment
generates all of its own experience, so a fresh clone can start training in
seconds.

```
python main.py train --headless      # evolve drivers, no window
python main.py train                 # same, with the live visualiser
python main.py replay                # watch the best saved policy drive
python main.py evaluate              # score the saved policy, print a table
```

## How it works

**The track** (`track.py`) is two closed polygons — an outer and an inner
boundary. The segments between them are walls; the lines joining outer point *i*
to inner point *i* are the checkpoints a car must cross in order. Two layouts
ship with the project: `circuit` (a hand-laid course with a chicane) and `oval`
(procedurally generated). Ray casting and collision are vectorised with numpy:
one array operation tests a segment against every wall at once.

**The car** (`car.py`) is a rectangle with simple bicycle-ish physics —
speed-dependent steering, throttle/brake, friction. It casts `NUM_RAYS` rays
across a 180 degree forward arc every step. Hitting a wall is not instantly
fatal: the car is teleported back to its last checkpoint, takes a reward
penalty, and its wall-hit counter goes up. Cars are retired when they hit the
walls repeatedly *without reaching the next checkpoint*, crawl for too long, or
fail to reach a checkpoint in time — a car that keeps making progress is still
learning the route, so only genuinely stuck cars are removed.

**The state** the network sees (16 floats by default):

| slice | meaning |
| --- | --- |
| `0 .. NUM_RAYS-1` | normalised LiDAR distances (0 = wall on the nose, 1 = clear) |
| `NUM_RAYS` | speed / `MAX_SPEED` |
| `NUM_RAYS + 1` | angular velocity / `TURN_RATE` |
| next `LOOKAHEAD_CPS` | signed heading error to the next checkpoints, in half-turns |
| last | distance to the nearest wall (min ray) |

**The learner** (`ai.py`) is neuroevolution, not backprop. Every generation the
population drives, each car is scored, and the next population is built from the
elite plus tournament-selected parents with uniform crossover and gaussian
mutation. Two anti-plateau mechanisms are built in: the mutation std is bumped
after `STAG_LIM` generations without improvement, and after `ISLAND_STAG` the
bottom `ISLAND_FRAC` of the population is replaced with fresh random networks.

A gradient-based trainer ships alongside it: `--algo ppo` runs proximal policy
optimisation over the same network, treating every car in the field as a
parallel environment for one shared policy. It uses the critic head and the
per-step reward signal that the genetic trainer ignores (checkpoints, laps,
crashes, wall proximity, racing-line alignment). The genetic trainer is the
default and the one the reward weights were tuned around; PPO is there to
compare against.

Fitness runs in two phases. Phase 1 only cares about getting round —
`laps * 10000 + checkpoints * 100 + speed`. The moment any car completes
`TOTAL_LAPS`, the trainer flips to phase 2, which balances lap completion, best
lap time, speed carried through checkpoints, sector-split consistency and a
penalty per wall contact, each term scaled to roughly 0-1000 so no single signal
dominates.

## Install

```bash
git clone https://github.com/SdSarthak/race-simulator.git
cd race-simulator
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`raylib` is only needed for the window. On a headless box install just
`numpy` and `torch` and pass `--headless`; the CLI falls back to headless
automatically if raylib cannot be imported.

For the test suite:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Usage

### Train

```bash
python main.py train --headless --generations 300
python main.py train --pop 60 --laps 3 --layout oval --seed 42
python main.py train --slow                  # visualiser at 1x instead of fast-forward
python main.py train --headless --algo ppo   # gradient-based trainer instead
```

Training resumes from `models/best.pt` when it exists (`--no-resume` to start
clean). The checkpoint stores the generation and phase alongside the weights, so
a resumed run picks up in phase 2 rather than relearning the route. Numbered
snapshots land in `models/gen_<n>.pt` every `--checkpoint-every` generations and
a CSV of per-generation stats is written to `logs/`.

Keys in the visualiser:

| key | action |
| --- | --- |
| `F` | toggle fast-forward (500 physics ticks per frame) |
| `V` | toggle drawing entirely — training keeps running |
| `R` | skip the rest of this generation |
| `ESC` | save and quit |

### Replay and evaluate

```bash
python main.py replay --model models/gen_200.pt
python main.py evaluate --episodes 5 --stochastic
```

`evaluate` needs no display and prints laps, checkpoints, wall hits, best lap
time and total reward per episode. The learned policy is deterministic, so
repeated episodes are identical unless `--stochastic` is passed to sample from
the policy distribution instead of taking its mean.

## Configuration

`config.py` holds every tunable: physics, sensor geometry, GA hyperparameters,
reward weights and retirement rules. `STATE_DIM` is derived from `NUM_RAYS` and
`LOOKAHEAD_CPS`, so changing the sensor layout keeps the network in sync — but
it also invalidates old checkpoints, and loading one raises a clear error rather
than silently mismatching.

Paths and a few runtime knobs can be set from the environment (or a local `.env`
file, which `config.py` loads itself — no python-dotenv needed). Copy
`.env.example` to `.env` to start. There are no secrets or API keys anywhere in
this project.

## Layout

| file | role |
| --- | --- |
| `main.py` | CLI: `train`, `replay`, `evaluate` |
| `simulation.py` | headless population runner + `Trainer` (evolution, logging, checkpoints) |
| `car.py` | physics, LiDAR, checkpoints, reward and telemetry |
| `track.py` | layouts, vectorised ray casting and collision |
| `ai.py` | `PolicyNet`, `GeneticAgent`, `PPOAgent`, `PolicyAgent`, checkpoint I/O |
| `renderer.py` | raylib visualiser — draws the very objects the trainer scores |
| `config.py` | all tunables, env overrides, `.env` loading |
| `tests/` | deterministic tests, no downloads, no display |

Nothing in `simulation.py`, `car.py`, `track.py` or `ai.py` imports a rendering
library, which is why the tests and headless training run anywhere.

## Notes on training

Expect phase 1 to take a while: the first generations are cars driving into
walls. Checkpoint counts creeping up generation over generation is the signal to
watch — `BestCP` in the log line. It is normal for progress to sit flat for a
stretch before the adaptive mutation bump or the island restart shakes it loose.

Model weights are deliberately not committed to this repository. Train your own
with a few hundred generations of `python main.py train --headless`.
