"""Central tuning knobs for the race simulator.

Every module reads its constants from here, so a single edit changes the
physics, the sensors, the learner and the renderer consistently.

Paths can be overridden with environment variables (see `.env.example`) so the
same code runs from a checkout, a scratch disk or CI without edits.
"""

import os as _os


def _load_dotenv(path=None):
    """Load KEY=VALUE lines from a local `.env` without clobbering real env vars.

    Keeps the project dependency-free — no python-dotenv needed.
    """
    path = path or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env")
    if not _os.path.isfile(path):
        return {}
    loaded = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'\"")
                if key and key not in _os.environ:
                    _os.environ[key] = value
                    loaded[key] = value
    except OSError:
        return {}
    return loaded


_load_dotenv()


def _env_int(name, default):
    raw = _os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Window
WIDTH = 1400
HEIGHT = 900
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 60, 60)
GREEN = (50, 255, 100)
BLUE = (80, 130, 255)
YELLOW = (255, 255, 60)
ORANGE = (255, 160, 40)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
DARK_BG = (15, 15, 25)
GRAY = (80, 80, 80)

# Wall section colors
WALL_COLORS = [YELLOW, GREEN, BLUE, RED]

# Car
CAR_WIDTH = 10
CAR_LENGTH = 20
MAX_SPEED = 20.0
ACCELERATION = 0.50
BRAKE_FORCE = 0.7
TURN_RATE = 3.5
FRICTION = 0.02
TRAIL_LENGTH = 30        # trail points kept for the exhaust effect

# Sensors
NUM_RAYS = 11            # denser LiDAR fan for better spatial awareness
MAX_RAY_LENGTH = 250     # ray range in pixels
RAY_SPREAD = 180         # full forward hemisphere; helps with tight corners

# Genetic Algorithm
POP_SIZE         = 40     # population per generation
ELITE_FRACTION   = 0.25   # top fraction survive unchanged
CROSSOVER_PROB   = 0.50   # probability a child is bred rather than cloned
MUTATION_RATE    = 0.15   # per-weight mutation probability
MUTATION_STD_P1  = 0.08   # noise std in phase 1 (explore)
MUTATION_STD_P2  = 0.05   # noise std in phase 2 (speed search)
NUM_GENERATIONS  = 500
MAX_STEPS_GEN    = 2500   # max steps before killing a generation
TOTAL_LAPS       = 3

# Plateau-breaking parameters
STAG_LIM         = 15     # gens without improvement before adaptive mutation kicks in
STAG_BUMP        = 0.08   # extra mutation std added when stagnating
ISLAND_STAG      = 50     # gens without improvement before island restart
ISLAND_FRAC      = 0.30   # fraction of bottom population to randomise on restart
TOURN_K          = 3      # tournament selection: candidates per draw

# Retirement rules — stop simulating cars that will never score again.
# The wall-hit rule counts contacts *since the last checkpoint*: a car that
# keeps making progress is still learning the route, while one that keeps
# bouncing off the same corner is not.
SECTOR_WALL_HITS   = 5    # wall contacts since the last checkpoint before retirement
IDLE_LIMIT_STEPS   = 150  # consecutive steps below IDLE_SPEED before retirement
STALL_LIMIT_STEPS  = 400  # steps without reaching a checkpoint before retirement

# PPO
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 64
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
MAX_STEPS = 2000
NUM_EPISODES = 3000
NUM_CARS = POP_SIZE

# Network
HIDDEN_SIZE = 128

# State extras
LOOKAHEAD_CPS = 2          # how many checkpoints ahead to encode in state (angles)

# State dimensionality (auto-computed so it stays in sync with car.py's get_state)
# NUM_RAYS + speed + angular_vel + LOOKAHEAD_CPS angles + min_ray proximity
STATE_DIM  = NUM_RAYS + 2 + LOOKAHEAD_CPS + 1
ACTION_DIM = 2   # [steering, throttle]

# Rewards
CHECKPOINT_REWARD = 3.0    # checkpoints are the primary navigation signal
LAP_REWARD = 15.0
CRASH_PENALTY = -15.0      # applied on every wall contact
SPEED_REWARD = 0.15        # per-step speed bonus
IDLE_PENALTY = -0.2        # per-step penalty for crawling
IDLE_SPEED = 0.5

# Phase-2 fitness shaping
SPLIT_REF_STEPS = 100.0            # sector time (in steps) worth zero split score
SPLIT_CONSISTENCY_WEIGHT = 0.5     # penalty per step of sector-time std deviation
WALL_HIT_FITNESS_PENALTY = 10.0    # fitness lost per wall contact

# Racing-line reward shaping
WALL_PROXIMITY_PENALTY = -0.05   # per step, scaled by how close the nearest wall is
WALL_PROXIMITY_BAND = 0.3        # normalised ray distance below which the penalty applies
LOOKAHEAD_ALIGN_REWARD = 0.05    # bonus when heading aligns with the next-next checkpoint
CP_SPEED_BONUS_SCALE = 0.08      # bonus = speed/MAX_SPEED * this, awarded at each checkpoint

# Rendering
FAST_STEPS = _env_int("RACE_SIM_FAST_STEPS", 500)   # physics ticks per frame in fast mode

# Paths (env-overridable)
MODEL_DIR  = _os.getenv("RACE_SIM_MODEL_DIR", "models")
LOG_DIR    = _os.getenv("RACE_SIM_LOG_DIR", "logs")
BEST_MODEL = _os.getenv("RACE_SIM_BEST_MODEL", _os.path.join(MODEL_DIR, "best.pt"))
CHECKPOINT_EVERY = _env_int("RACE_SIM_CHECKPOINT_EVERY", 50)  # gens between snapshots

# Reproducibility — leave unset for a random seed each run
SEED = _env_int("RACE_SIM_SEED", -1)
if SEED < 0:
    SEED = None

# Car colors for multi-car
CAR_COLORS = [ORANGE, CYAN, MAGENTA, GREEN, RED]
