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

# Sensors
NUM_RAYS = 11            # increased from 7 — denser LiDAR fan for better spatial awareness
MAX_RAY_LENGTH = 250     # slightly longer range so the car can see further ahead
RAY_SPREAD = 180         # full forward hemisphere (was 140°); helps with tight corners

# Genetic Algorithm
POP_SIZE         = 40     # doubled from 20 — more diversity, breaks plateaus
ELITE_FRACTION   = 0.25   # top fraction survive unchanged
CROSSOVER_PROB   = 0.50   # higher crossover probability for diversity
MUTATION_RATE    = 0.15   # per-weight mutation probability
MUTATION_STD_P1  = 0.08   # noise std in phase 1 (explore)
MUTATION_STD_P2  = 0.05   # noise std in phase 2 (slightly larger base for speed search)
NUM_GENERATIONS  = 500
MAX_STEPS_GEN    = 2500   # max steps before killing a generation
TOTAL_LAPS       = 3

# Plateau-breaking parameters
STAG_LIM         = 15     # gens without improvement before adaptive mutation kicks in
STAG_BUMP        = 0.08   # extra mutation std added when stagnating
ISLAND_STAG      = 50     # gens without improvement before island restart
ISLAND_FRAC      = 0.30   # fraction of bottom population to randomise on restart
TOURN_K          = 3      # tournament selection: candidates per draw

# PPO (kept for backward compat / single-car replay)
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_EPSILON = 0.2
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 64
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
CHECKPOINT_REWARD = 3.0    # increased — checkpoints are the primary navigation signal
LAP_REWARD = 15.0          # increased lap reward
CRASH_PENALTY = -15.0      # stronger crash deterrent
SPEED_REWARD = 0.15        # per-step speed bonus (encourages not braking unnecessarily)
IDLE_PENALTY = -0.2        # stronger idle penalty (was -0.1)
IDLE_SPEED = 0.5

# Racing-line reward shaping (new)
WALL_PROXIMITY_PENALTY = -0.05   # per step, scaled by how close the nearest wall is
LOOKAHEAD_ALIGN_REWARD = 0.05    # bonus when heading aligns with the next-next checkpoint
CP_SPEED_BONUS_SCALE = 0.08      # bonus = speed/MAX_SPEED * this, awarded at each checkpoint

# Paths
MODEL_DIR = "models"
BEST_MODEL = "models/best.pt"

# Car colors for multi-car
CAR_COLORS = [ORANGE, CYAN, MAGENTA, GREEN, RED]
