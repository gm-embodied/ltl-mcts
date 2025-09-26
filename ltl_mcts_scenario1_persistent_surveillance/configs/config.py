#!/usr/bin/env python3
"""
Scenario 1 Configuration - Persistent Surveillance Task
LTL Formula: G(F(ap_exist_A)) & G(F(ap_exist_B))
"""

import numpy as np
import logging

# ========== 1. Environment Settings ==========
MAP_WIDTH = 1000.0
MAP_HEIGHT = 1000.0
SIMULATION_TIME_STEPS = 50
TIME_STEP_DURATION = 1.0
LOG_LEVEL = "INFO"

# ========== 2. Agent Configuration ==========
AGENT_INITIAL_POSITION = [500.0, 500.0]
AGENT_SPEED = 40.0
AGENT_CONTROL_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# ========== 3. Sensor Parameters ==========
SENSOR_RANGE = 200.0        # Sensor detection range
SENSOR_FOV_DEGREES = 90.0    # 90-degree field of view, balancing detection capability and challenge
SENSOR_PROB_DETECTION = 0.9
SENSOR_MEASUREMENT_NOISE_STD = 10.0
SENSOR_MEASUREMENT_NOISE_COV = np.diag([SENSOR_MEASUREMENT_NOISE_STD**2, 
                                        SENSOR_MEASUREMENT_NOISE_STD**2])
SENSOR_CLUTTER_RATE = 1.0  # Clutter rate for false alarms

# ========== 4. Target Dynamics ==========
TARGET_PROCESS_NOISE_STD = 1.5  # Process noise standard deviation
TARGET_SURVIVAL_PROBABILITY = 0.99  # Target survival probability
TARGET_INITIAL_COUNT = 2  # Initial number of targets
TARGET_BIRTH_RATE = 0.0

# Regional target spawning - baseline target density
REGIONAL_BIRTHS_PER_STEP_A = 0.15  # baseline target birth rate
REGIONAL_BIRTHS_PER_STEP_B = 0.15  # baseline target birth rate
BACKGROUND_BIRTHS_PER_STEP = 0.005  # baseline background clutter

# Target state transition model
dt = TIME_STEP_DURATION
TARGET_TRANSITION_MATRIX = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Process noise covariance matrix
q = TARGET_PROCESS_NOISE_STD**2
TARGET_PROCESS_NOISE_COV = q * np.array([
    [dt**4/4, 0,       dt**3/2, 0],
    [0,       dt**4/4, 0,       dt**3/2],
    [dt**3/2, 0,       dt**2,   0],
    [0,       dt**3/2, 0,       dt**2]
])

TARGET_STATE_DIM = 4
OBSERVATION_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

INITIAL_TARGET_PLACEMENT = "regions"

# ========== 5. PMBM Filter Parameters ==========
PMBM_DETECTION_PROB = 0.9
PMBM_SURVIVAL_PROB = 0.995     # target survival probability
PMBM_CLUTTER_RATE = 1.0
PMBM_PRUNE_THRESHOLD = 0.5     # increased pruning threshold to reduce clutter components
PMBM_MAX_COMPONENTS = 50       # reduced max components for efficiency
PMBM_GATING_THRESHOLD = 30.0   # tightened gating to reduce false associations

# ========== 6. Region Definitions ==========
# baseline region setup: reasonable distance, ensuring reachability
REGION_A = [200.0, 400.0, 600.0, 800.0]    # upper-left region
REGION_B = [600.0, 800.0, 200.0, 400.0]    # lower-right region

# ========== 7. LTL Task Parameters ==========
LTL_WINDOW_STEPS = 40          # baseline window setting, approximately 1.5 round trips
LTL_FORMULA = "G(F(ap_exist_A)) & G(F(ap_exist_B))"

# Atomic proposition thresholds - unified threshold setting
AP_EXIST_PROB_ON = 0.6         # existence probability threshold (unified usage)
AP_EXIST_PROB_OFF = 0.4        # exit threshold (hysteresis effect)

# Hysteresis thresholds (consistent with above)
AP_PROBABILITY_UPPER_THRESHOLD = 0.6
AP_PROBABILITY_LOWER_THRESHOLD = 0.4

# ========== 8. MCTS Planner Parameters ==========
MCTS_NUM_SIMULATIONS = 500    # simulation count for planning
MCTS_MAX_DEPTH = 6          # planning depth
MCTS_EXPLORATION_CONSTANT = 1.4  # UCB1 exploration parameter
MCTS_DISCOUNT_FACTOR = 0.95  # discount factor

# ========== 9. Poisson Point Process ==========
PPP_DENSITY = 1e-5              # baseline PPP density setting

# ========== 10. Evaluation Parameters ==========
# GOSPA (Generalized OSPA) Parameters - Better for high-clutter environments
OSPA_CUTOFF = 120.0  # cutoff distance for OSPA metric
OSPA_ORDER = 2       # L2 norm order
GOSPA_ALPHA = 2.0    # cardinality penalty parameter (alpha >= p)
OSPA_EXISTENCE_THRESHOLD = 0.75  # existence threshold for filtering
OSPA_MAX_TARGETS_RATIO = 1.3     # maximum targets ratio for estimation

# ========== 11. Experiment Parameters ==========
NUM_MONTE_CARLO_RUNS = 500   # number of Monte Carlo runs for statistical analysis
RANDOM_SEED_BASE = 42

# Baseline methods list
BASELINE_METHODS = [
    "LTL_MCTS",
    "LTL_Myopic", 
    "Info_Myopic",
    "Random",
    "Passive",
    "Discretized_POMCP"
]

# ========== 12. Visualization Parameters ==========
REGION_A_COLOR = 'lightgreen'
REGION_A_ALPHA = 0.3
REGION_A_LABEL = 'Surveillance Zone A'

REGION_B_COLOR = 'lightcoral'
REGION_B_ALPHA = 0.3
REGION_B_LABEL = 'Surveillance Zone B'

TRAJECTORY_COLORS = {
    'LTL_MCTS': 'blue',
    'LTL_Myopic': 'red',
    'Info_Myopic': 'green',
    'Random': 'orange',
    'Passive': 'gray',
    'Discretized_POMCP': 'purple'
}

# Ensure all chart text uses English fonts to avoid font warnings
import matplotlib
import matplotlib.pyplot as plt
# Use system default fonts to avoid Arial not found warnings
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# ========== 13. Debug Parameters ==========
DEBUG_BRIDGE = False
DEBUG_MCTS = False

# ========== 14. Passive Planner Configuration ==========
# passive baseline design: stationary rotating sensor (more reasonable passive behavior)
# agent stays at fixed position, only rotates sensor for scanning

# passive baseline type selection
PASSIVE_TYPE = "stationary_rotating"  # options: "waypoint_patrol", "stationary_rotating"

# fixed position passive baseline configuration
PASSIVE_STATIONARY_POSITION = [500.0, 500.0]  # map center position
PASSIVE_ROTATION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]  # 8-direction scanning
PASSIVE_ROTATION_STEPS_PER_ANGLE = 5  # steps per angle

# alternative: waypoint patrol passive baseline (for comparison if needed)
PASSIVE_WAYPOINTS = [
    [350.0, 650.0],  # near region A but offset from center
    [500.0, 600.0],  # middle position upper
    [650.0, 350.0],  # near region B but offset from center
    [500.0, 400.0],  # middle position lower
]
PASSIVE_WAYPOINT_THRESHOLD = 50.0

# ========== 15. Other Necessary Parameters ==========
ENABLE_VISUALIZATION = True
SAVE_TRAJECTORY_DATA = True
RESULTS_DIR = "results"

# Grid size for discretized planners
GRID_SIZE = 20

# LTL confirmation threshold for myopic planner
LTL_CONFIRM_THRESHOLD_TAU = 50.0  # further reduced confirmation threshold to make targets easier to confirm
