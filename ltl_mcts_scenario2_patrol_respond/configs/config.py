"""
Scenario 2: Patrol and Respond Task Configuration
Goal: Agent must continuously monitor patrol region A, if it believes there is a target in that region,
it must eventually go confirm the target's location. Must resist the temptation to go to region B chasing new targets.
LTL Formula: phi_2 = G(ap_exist(R_A) ⟹ F(ap_loc(any, R_A)))
"""

import numpy as np
import logging

# ========== 1. Environment Settings ==========
MAP_WIDTH = 1000.0
MAP_HEIGHT = 1000.0
SIMULATION_TIME_STEPS = 100
TIME_STEP_DURATION = 1.0
LOG_LEVEL = "INFO"

# ========== 2. Agent Configuration ==========
AGENT_INITIAL_POSITION = [400.0, 500.0]  # Near patrol region A
AGENT_SPEED = 45.0
AGENT_CONTROL_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# ========== 3. Sensor Parameters ==========
SENSOR_RANGE = 200.0
SENSOR_FOV_DEGREES = 120.0
SENSOR_PROB_DETECTION = 0.9
SENSOR_MEASUREMENT_NOISE_STD = 10.0
SENSOR_MEASUREMENT_NOISE_COV = np.diag([SENSOR_MEASUREMENT_NOISE_STD**2, 
                                        SENSOR_MEASUREMENT_NOISE_STD**2])
SENSOR_CLUTTER_RATE = 0.5

# ========== 4. Target Dynamics ==========
TARGET_SURVIVAL_PROBABILITY = 0.995
TARGET_PROCESS_NOISE_STD = 2.0
TARGET_STATE_DIM = 4

# Target state transition matrix
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

# Observation matrix
OBSERVATION_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

# ========== 5. PMBM Filter Parameters ==========
PMBM_DETECTION_PROB = 0.9
PMBM_SURVIVAL_PROB = 0.995
PMBM_CLUTTER_RATE = 0.5
PMBM_PRUNE_THRESHOLD = 0.3
PMBM_MAX_COMPONENTS = 100
PMBM_GATING_THRESHOLD = 50.0

# ========== 6. Scenario 2 Region Definitions ==========
SCENARIO_2_REGION_A = [0.0, 300.0, 0.0, 1000.0]    # Patrol region R_A: left side of map
SCENARIO_2_REGION_B = [700.0, 1000.0, 400.0, 600.0]  # Distraction region R_B: right side of map

# ========== 7. Scenario 2 Scripted Events ==========
SCENARIO_2_TARGET_A_SPAWN_TIME = 5   # A region target spawn time
SCENARIO_2_TARGET_B_SPAWN_TIME = 8   # B region distraction target spawn time (enhance conflict)
SCENARIO_2_TARGET_A_SPAWN_POS = [280.0, 500.0]  # A region target position
SCENARIO_2_TARGET_B_SPAWN_POS = [720.0, 500.0]  # B region target position (adjusted for detectability)
SCENARIO_2_TARGET_A_VELOCITY = [2.0, -3.0]   # A region target velocity  
SCENARIO_2_TARGET_B_VELOCITY = [-2.0, -3.0]  # B region target velocity  

# Additional target spawning for more patrol duties  
SCENARIO_2_ADDITIONAL_SPAWNS = [
    {'time': 25, 'region': 'A', 'pos': [250.0, 300.0], 'vel': [1.5, -2.0]},    
    {'time': 45, 'region': 'A', 'pos': [280.0, 700.0], 'vel': [-1.0, -2.5]},   
    {'time': 65, 'region': 'A', 'pos': [200.0, 600.0], 'vel': [2.0, -1.5]},   
    {'time': 35, 'region': 'B', 'pos': [750.0, 450.0], 'vel': [-2.0, -2.0]},   
    {'time': 55, 'region': 'B', 'pos': [800.0, 550.0], 'vel': [-1.5, -2.5]},   
]

# ========== 8. LTL Task Parameters (Key Tuning Parameters) ==========
LTL_EXIST_THRESHOLD = 0.6    # ap_exist(R_A) trigger threshold [higher threshold to avoid false activation]
LTL_LOC_THRESHOLD = 0.7      # ap_loc(any, R_A) completion threshold [higher threshold for more challenging task]
LTL_FINALLY_GRACE_PERIOD = 30  # F operator grace period [longer time limit for fairness]

# Hysteresis thresholds
AP_PROBABILITY_UPPER_THRESHOLD = 0.5
AP_PROBABILITY_LOWER_THRESHOLD = 0.3

# Atomic proposition definitions
LTL_FORMULA = "G(ap_exist_A => F(ap_loc_any_A))"
ATOMIC_PROPOSITIONS = {
    'ap_exist_A': {
        'description': 'At least one target exists in patrol region R_A',
        'region': SCENARIO_2_REGION_A,
        'threshold': LTL_EXIST_THRESHOLD
    },
    'ap_loc_any_A': {
        'description': 'Any tracked target position estimate is in patrol region R_A',
        'region': SCENARIO_2_REGION_A,
        'threshold': LTL_LOC_THRESHOLD
    }
}

# ========== 9. MCTS Planner Parameters (Optimized for High Difficulty) ==========
MCTS_NUM_SIMULATIONS = 50       # Reduced simulation count for faster performance
MCTS_MAX_DEPTH = 8              # Shorter depth for more focused planning
MCTS_EXPLORATION_CONSTANT = 1.4  # Balanced exploration constant
MCTS_DISCOUNT_FACTOR = 0.95


# ========== 10. Scenario 2 Reward Parameters ==========
R_PATROL_SUCCESS = 200.0         # Reward for successfully completing patrol duty (increased)
R_DUTY_VIOLATION = -300.0        # Penalty for violating patrol duty (increased)
R_TEMPTATION_RESIST = 8.0        # Reward for resisting temptation to go to region B (increased)
R_APPROACH_A = 15.0              # Reward for approaching region A when duty is active (increased)
R_LOCALIZATION_PROGRESS = 30.0   # Reward for increasing localization probability (increased)
R_STEP = -0.5                    # Small step penalty to encourage efficiency (reduced)

# ========== 11. Poisson Point Process ==========
PPP_DENSITY = 1e-9      # Extremely low PPP density to avoid false activations

# ========== 12. Evaluation Parameters ==========
OSPA_CUTOFF = 120.0
OSPA_ORDER = 2
OSPA_EXISTENCE_THRESHOLD = 0.75
OSPA_MAX_TARGETS_RATIO = 1.3
GOSPA_ALPHA = 2.0

# ========== 13. Experiment Parameters ==========
NUM_MONTE_CARLO_RUNS = 500
RANDOM_SEED_BASE = 42

# Baseline methods list
BASELINE_METHODS = [
    "LTL_MCTS",
    "LTL_Myopic", 
    "Info_Myopic",
    "Random",
    "Passive"
]

# ========== 14. Visualization Parameters ==========
REGION_A_COLOR = 'lightblue'
REGION_A_ALPHA = 0.3
REGION_A_LABEL = 'Patrol Region A'

REGION_B_COLOR = 'lightcoral'
REGION_B_ALPHA = 0.3
REGION_B_LABEL = 'Distraction Region B'

TRAJECTORY_COLORS = {
    'LTL_MCTS': 'blue',
    'LTL_Myopic': 'red',
    'Info_Myopic': 'green',
    'Random': 'orange',
    'Passive': 'gray'
}

# Ensure all chart text uses English fonts to avoid font warnings
import matplotlib
import matplotlib.pyplot as plt
# Use system default fonts to avoid Arial not found warnings
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# ========== 15. Debug Parameters ==========
DEBUG_BRIDGE = False
DEBUG_MCTS = False

# ========== 16. Other Necessary Parameters ==========
ENABLE_VISUALIZATION = True
SAVE_TRAJECTORY_DATA = True
RESULTS_DIR = "results"

# Grid size for discretized planners
GRID_SIZE = 20

# LTL confirmation threshold for myopic planner
LTL_CONFIRM_THRESHOLD_TAU = 50.0

# Passive planner waypoints for scenario 2
PASSIVE_WAYPOINTS = []
y_positions = np.arange(200, MAP_HEIGHT - 50, 220)
for i, y in enumerate(y_positions):
    if i % 2 == 0:
        x_positions = np.arange(200, MAP_WIDTH - 50, 220)
    else:
        x_positions = np.arange(MAP_WIDTH - 200, 50, -220)
    for x in x_positions:
        PASSIVE_WAYPOINTS.append([float(x), float(y)])

PASSIVE_WAYPOINT_THRESHOLD = 30.0

# ========== 17. Expected Behaviors for Logging ==========
EXPECTED_BEHAVIORS = {
    'LTL_MCTS': 'Should efficiently patrol region A and respond to targets while resisting temptation from region B',
    'LTL_Myopic': 'May get distracted by immediate rewards and fail to maintain patrol duty',
    'Info_Myopic': 'Focuses on information gain but may not respect LTL constraints',
    'Random': 'Random behavior, likely to fail patrol duty',
    'Passive': 'Fixed patrol pattern, may miss dynamic target responses'
}
