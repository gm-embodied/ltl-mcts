# LTL-MCTS Scenario 2: Patrol and Respond Task

## Overview

This project implements **Scenario 2: Patrol and Respond Task** for the LTL-MCTS (Linear Temporal Logic Monte Carlo Tree Search) active perception framework. The agent must patrol a designated region and respond to targets when they appear, while resisting temptation from distraction targets in other areas.

## LTL Formula

The task is specified by the Linear Temporal Logic formula:

```
G(ap_exist(R_A) ⟹ F(ap_loc(any, R_A)))
```

Where:
- `G` (Globally): Always
- `F` (Finally): Eventually
- `⟹` (Implies): Logical implication
- `ap_exist(R_A)`: At least one target exists in patrol region A
- `ap_loc(any, R_A)`: Any tracked target's position estimate is localized in region A

**Meaning**: Whenever the agent believes a target exists in patrol region A, it must eventually go there to localize/confirm the target's position.

## Key Features

- **Scripted Events**: Deterministic target spawning for controlled evaluation
- **Temptation Resistance**: Distraction targets test the agent's focus on patrol duty
- **PMBM Filter**: Multi-target tracking with existence and localization probabilities
- **Reward Shaping**: Specialized reward function for patrol and respond behavior
- **Comprehensive Metrics**: Duty completion rates, temptation resistance, success rates

## Project Structure

```
ltl_mcts_scenario2_patrol_respond/
├── configs/
│   └── config.py                    # Configuration parameters
├── src/
│   ├── scenarios/
│   │   └── patrol_respond.py        # Main environment implementation
│   ├── filters/
│   │   └── proper_pmbm_filter.py    # PMBM filter implementation
│   ├── planners/
│   │   ├── ltl_mcts_planner.py      # LTL-MCTS planner
│   │   ├── ltl_myopic_planner.py    # LTL myopic baseline
│   │   ├── info_myopic_planner.py   # Information-theoretic baseline
│   │   ├── random_planner.py        # Random baseline
│   │   └── passive_planner.py       # Fixed patrol pattern baseline
│   └── utils/
│       ├── data_structures.py       # Core data structures
│       └── evaluation_metrics.py    # OSPA and other metrics
├── experiments/
│   └── run_experiment.py            # Main experiment runner
└── results/                         # Experimental results (generated)
```

## Installation

1. **Prerequisites**:
   ```bash
   pip install numpy scipy matplotlib pandas seaborn
   ```

2. **Clone and Setup**:
   ```bash
   cd /path/to/ltl_mcts_scenario2_patrol_respond
   ```

## Usage

### Run Experiments

Execute all planners with multiple Monte Carlo runs:

```bash
python experiments/run_experiment.py
```

### Configuration

Key parameters can be modified in `configs/config.py`:

- **Environment**: Map size, simulation time, sensor parameters
- **Regions**: 
  - `SCENARIO_2_REGION_A = [0.0, 300.0, 0.0, 1000.0]` (Patrol region - left side)
  - `SCENARIO_2_REGION_B = [700.0, 1000.0, 400.0, 600.0]` (Distraction region - right side)
- **Scripted Events**: Target spawn times and positions
- **LTL Parameters**: Existence/localization thresholds, grace period
- **MCTS Parameters**: Optimized for patrol and respond behavior

### Scripted Events

The scenario includes deterministic events for controlled evaluation:

1. **Step 5**: Target spawns in patrol region A at `[280.0, 500.0]`
2. **Step 8**: Distraction target spawns in region B at `[720.0, 500.0]`

These events create a controlled conflict where the agent must:
- Detect the target in region A (triggers patrol duty)
- Resist the temptation of the distraction target in region B
- Successfully localize the target in region A within the grace period

### Expected Results

The experiment will generate:

1. **Performance Metrics**:
   - Success rate (percentage of runs without LTL violations)
   - Duty completion rate (successful responses to patrol duties)
   - Duty activations (number of times patrol duty was triggered)
   - Temptation resistance (avoiding distraction targets)
   - OSPA tracking error

2. **Visualizations**:
   - Success rate comparison charts
   - Duty completion analysis
   - Planning time comparison
   - Performance radar charts

3. **Output Files**:
   - `scenario2_results.csv`: Detailed results for each run
   - `scenario2_summary.csv`: Aggregated statistics
   - `experiment.log`: Detailed execution log
   - Various `.png` charts

## Planners Compared

1. **LTL_MCTS**: Proposed method with patrol-specific reward shaping
2. **LTL_Myopic**: Greedy approach optimizing immediate LTL satisfaction
3. **Info_Myopic**: Information-theoretic approach maximizing belief entropy reduction
4. **Random**: Random action selection baseline
5. **Passive**: Fixed patrol pattern baseline

## Key Technical Components

### Patrol Duty State Machine
- **Inactive**: No patrol duty required
- **Active**: Target detected in region A, must respond within grace period
- **Completed**: Successfully localized target, duty fulfilled

### Atomic Propositions
- `ap_exist_A`: Probability ≥ threshold that target exists in patrol region A
- `ap_loc_any_A`: Probability ≥ threshold that any tracked target is localized in region A

### Reward Function
Specialized rewards for patrol and respond behavior:
- `R_PATROL_SUCCESS`: Large reward for completing patrol duty
- `R_DUTY_VIOLATION`: Large penalty for violating LTL constraint
- `R_TEMPTATION_RESIST`: Small reward for avoiding distraction region
- `R_APPROACH_A`: Reward for approaching patrol region when duty is active
- `R_LOCALIZATION_PROGRESS`: Reward for increasing localization probability

### LTL Constraint Handling
- Grace period mechanism prevents immediate violations
- Hysteresis thresholds avoid oscillations
- Clear success/failure criteria for evaluation

## Performance Expectations

Typical results show:
- **LTL_MCTS**: Highest success rate (~85-95%), efficient duty completion
- **LTL_Myopic**: Reasonable performance but may get distracted
- **Info_Myopic**: Good tracking but poor LTL constraint satisfaction
- **Random/Passive**: Very poor performance, useful as lower bounds

## Key Parameters

### Critical Tuning Parameters
- `LTL_EXIST_THRESHOLD = 0.4`: Trigger threshold for patrol duty
- `LTL_LOC_THRESHOLD = 0.10`: Completion threshold for localization
- `LTL_FINALLY_GRACE_PERIOD = 35`: Time limit for completing patrol duty

### MCTS Parameters (Optimized)
- `MCTS_NUM_SIMULATIONS = 600`: Increased for better performance
- `MCTS_MAX_DEPTH = 8`: Balanced depth for efficiency
- `MCTS_EXPLORATION_CONSTANT = 1.4`: Balanced exploration

## Troubleshooting

1. **Low Success Rates**: Check LTL thresholds and grace period
2. **Import Errors**: Ensure all dependencies are installed
3. **Memory Issues**: Reduce `NUM_MONTE_CARLO_RUNS` for testing
4. **Performance Issues**: Adjust MCTS parameters

## Expected Behaviors

- **LTL_MCTS**: Should efficiently patrol region A and respond to targets while resisting temptation from region B
- **LTL_Myopic**: May get distracted by immediate rewards and fail to maintain patrol duty
- **Info_Myopic**: Focuses on information gain but may not respect LTL constraints
- **Random**: Random behavior, likely to fail patrol duty
- **Passive**: Fixed patrol pattern, may miss dynamic target responses

## Citation

If you use this code, please cite:

```bibtex
@article{ltl_mcts_active_perception,
  title={LTL-Driven Active Perception for Multi-Target Tracking},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
