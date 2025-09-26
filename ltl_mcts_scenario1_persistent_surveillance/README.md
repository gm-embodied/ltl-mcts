# LTL-MCTS Scenario 1: Persistent Surveillance Task

## Overview

This project implements **Scenario 1: Persistent Surveillance Task** for the LTL-MCTS (Linear Temporal Logic Monte Carlo Tree Search) active perception framework. The agent must continuously monitor two surveillance zones and ensure that both regions are periodically visited to detect and track targets using a sensor-equipped mobile platform.

## LTL Formula

The task is specified by the Linear Temporal Logic formula:

```
G(F(ap_exist_A)) ∧ G(F(ap_exist_B))
```

Where:
- `G` (Globally): Always
- `F` (Finally): Eventually  
- `ap_exist_A`: At least one target exists in surveillance zone A
- `ap_exist_B`: At least one target exists in surveillance zone B

**Meaning**: The agent must always ensure that eventually it detects targets in both surveillance zones A and B.

## Key Features

- **LTL-Guided Planning**: Non-myopic MCTS planner with LTL-aware reward shaping
- **PMBM Filter**: Poisson Multi-Bernoulli Mixture filter for multi-target tracking under uncertainty
- **Semantic-Probabilistic Bridge**: Grounds LTL semantics on continuous belief states
- **Comprehensive Baselines**: Multiple comparison methods including information-theoretic and passive approaches
- **Windowed LTL**: Sliding window approach to handle temporal constraints

## Project Structure

```
ltl_mcts_scenario1_persistent_surveillance/
├── configs/
│   └── config.py                    # Configuration parameters
├── src/
│   ├── scenarios/
│   │   └── persistent_surveillance.py  # Main environment implementation
│   ├── filters/
│   │   └── proper_pmbm_filter.py    # PMBM filter implementation
│   ├── planners/
│   │   ├── ltl_mcts_planner.py      # LTL-MCTS planner (proposed method)
│   │   ├── ltl_myopic_planner.py    # LTL myopic baseline
│   │   ├── info_myopic_planner.py   # Information-theoretic baseline
│   │   ├── random_planner.py        # Random baseline
│   │   ├── passive_planner.py       # Stationary rotating sensor baseline
│   │   └── discretized_pomcp_planner.py  # Discretized POMCP baseline
│   └── utils/
│       ├── data_structures.py       # Core data structures
│       ├── ltl_window.py            # Windowed LTL implementation
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
   git clone <repository-url>
   cd ltl_mcts_scenario1_persistent_surveillance
   ```

## Usage

### Run Experiments

Execute all planners with multiple Monte Carlo runs:

```bash
python experiments/run_experiment.py
```

### Configuration

Key parameters can be modified in `configs/config.py`:

- **Environment**: 
  - Map size: 1000×1000 units
  - Simulation time: 50 steps
  - Agent speed: 40.0 units/step

- **Surveillance Zones**: 
  - `REGION_A = [200.0, 400.0, 600.0, 800.0]` (Upper-left)
  - `REGION_B = [600.0, 800.0, 200.0, 400.0]` (Lower-right)

- **Sensor Parameters**:
  - Range: 200.0 units
  - Field of view: 90 degrees
  - Detection probability: 0.9

- **LTL Parameters**: 
  - Window size: 40 steps
  - Probability thresholds: 0.6 (upper), 0.4 (lower)

- **MCTS Parameters**: 
  - Simulations: 80
  - Max depth: 8
  - Exploration constant: 1.4

### Expected Results

The experiment generates:

1. **Performance Metrics**:
   - **Success Rate**: Percentage of runs completing LTL task without violations
   - **Patrol Efficiency**: Movement efficiency metric (lower is better)
   - **Max Interval**: Maximum time between visits to each region
   - **GOSPA Error**: Multi-target tracking accuracy (Generalized OSPA)
   - **Planning Time**: Computational efficiency

2. **Visualizations**:
   - Success rate comparison with error bars
   - Planning time analysis
   - Performance radar charts
   - Completion time distributions

3. **Output Files**:
   - `scenario1_results.csv`: Detailed results for each run
   - `scenario1_summary.csv`: Aggregated statistics
   - `experiment.log`: Detailed execution log
   - Various `.png` visualization charts

## Planners Compared

1. **LTL_MCTS** (Proposed): Non-myopic MCTS with LTL-guided belief abstraction
2. **LTL_Myopic**: Greedy approach optimizing immediate LTL satisfaction
3. **Info_Myopic**: Information-theoretic approach with task-aware balancing
4. **Random**: Random action selection baseline
5. **Passive**: Stationary agent with rotating sensor (realistic passive baseline)
6. **Discretized_POMCP**: Traditional POMCP with discretized belief space

## Key Technical Components

### LTL-Guided Belief Abstraction
- Projects high-dimensional PMBM belief to task-relevant summary
- Enables tractable planning in continuous belief spaces
- Preserves LTL-relevant information while reducing dimensionality

### Semantic-Probabilistic Bridge
- Evaluates atomic propositions on PMBM belief states
- Uses MAP hypothesis approximation for computational efficiency
- Implements hysteresis thresholding to prevent oscillations

### PMBM Filter Integration
- Tracks multiple targets with unknown and time-varying cardinality
- Handles target birth, death, and data association uncertainty
- Provides probabilistic foundation for LTL constraint evaluation

### Windowed LTL Constraints
- Implements sliding window to make `G(F(...))` constraints tractable
- Prevents constraint satisfaction from becoming impossible
- Balances constraint satisfaction with computational feasibility

## Performance Expectations

Typical results (200 Monte Carlo runs):

| Method | Success Rate | Patrol Eff. | Max Interval (A/B) | GOSPA | Plan Time |
|--------|--------------|-------------|-------------------|------|-----------|
| **LTL_MCTS** | **86.5%** | 8.2 | 19.3/20.5 | 116.9 | 12.5ms |
| **LTL_Myopic** | 58.0% | 12.9 | 14.4/17.4 | 115.6 | 0.8ms |
| **Info_Myopic** | 32.5% | 16.8 | 8.8/9.7 | 113.9 | 1.5ms |
| **Passive** | 42.5% | 2.1 | 27.6/18.4 | 118.6 | 0.02ms |
| **Random** | 22.5% | 5.0 | 12.0/12.1 | 117.5 | 0.01ms |

**Key Insights**:
- LTL_MCTS achieves highest success rate while maintaining balanced region coverage
- Information-driven approaches struggle with LTL constraint satisfaction
- Passive baseline provides realistic lower bound for comparison

## Troubleshooting

1. **Import Errors**: Ensure all dependencies are installed and Python path includes project root
2. **Memory Issues**: Reduce `NUM_MONTE_CARLO_RUNS` in config.py for testing
3. **Performance Issues**: Adjust MCTS parameters (`MCTS_NUM_SIMULATIONS`, `MCTS_MAX_DEPTH`)
4. **Plotting Issues**: Ensure matplotlib backend supports display or save plots to file

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ltl_mcts_active_perception,
  title={LTL-Driven Active Perception for Multi-Target Tracking in Random Finite Set Domains},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  note={Code available at: [Repository URL]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@domain.com].