import time
"""
Discretized LTL-POMCP planner
Coarsens continuous belief to 20x20 grid and uses simple heuristics to simulate POMCP selection

"""

import numpy as np
from typing import Any, Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.planners.base_planner import BasePlanner
from configs.config import MAP_WIDTH, MAP_HEIGHT, AGENT_CONTROL_ANGLES, GRID_SIZE, REGION_A, REGION_B

class DiscretizedPOMCPPlanner(BasePlanner):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Discretized_POMCP", config)

    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        start_time = time.time()
        # map belief to grid: count sum of estimated target existence probabilities in each cell
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

        map_hyp = pmbm_belief.get_map_hypothesis()
        if map_hyp is not None:
            for comp in map_hyp.bernoulli_components:
                pos = comp.get_position_mean()
                i = int(np.clip(pos[0] / (MAP_WIDTH / GRID_SIZE), 0, GRID_SIZE - 1))
                j = int(np.clip(pos[1] / (MAP_HEIGHT / GRID_SIZE), 0, GRID_SIZE - 1))
                grid[j, i] += comp.existence_prob

        # Modify strategy: consider LTL task requirements, balance planning between two surveillance regions
        agent_pos = world_state.agent_state.position

        # Compute probability mass of two surveillance regions
        region_A_mass = self._compute_region_mass(grid, REGION_A)
        region_B_mass = self._compute_region_mass(grid, REGION_B)
        
        # Region center
        region_A_center = np.array([(REGION_A[0] + REGION_A[1])/2, (REGION_A[2] + REGION_A[3])/2])
        region_B_center = np.array([(REGION_B[0] + REGION_B[1])/2, (REGION_B[2] + REGION_B[3])/2])
        
        # Decision logic: prioritize regions with lower probability mass
        if region_A_mass < 0.3 and region_B_mass >= 0.3:
            # Region A needs attention
            target = region_A_center
        elif region_B_mass < 0.3 and region_A_mass >= 0.3:
            # Region B needs attention
            target = region_B_center
        elif region_A_mass < 0.3 and region_B_mass < 0.3:
            # Both regions need attention, choose the closer one
            dist_A = np.linalg.norm(agent_pos - region_A_center)
            dist_B = np.linalg.norm(agent_pos - region_B_center)
            target = region_A_center if dist_A < dist_B else region_B_center
        else:
            # Both regions have targets, maintain balance
            # Move towards the farther region to maintain balance
            dist_A = np.linalg.norm(agent_pos - region_A_center)
            dist_B = np.linalg.norm(agent_pos - region_B_center)
            target = region_A_center if dist_A > dist_B else region_B_center

        # Compute angle towards target
        vec = target - agent_pos
        if np.linalg.norm(vec) > 1e-6:  # Avoid division by zero
            ang = np.rad2deg(np.arctan2(vec[1], vec[0]))
            if ang < 0:
                ang += 360
            # Choose nearest angle
            diffs = [min(abs(ang - a), 360 - abs(ang - a)) for a in AGENT_CONTROL_ANGLES]
            best_action = int(np.argmin(diffs))
        else:
            best_action = 0

        self.action_history.append(best_action)
        planning_time = time.time() - start_time
        self.planning_time_history.append(planning_time)
        return best_action
    
    def _compute_region_mass(self, grid: np.ndarray, region) -> float:
        """Compute probability mass sum of specified region"""
        # Convert region coordinates to grid indices
        x_start = int(region[0] / (MAP_WIDTH / GRID_SIZE))
        x_end = int(region[1] / (MAP_WIDTH / GRID_SIZE))
        y_start = int(region[2] / (MAP_HEIGHT / GRID_SIZE))
        y_end = int(region[3] / (MAP_HEIGHT / GRID_SIZE))
        
        # Ensure indices are within valid range
        x_start = max(0, min(x_start, GRID_SIZE - 1))
        x_end = max(0, min(x_end, GRID_SIZE - 1))
        y_start = max(0, min(y_start, GRID_SIZE - 1))
        y_end = max(0, min(y_end, GRID_SIZE - 1))
        
        # Compute probability mass sum of specified region
        region_mass = np.sum(grid[y_start:y_end+1, x_start:x_end+1])
        return region_mass


