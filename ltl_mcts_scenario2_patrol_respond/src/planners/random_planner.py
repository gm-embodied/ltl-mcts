"""
Random planner implementation - the most basic baseline algorithm
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.planners.base_planner import BasePlanner
from configs.config import AGENT_CONTROL_ANGLES

class RandomPlanner(BasePlanner):
    """Random planner - completely random action selection"""
    
    def __init__(self, config=None):
        super().__init__("Random", config)
        
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """Completely random action selection, ignore all inputs"""
        start_time = time.time()
        action = np.random.randint(0, len(AGENT_CONTROL_ANGLES))
        
        planning_time = time.time() - start_time
        self.planning_time_history.append(planning_time)
        self.action_history.append(action)
        
        return action
