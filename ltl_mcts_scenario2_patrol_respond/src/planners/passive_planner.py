"""
Passive planner implementation - scenario 2 version
Rotate the sensor, not move position, reference scenario 1 design
"""

import numpy as np
import time
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.planners.base_planner import BasePlanner
from configs.config import AGENT_CONTROL_ANGLES

class PassivePlanner(BasePlanner):
    """Passive planner - rotate the sensor"""
    
    def __init__(self, config=None):
        super().__init__("Passive", config)
        # Rotation parameters
        self.rotation_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        self.steps_per_angle = 8  # Number of steps per angle
        self.current_angle_index = 0
        self.angle_step_counter = 0
        
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        Plan next action: passive strategy - fixed direction movement
        Completely ignore world_state, pmbm_belief and ltl_state information
        Fix: no longer rotate, but fixed direction movement, more consistent with "passive" concept
        """
        start_time = time.time()
        
        # Passive strategy: always choose action 0 (move towards 0 degree direction)
        # This way the agent will move along a fixed direction, truly "passive"
        action = 0  # Always move towards east direction
        
        # Record statistics
        planning_time = time.time() - start_time
        self.planning_time_history.append(planning_time)
        self.action_history.append(action)
        
        return action
    
    def _find_closest_control_angle(self, target_angle_deg: float) -> int:
        """Find the control angle index closest to the target angle"""
        min_diff = float('inf')
        best_action = 0
        
        for i, control_angle in enumerate(AGENT_CONTROL_ANGLES):
            # Calculate angle difference (consider wrapping)
            diff = abs(target_angle_deg - control_angle)
            diff = min(diff, 360 - diff)
            
            if diff < min_diff:
                min_diff = diff
                best_action = i
        
        return best_action
    
    def reset(self):
        """Reset planner state"""
        super().reset()
        self.current_angle_index = 0
        self.angle_step_counter = 0
    
    def get_current_rotation_angle(self) -> float:
        """Get current rotation angle (for visualization)"""
        return self.rotation_angles[self.current_angle_index]
