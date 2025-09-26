"""
Passive planner implementation - Scenario 1 version
Stationary rotating sensor, no position movement, completely passive behavior
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
    """Passive planner - stationary rotating sensor, no movement"""
    
    def __init__(self, config=None):
        super().__init__("Passive", config)
        # rotation parameters: 8 directions, hold each direction for 1 step (minimal circular motion)
        self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.steps_per_angle = 1  # steps per angle, minimized for near-stationary circular motion
        self.current_angle_index = 0
        self.angle_step_counter = 0
        
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        Plan next action: passive strategy - small-range circular motion in place
        Completely ignores world_state, pmbm_belief and ltl_state information
        The agent does a very small circular motion, basically staying in place, but the sensor will rotate
        """
        start_time = time.time()
        
        # Update angle counter
        self.angle_step_counter += 1
        
        # Check if it is time to switch to the next angle
        if self.angle_step_counter >= self.steps_per_angle:
            self.angle_step_counter = 0
            self.current_angle_index = (self.current_angle_index + 1) % len(self.rotation_angles)
        
        # Get current target angle
        target_angle = self.rotation_angles[self.current_angle_index]
        
        # Passive strategy: do a small range circular motion, hold each angle for several steps
        # So the agent basically stays in place, but the sensor orientation will change
        action = self._find_closest_control_angle(target_angle)
        
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
            # Calculate angle difference (considering wrapping)
            diff = abs(target_angle_deg - control_angle)
            diff = min(diff, 360 - diff)
            
            if diff < min_diff:
                min_diff = diff
                best_action = i
        
        return best_action
    
    def reset(self):
        """Reset the planner state"""
        super().reset()
        self.current_angle_index = 0
        self.angle_step_counter = 0
    
    def get_current_rotation_angle(self) -> float:
        """Get the current rotation angle (for visualization)"""
        return self.rotation_angles[self.current_angle_index]
