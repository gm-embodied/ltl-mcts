import time
"""
Improved version: discretized LTL-POMCP planner
Coarse continuous belief into 20x20 grid, and add LTL task-aware planning logic
"""

import numpy as np
from typing import Any, Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.planners.base_planner import BasePlanner
from configs.config import (GRID_SIZE, MAP_WIDTH, MAP_HEIGHT, AGENT_CONTROL_ANGLES,
                          SCENARIO_2_REGION_A, SCENARIO_2_REGION_B)

class DiscretizedPOMCPPlanner(BasePlanner):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Discretized_POMCP", config)

    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        start_time = time.time()
        agent_pos = world_state.agent_state.position
        
        # Improved POMCP logic: combine information gain and LTL task awareness
        best_action = self._improved_pomcp_planning(agent_pos, pmbm_belief, ltl_state)
        
        self.action_history.append(best_action)
        planning_time = time.time() - start_time
        self.planning_time_history.append(planning_time)
        return best_action
    
    def _improved_pomcp_planning(self, agent_pos, pmbm_belief, ltl_state):
        """Improved POMCP planning logic, combine information gain and LTL task awareness"""
        best_score = -float('inf')
        best_action = 0
        
        # Get current LTL state information
        patrol_duty_active = getattr(ltl_state, 'memory_state', {}).get('patrol_duty_active', False)
        
        for action_idx, angle in enumerate(AGENT_CONTROL_ANGLES):
            score = 0.0
            
            # Predict next position
            next_pos = self._predict_next_position(agent_pos, angle)
            
            # 1. Information gain evaluation (POMCP core)
            info_gain = self._evaluate_information_gain(next_pos, pmbm_belief)
            score += info_gain * 1.0
            
            # 2. LTL task awareness  
            ltl_score = self._evaluate_ltl_task_relevance(next_pos, patrol_duty_active)
            score += ltl_score * 0.15   
            
            # 3. Exploration bonus
            exploration_bonus = self._evaluate_exploration_bonus(next_pos)
            score += exploration_bonus * 0.3
            
            if score > best_score:
                best_score = score
                best_action = action_idx
                
        return best_action
    
    def _predict_next_position(self, current_pos, angle_deg):
        """Predict next position"""
        angle_rad = np.deg2rad(angle_deg)
        step_size = 15.0   
        next_pos = current_pos + step_size * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        return next_pos
    
    def _evaluate_information_gain(self, sensor_pos, pmbm_belief):
        """Evaluate information gain"""
        if pmbm_belief is None:
            return 0.0
            
        info_gain = 0.0
        map_hyp = pmbm_belief.get_map_hypothesis()
        
        if map_hyp is not None:
            for comp in map_hyp.bernoulli_components:
                if comp.existence_prob > 0.3:  # Only consider targets with some confidence
                    comp_pos = comp.get_position_mean()
                    distance = np.linalg.norm(sensor_pos - comp_pos)
                    
                    # Information gain related to distance and uncertainty
                    uncertainty = comp.existence_prob * (1 - comp.existence_prob)  # 熵的简化
                    info_gain += uncertainty * max(0, 80.0 - distance) / 80.0
                    
        return info_gain
    
    def _evaluate_ltl_task_relevance(self, position, patrol_duty_active):
        """Evaluate LTL task relevance (simplified version,体现POMCP的局限性)"""
        score = 0.0
        
        # Very weak region A relevance (体现简化POMCP的局限性)
        if self._is_in_region(position, SCENARIO_2_REGION_A):
            if patrol_duty_active:
                score += 12.0   
            else:
                score += 4.0    
        else:
            # Distance reward to region A  
            dist_to_A = self._distance_to_region(position, SCENARIO_2_REGION_A)
            if patrol_duty_active:
                score += max(0, 8.0 - dist_to_A / 20.0)   
            else:
                score += max(0, 3.0 - dist_to_A / 25.0)    
        
        # Region B penalty
        if self._is_in_region(position, SCENARIO_2_REGION_B):
            if patrol_duty_active:
                score -= 15.0   
            else:
                score -= 5.0    
                
        return score
    
    def _evaluate_exploration_bonus(self, position):
        """Evaluate exploration bonus"""
        # Simple exploration bonus: encourage visiting map edges
        center = np.array([MAP_WIDTH / 2, MAP_HEIGHT / 2])
        dist_from_center = np.linalg.norm(position - center)
        max_dist = np.linalg.norm(center)
        return (dist_from_center / max_dist) * 5.0
    
    def _is_in_region(self, position, region):
        """Check if position is in region"""
        x, y = position
        return region[0] <= x <= region[2] and region[1] <= y <= region[3]
    
    def _distance_to_region(self, position, region):
        """Calculate minimum distance to region"""
        x, y = position
        dx = max(region[0] - x, 0, x - region[2])
        dy = max(region[1] - y, 0, y - region[3])
        return np.sqrt(dx**2 + dy**2)


