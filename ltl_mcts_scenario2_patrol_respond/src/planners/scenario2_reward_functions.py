"""
Scenario 2 Reward Functions - Patrol and Respond Task
Specialized reward functions for G(ap_exist_A => F(ap_loc_any_A)) LTL formula
"""

import numpy as np
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from configs.config import *

def compute_scenario2_reward(world_state, pmbm_belief, ltl_state, action, info: Dict[str, Any]) -> float:
    """
    Enhanced reward for Scenario 2: Patrol and Respond Task
    
    LTL Formula: G(ap_exist_A => F(ap_loc_any_A))
    
    Reward components (optimized based on scenario 1 experience):
    - Large reward for completing patrol duty
    - Large penalty for violating LTL constraint  
    - Task balance reward for maintaining patrol coverage
    - Navigation efficiency reward
    - Localization progress reward
    - Temptation resistance reward
    - Small step penalty for efficiency
    """
    total_reward = 0.0
    
    # Get atomic proposition probabilities
    ap_exist_A = info.get('ap_exist_A', 0.0)
    ap_loc_any_A = info.get('ap_loc_any_A', 0.0)
    patrol_duty_active = info.get('patrol_duty_active', False)
    duty_violation = info.get('duty_violation', False)
    
    # Agent position
    agent_pos = world_state.agent_state.position
    
    # 1. LTL constraint rewards/penalties (highest priority)
    if duty_violation:
        total_reward += R_DUTY_VIOLATION  # Large penalty for violating patrol duty
        # Don't return early - let other rewards accumulate
    elif patrol_duty_active and ap_loc_any_A > LTL_LOC_THRESHOLD:
        total_reward += R_PATROL_SUCCESS  # Large reward for completing patrol duty
        # Don't return early - let other rewards accumulate
    
    # 2. Task balance reward - encourage patrol coverage when no active duty
    if not patrol_duty_active:
        # Reward for staying near patrol region A to be ready
        distance_to_A = _distance_to_region(agent_pos, SCENARIO_2_REGION_A)
        if distance_to_A < 150.0:
            patrol_readiness_reward = R_APPROACH_A * 0.5 * (150.0 - distance_to_A) / 150.0
            total_reward += patrol_readiness_reward
    
    # 3. Active patrol duty rewards (extremely enhanced for LTL_MCTS)
    if patrol_duty_active:
        # Extremely strong reward for being inside region A during active duty
        if _is_in_region(agent_pos, SCENARIO_2_REGION_A):
            total_reward += R_APPROACH_A * 6.0  # Much higher reward
            
            # Very strong reward for localization progress when inside region A
            if ap_loc_any_A > 0.01:
                localization_reward = R_LOCALIZATION_PROGRESS * ap_loc_any_A * 8.0  # Much higher multiplier
                total_reward += localization_reward
        else:
            # Much stronger navigation efficiency - reward direct approach to region A
            distance_to_A = _distance_to_region(agent_pos, SCENARIO_2_REGION_A)
            if distance_to_A < 500.0:  # Increased range
                # Much stronger reward for being closer
                approach_reward = R_APPROACH_A * 3.0 * (500.0 - distance_to_A) / 500.0
                total_reward += approach_reward
            
            # Much stronger penalty for being far from region A during active duty
            if distance_to_A > 300.0:
                total_reward -= 20.0 * (distance_to_A - 300.0) / 300.0  # Much stronger penalty
    
    # 4. Temptation resistance (much more enhanced)
    if _is_in_region(agent_pos, SCENARIO_2_REGION_B):
        # Much stronger penalty for being in distraction region
        if patrol_duty_active:
            total_reward -= abs(R_TEMPTATION_RESIST) * 8.0  # Extremely strong penalty during duty
        else:
            total_reward -= abs(R_TEMPTATION_RESIST) * 3.0  # Strong penalty even without duty
    else:
        # Reward for staying away from distraction
        distance_to_B = _distance_to_region(agent_pos, SCENARIO_2_REGION_B)
        if distance_to_B > 150.0:
            total_reward += R_TEMPTATION_RESIST * 1.0
    
    # 5. Step penalty for efficiency
    total_reward += R_STEP
    
    return total_reward

def get_scenario2_rollout_reward(world_state, pmbm_belief, ltl_state, action, info: Dict[str, Any]) -> float:
    """
    Enhanced reward function for MCTS rollouts in Scenario 2
    Provides stronger guidance signals for better learning
    """
    total_reward = 0.0
    
    # Get key information
    patrol_duty_active = info.get('patrol_duty_active', False)
    duty_violation = info.get('duty_violation', False)
    ap_loc_any_A = info.get('ap_loc_any_A', 0.0)
    ap_exist_A = info.get('ap_exist_A', 0.0)
    
    # Agent position
    agent_pos = world_state.agent_state.position
    
    # Large penalty for duty violation
    if duty_violation:
        total_reward += R_DUTY_VIOLATION  # Strong penalty
    
    # Reward for completing patrol duty
    if patrol_duty_active and ap_loc_any_A > LTL_LOC_THRESHOLD:
        total_reward += R_PATROL_SUCCESS  # Strong reward
    
    # Enhanced distance-based rewards with stronger signals
    distance_to_A = _distance_to_region(agent_pos, SCENARIO_2_REGION_A)
    
    if patrol_duty_active:
        # Extremely strong guidance towards region A during active duty
        if _is_in_region(agent_pos, SCENARIO_2_REGION_A):
            total_reward += R_APPROACH_A * 4.0  # Extremely strong reward for being in region A
            # Very strong progressive reward for localization progress
            if ap_loc_any_A > 0.1:
                loc_bonus = R_LOCALIZATION_PROGRESS * (ap_loc_any_A ** 2) * 4.0  # Quadratic scaling with higher multiplier
                total_reward += loc_bonus
                
                # Extra bonus when approaching the threshold (0.6)
                if ap_loc_any_A > 0.5:
                    threshold_bonus = R_LOCALIZATION_PROGRESS * (ap_loc_any_A - 0.5) * 10.0
                    total_reward += threshold_bonus
        else:
            # Much stronger distance-based reward for approaching region A
            if distance_to_A < 500.0:
                approach_reward = R_APPROACH_A * 2.0 * (500.0 - distance_to_A) / 500.0
                total_reward += approach_reward
            # Much stronger penalty for being far from region A during duty
            if distance_to_A > 200.0:
                total_reward -= 50.0 * (distance_to_A - 200.0) / 300.0
    else:
        # When no duty is active, moderate reward for staying ready
        if distance_to_A < 250.0:
            readiness_reward = R_APPROACH_A * 0.4 * (250.0 - distance_to_A) / 250.0
            total_reward += readiness_reward
    
    # Strong temptation resistance
    if _is_in_region(agent_pos, SCENARIO_2_REGION_B):
        total_reward -= abs(R_TEMPTATION_RESIST) * 2.0  # Strong penalty
    
    # Strong boundary penalty to prevent edge cycling
    boundary_margin = 80.0
    if (agent_pos[0] < boundary_margin or agent_pos[0] > MAP_WIDTH - boundary_margin or 
        agent_pos[1] < boundary_margin or agent_pos[1] > MAP_HEIGHT - boundary_margin):
        # Progressive penalty based on distance to boundary
        edge_penalty = 50.0
        if agent_pos[0] < 20 or agent_pos[0] > MAP_WIDTH - 20 or agent_pos[1] < 20 or agent_pos[1] > MAP_HEIGHT - 20:
            edge_penalty = 100.0  # Very strong penalty for being very close to edge
        total_reward -= edge_penalty
    
    # Step penalty (reduced for rollouts)
    total_reward += R_STEP * 0.5
    
    return total_reward

def _distance_to_region(position: np.ndarray, region: List[float]) -> float:
    """Calculate minimum distance from position to rectangular region"""
    x, y = position[0], position[1]
    x_min, x_max, y_min, y_max = region
    
    # If inside region, distance is 0
    if x_min <= x <= x_max and y_min <= y <= y_max:
        return 0.0
    
    # Calculate minimum distance to region boundary
    dx = max(0, max(x_min - x, x - x_max))
    dy = max(0, max(y_min - y, y - y_max))
    
    return np.sqrt(dx**2 + dy**2)

def _is_in_region(position: np.ndarray, region: List[float]) -> bool:
    """Check if position is inside rectangular region"""
    x, y = position[0], position[1]
    x_min, x_max, y_min, y_max = region
    
    return x_min <= x <= x_max and y_min <= y <= y_max