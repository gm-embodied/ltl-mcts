"""
Scenario 2 specialized reward functions
Designed for G(ap_exist(R_A) ⟹ F(ap_loc(any, R_A))) LTL specification
"""

import numpy as np
from typing import Dict, Any, Optional
from configs.scenario2_config_clean import (
    R_PATROL_SUCCESS, R_DUTY_VIOLATION, R_TEMPTATION_RESIST, 
    R_APPROACH_A, R_STEP, R_LOCALIZATION_PROGRESS,
    SCENARIO_2_REGION_A, SCENARIO_2_REGION_B,
    LTL_FINALLY_GRACE_PERIOD
)

def compute_scenario2_reward(
    current_state: Dict[str, Any],
    action: int,
    next_state: Dict[str, Any],
    ltl_violation: bool = False
) -> float:
    """
    Scenario 2 specialized reward function: Patrol and Response task
    
    LTL specification: G(ap_exist(R_A) ⟹ F(ap_loc(any, R_A)))
    Core idea: Reward fulfilling patrol duty, penalize being tempted by region B
    
    Args:
        current_state: current state information
        action: executed action
        next_state: next state information
        ltl_violation: Whether an LTL violation has occurred
        
    Returns:
        float: Reward Value
    """
    reward = 0.0
    
    # 1. Terminal Rewards: Severe Penalties for LTL Violations
    if ltl_violation:
        return R_DUTY_VIOLATION   
    
    # 2. Patrol Responsibility Completion Rewards
    duty_completed = next_state.get('duty_completed', False)
    if duty_completed and not current_state.get('duty_completed', False):
        reward += R_PATROL_SUCCESS   
        return reward   
    
    # 3. Positioning Probability Improvement Reward
    current_loc_prob = current_state.get('ap_loc_any_A_prob', 0.0)
    next_loc_prob = next_state.get('ap_loc_any_A_prob', 0.0)
    if next_loc_prob > current_loc_prob:
        loc_improvement = next_loc_prob - current_loc_prob
        reward += R_LOCALIZATION_PROGRESS * loc_improvement   
    
    #  Guidance rewards based on responsibility status
    duty_triggered = next_state.get('duty_triggered', False)
    steps_since_duty = next_state.get('steps_since_duty', 0)
    
    if duty_triggered:
        # Responsibility has been triggered, guiding the agent to fulfill its responsibilities
        
        # 3a. Time urgency: The closer you are to the timeout, the higher the reward.
        urgency_factor = min(1.0, steps_since_duty / LTL_FINALLY_GRACE_PERIOD)
        urgency_reward = urgency_factor * 2.0
        
        # 3b. Position Guidance: Rewards for moving towards Area A
        agent_pos = next_state.get('agent_position', np.array([500.0, 500.0]))
        distance_to_A = calculate_distance_to_region(agent_pos, SCENARIO_2_REGION_A)
        prev_agent_pos = current_state.get('agent_position', agent_pos)
        prev_distance_to_A = calculate_distance_to_region(prev_agent_pos, SCENARIO_2_REGION_A)
        
        #  
        if distance_to_A < prev_distance_to_A:
            approach_reward = R_APPROACH_A * (prev_distance_to_A - distance_to_A) / 100.0
            reward += approach_reward
        
        # 3c. Resist temptation reward: If there is a target in area B but the agent insists on moving towards area A
        distance_to_B = calculate_distance_to_region(agent_pos, SCENARIO_2_REGION_B)
        b_region_has_target = next_state.get('b_region_has_target', False)
        
        if b_region_has_target and distance_to_A < distance_to_B:
            #  
            reward += R_TEMPTATION_RESIST
        
        reward += urgency_reward
    else:
        # Duty not triggered, normal patrol behavior
        # Patrols are encouraged to remain near Area A to prepare for possible duty triggering
        agent_pos = next_state.get('agent_position', np.array([500.0, 500.0]))
        distance_to_A = calculate_distance_to_region(agent_pos, SCENARIO_2_REGION_A)
        
        if distance_to_A < 100.0:   
            reward += 0.1   
    
    # 4. Basic Step Cost
    reward += R_STEP   
    
    return reward

def calculate_distance_to_region(point: np.ndarray, region: list) -> float:
    """
    Calculates the shortest distance from a point to a rectangular region

    Args:
    point: Point coordinates [x, y]
    region: Rectangular region [xmin, xmax, ymin, ymax]

    Returns:
    float: Shortest distance (returns 0 if the point is within the region)
    """
    x, y = point[0], point[1]
    xmin, xmax, ymin, ymax = region
    
    #  
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return 0.0
    
    #  
    dx = max(0, max(xmin - x, x - xmax))
    dy = max(0, max(ymin - y, y - ymax))
    
    return np.sqrt(dx**2 + dy**2)

def compute_belief_guided_reward(
    belief_state: Dict[str, Any],
    ltl_state: Dict[str, Any]
) -> float:
    """
    Belief-based reward guidance

    Args:
        belief_state: PMBM belief state information
        ltl_state: LTL state information
    Returns:
    float: Belief-based reward guidance
    """
    reward = 0.0
    
    # Get atomic proposition probabilities
    ap_exist_A = belief_state.get('ap_exist_A_prob', 0.0)
    ap_loc_A = belief_state.get('ap_loc_A_prob', 0.0)
    
    # Encourage belief in increased A-region target existence
    if ap_exist_A > 0.3:
        reward += ap_exist_A * 0.5
    
    # If duty has been triggered, encourage belief in increased localization probability
    duty_triggered = ltl_state.get('duty_triggered', False)
    if duty_triggered and ap_loc_A > 0.2:
        reward += ap_loc_A * 1.0
    
    return reward

    Returns:
    float: Belief-based reward guidance
    """
    reward = 0.0
    
    #  
    ap_exist_A = belief_state.get('ap_exist_A_prob', 0.0)
    ap_loc_A = belief_state.get('ap_loc_A_prob', 0.0)
    
    #  
    if ap_exist_A > 0.3:
        reward += ap_exist_A * 0.5
    
    #  
    duty_triggered = ltl_state.get('duty_triggered', False)
    if duty_triggered and ap_loc_A > 0.2:
        reward += ap_loc_A * 1.0
    
    return reward

def get_scenario2_rollout_reward(
    trajectory: list,
    final_ltl_state: Dict[str, Any]
) -> float:
    """
    Scenario 2 rollout terminal reward
    
    Args:
        trajectory: trajectory information
        final_ltl_state: final LTL state
        
    Returns:
        float: Terminal reward

   
