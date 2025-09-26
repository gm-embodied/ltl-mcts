"""
LTL myopic planner implementation
Understands LTL tasks but only makes single-step lookahead greedy decisions
"""

import numpy as np
import time
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.planners.base_planner import BasePlanner
from configs.config import (
    AGENT_CONTROL_ANGLES, AGENT_SPEED, TIME_STEP_DURATION,
    SENSOR_RANGE, SENSOR_FOV_DEGREES, MAP_WIDTH, MAP_HEIGHT,
    LTL_CONFIRM_THRESHOLD_TAU, REGION_A, REGION_B
)

class LTLMyopicPlanner(BasePlanner):
    """LTL myopic planner"""
    
    def __init__(self, config=None):
        super().__init__("LTL_Myopic", config)
        
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        Plan next action: select single-step action most favorable for LTL task
        Uses same semantic-probabilistic bridge as LTL-MCTS, but only makes greedy choices
        """
        start_time = time.time()
        
        agent_pos = world_state.agent_state.position
        best_action = 0
        best_score = -float('inf')
        
        # Iterate through all possible control angles
        for action_idx, angle_deg in enumerate(AGENT_CONTROL_ANGLES):
            score = self._evaluate_ltl_reward(agent_pos, angle_deg, pmbm_belief, ltl_state)
            
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        # Record statistics
        planning_time = time.time() - start_time
        self.planning_time_history.append(planning_time)
        self.action_history.append(best_action)
        
        return best_action
    
    def _evaluate_ltl_reward(self, agent_pos: np.ndarray, angle_deg: float,
                           pmbm_belief, ltl_state) -> float:
        """
        Evaluate the single-step contribution of moving in a certain direction to the LTL task
        
        Args:
            agent_pos: Agent current position
            angle_deg: Control angle (degrees)
            pmbm_belief: PMBM belief state
            ltl_state: LTL state
            
        Returns:
            float: LTL reward score
        """
        angle_rad = np.deg2rad(angle_deg)
        
        # Predict agent's position after movement
        velocity = AGENT_SPEED * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        predicted_pos = agent_pos + velocity * TIME_STEP_DURATION
        
        # Boundary check
        predicted_pos[0] = np.clip(predicted_pos[0], 0, MAP_WIDTH)
        predicted_pos[1] = np.clip(predicted_pos[1], 0, MAP_HEIGHT)
        
        # Compute LTL reward
        ltl_reward = self._compute_ltl_reward(predicted_pos, angle_rad, pmbm_belief, ltl_state)
        
        return ltl_reward
    
    def _compute_ltl_reward(self, sensor_pos: np.ndarray, sensor_heading: float,
                          pmbm_belief, ltl_state) -> float:
        """
        Compute LTL task-related rewards
        For persistent surveillance task: G(F(ap_exist_A)) & G(F(ap_exist_B))
        Need to maintain targets in both regions
        """
        total_reward = 0.0
        
        # Get current MAP hypothesis
        map_hypothesis = pmbm_belief.get_map_hypothesis()
        if map_hypothesis is None:
            # If no target hypothesis, encourage moving towards surveillance region
            return self._compute_exploration_reward(sensor_pos)
        
        # Compute the existence probability of two surveillance regions
        p_exist_A = self._prob_exist_in_region(map_hypothesis, REGION_A)
        p_exist_B = self._prob_exist_in_region(map_hypothesis, REGION_B)
        
        # Region center position
        region_A_center = np.array([(REGION_A[0] + REGION_A[1])/2, (REGION_A[2] + REGION_A[3])/2])
        region_B_center = np.array([(REGION_B[0] + REGION_B[1])/2, (REGION_B[2] + REGION_B[3])/2])
        
        dist_to_A = np.linalg.norm(sensor_pos - region_A_center)
        dist_to_B = np.linalg.norm(sensor_pos - region_B_center)
        
        # Fixed reward strategy: based on LTL task requirements
        # 1. If the existence probability of a region is low, need to explore/maintain
        urgency_A = max(0, 0.5 - p_exist_A)  # The lower the existence probability, the more urgent
        urgency_B = max(0, 0.5 - p_exist_B)
        
        # 2. Distance reward: encourage moving towards urgent region
        if urgency_A > 0:
            reward_A = max(0, 5.0 - dist_to_A / 100.0) * (urgency_A + 1.0)
            total_reward += reward_A
            
        if urgency_B > 0:
            reward_B = max(0, 5.0 - dist_to_B / 100.0) * (urgency_B + 1.0)
            total_reward += reward_B
        
        # 3. Maintain reward: if both regions have targets, give maintain reward
        if p_exist_A > 0.4 and p_exist_B > 0.4:
            total_reward += 10.0
        
        # 4. Balance reward: encourage balancing attention to two regions
        if p_exist_A > 0.3 and p_exist_B > 0.3:
            balance_bonus = 5.0 * (1.0 - abs(p_exist_A - p_exist_B))
            total_reward += balance_bonus
        
        # 5. Basic exploration reward: if no clear target, move towards the nearest region
        if urgency_A == 0 and urgency_B == 0:
            if dist_to_A < dist_to_B:
                total_reward += max(0, 3.0 - dist_to_A / 150.0)
            else:
                total_reward += max(0, 3.0 - dist_to_B / 150.0)
        
        return total_reward
    
    def _prob_exist_in_region(self, map_hypothesis, region) -> float:
        """Compute the probability of target existence in the region"""
        prob_no_targets = 1.0
        
        for component in map_hypothesis.bernoulli_components:
            if component.existence_prob > 0.1:  # Only consider meaningful existence probability
                # Compute the probability of target existence in the region
                pos_mean = component.get_position_mean()
                pos_cov = component.get_position_cov()
                
                # Simplified calculation: if the target mean is in the region and the existence probability is high, it is considered that there is a target in the region
                if (region[0] <= pos_mean[0] <= region[1] and 
                    region[2] <= pos_mean[1] <= region[3]):
                    prob_no_targets *= (1.0 - component.existence_prob)
        
        return 1.0 - prob_no_targets
    
    def _compute_exploration_reward(self, sensor_pos: np.ndarray) -> float:
        """Compute exploration reward: encourage moving towards surveillance region"""
        region_A_center = np.array([(REGION_A[0] + REGION_A[1])/2, (REGION_A[2] + REGION_A[3])/2])
        region_B_center = np.array([(REGION_B[0] + REGION_B[1])/2, (REGION_B[2] + REGION_B[3])/2])
        
        dist_to_A = np.linalg.norm(sensor_pos - region_A_center)
        dist_to_B = np.linalg.norm(sensor_pos - region_B_center)
        
        # Select the nearest region to give reward
        min_dist = min(dist_to_A, dist_to_B)
        return max(0, 5.0 - min_dist / 100.0)
    
    def _compute_confirmation_reward(self, component, distance: float) -> float:
        """
        Compute confirmation reward
        The reward is proportional to the "near confirmation" degree of the target
        """
        # Current covariance trace
        current_trace = component.get_covariance_trace()
        
        # If already confirmed, give maintain reward
        if current_trace < LTL_CONFIRM_THRESHOLD_TAU:
            return 10.0 * component.existence_prob
        
        # Otherwise, the reward is proportional to the "near confirmation" degree
        # The higher the covariance trace is closer to the threshold, the higher the reward
        trace_ratio = LTL_CONFIRM_THRESHOLD_TAU / max(current_trace, 1.0)
        
        # The higher the existence probability, the higher the reward
        existence_factor = component.existence_prob
        
        # The closer the distance, the better the observation effect
        distance_factor = max(0.1, 1.0 - distance / SENSOR_RANGE)
        
        # Basic confirmation reward
        base_reward = 5.0
        
        confirm_reward = base_reward * trace_ratio * existence_factor * distance_factor
        
        return confirm_reward
    
    def _compute_exploration_reward(self, sensor_pos: np.ndarray, sensor_heading: float) -> float:
        """
        Compute exploration reward (when there is no tracked target)
        """
        # Simplified exploration reward: encourage visiting different areas of the map
        center_pos = np.array([MAP_WIDTH/2, MAP_HEIGHT/2])
        distance_from_center = np.linalg.norm(sensor_pos - center_pos)
        max_distance = np.linalg.norm(center_pos)
        
        # When the distance from the center is moderate, the exploration reward is highest
        normalized_distance = distance_from_center / max_distance
        exploration_reward = 2.0 * (1.0 - abs(normalized_distance - 0.7))  # The highest at 70% distance
        
        return exploration_reward
    
    def _is_position_in_fov(self, position: np.ndarray, sensor_pos: np.ndarray,
                           sensor_heading: float) -> bool:
        """Check if the position is within the sensor field of view"""
        # Calculate the angle of the position relative to the sensor
        relative_pos = position - sensor_pos
        target_angle = np.arctan2(relative_pos[1], relative_pos[0])
        
        # Calculate the angle difference
        angle_diff = np.abs(target_angle - sensor_heading)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Handle angle wrapping
        
        # Check if it is within the field of view
        fov_rad = np.deg2rad(SENSOR_FOV_DEGREES) / 2
        return angle_diff <= fov_rad
    
    def _compute_potential_based_reward(self, ltl_state) -> float:
        """
        Compute the reward based on the potential function
        The potential function is based on the shortest distance to the LTL accepting state
        """
        # Simplified implementation: there are only two states in scenario one
        if ltl_state.automaton_state == 0:  # Initial state
            return 0.0
        elif ltl_state.automaton_state == 1:  # Accepting state
            return 1.0
        else:
            return 0.0
    
    def get_ltl_value_map(self, pmbm_belief, ltl_state, grid_size: int = 50) -> np.ndarray:
        """
        Generate LTL value map for visualization
        Return the LTL reward value for each grid point
        """
        value_map = np.zeros((grid_size, grid_size))
        
        x_coords = np.linspace(0, MAP_WIDTH, grid_size)
        y_coords = np.linspace(0, MAP_HEIGHT, grid_size)
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                pos = np.array([x, y])
                # Compute the LTL reward in the east direction as an example
                reward = self._compute_ltl_reward(pos, 0.0, pmbm_belief, ltl_state)
                value_map[j, i] = reward  # Note: j corresponds to the y-axis, i corresponds to the x-axis
        
        return value_map
