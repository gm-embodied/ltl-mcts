"""
Information gain myopic planner implementation
Selects actions that maximize information gain at each step, completely ignoring LTL tasks
"""

import numpy as np
import time
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.planners.base_planner import BasePlanner
from configs.config import (
    AGENT_CONTROL_ANGLES, AGENT_SPEED, TIME_STEP_DURATION,
    SENSOR_RANGE, SENSOR_FOV_DEGREES, MAP_WIDTH, MAP_HEIGHT,
    REGION_A, REGION_B
)

class InfoMyopicPlanner(BasePlanner):
    """Information gain myopic planner"""
    
    def __init__(self, config=None):
        super().__init__("Info_Myopic", config)
        
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        Plan next action: select action that maximizes information gain
        Completely ignores LTL tasks, only cares about reducing belief uncertainty
        """
        start_time = time.time()
        
        agent_pos = world_state.agent_state.position
        best_action = 0
        best_score = -float('inf')
        
        # Iterate over all possible control angles
        for action_idx, angle_deg in enumerate(AGENT_CONTROL_ANGLES):
            score = self._evaluate_information_gain(agent_pos, angle_deg, pmbm_belief, ltl_state)
            
            if score > best_score:
                best_score = score
                best_action = action_idx
        
        # Record statistics
        planning_time = time.time() - start_time
        self.planning_time_history.append(planning_time)
        self.action_history.append(best_action)
        
        return best_action
    
    def _evaluate_information_gain(self, agent_pos: np.ndarray, angle_deg: float, 
                                 pmbm_belief, ltl_state=None) -> float:
        """
        Evaluate information gain for moving and observing in a specific direction (task-aware version)
        
        Args:
            agent_pos: Agent current position
            angle_deg: Control angle (degrees)
            pmbm_belief: PMBM belief state
            ltl_state: LTL state (for task awareness)
            
        Returns:
            float: Information gain score
        """
        angle_rad = np.deg2rad(angle_deg)
        
        # Predict agent's position after movement
        velocity = AGENT_SPEED * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        predicted_pos = agent_pos + velocity * TIME_STEP_DURATION
        
        # Boundary check
        predicted_pos[0] = np.clip(predicted_pos[0], 0, MAP_WIDTH)
        predicted_pos[1] = np.clip(predicted_pos[1], 0, MAP_HEIGHT)
        
        # Compute basic information gain
        fov_score = self._compute_fov_information_score(predicted_pos, angle_rad, pmbm_belief)
        
        # Add task-aware balance mechanism
        task_balance_score = self._compute_task_balance_score(predicted_pos, pmbm_belief, ltl_state)
        
        # If task balance score > 0, there is a region that needs balance, prioritize task
        if task_balance_score > 0:
            total_score = 0.05 * fov_score + 0.95 * task_balance_score  # Almost completely rely on task balance
        else:
            # If there is no task balance requirement, consider information gain
            total_score = fov_score + 1.0  # 给基础探索分数
        
        return total_score
    
    def _compute_fov_information_score(self, sensor_pos: np.ndarray, sensor_heading: float,
                                     pmbm_belief) -> float:
        """
        Compute information score within sensor field of view
        Score based on uncertainty of target existence in field of view
        """
        total_score = 0.0
        
        # Get current MAP hypothesis
        map_hypothesis = pmbm_belief.get_map_hypothesis()
        if map_hypothesis is None:
            return 0.0
        
        # 1. Evaluate information value of existing Bernoulli components
        for component in map_hypothesis.bernoulli_components:
            if self._is_position_in_fov(component.get_position_mean(), sensor_pos, sensor_heading):
                # Distance check
                distance = np.linalg.norm(component.get_position_mean() - sensor_pos)
                if distance <= SENSOR_RANGE:
                    # Information gain is proportional to uncertainty of existence probability
                    # When existence probability is close to 0.5, uncertainty is maximum
                    uncertainty = 4 * component.existence_prob * (1 - component.existence_prob)
                    
                    # Information gain is also proportional to covariance size (the higher the uncertainty, the higher the observation value)
                    covariance_factor = np.sqrt(component.get_covariance_trace())
                    
                    # Distance penalty (the farther the observation, the lower the precision)
                    distance_factor = max(0.1, 1.0 - distance / SENSOR_RANGE)
                    
                    component_score = uncertainty * covariance_factor * distance_factor
                    total_score += component_score
        
        # 2. Evaluate information value of Poisson components (undetected targets)
        if pmbm_belief.poisson_component is not None:
            # Compute expected number of targets in field of view - using simplified global expectation
            try:
                # Simplified: use fixed expected target density
                fov_area = self._compute_fov_area(sensor_pos, sensor_heading)
                # Assume 0.0001 targets per square unit
                expected_density = 0.0001
                expected_targets_in_fov = expected_density * fov_area
                
                # Information gain of Poisson components
                poisson_score = expected_targets_in_fov * 2.0  # Increase weight
                total_score += poisson_score
            except:
                # If Poisson component calculation fails, give a basic exploration score
                fov_area = self._compute_fov_area(sensor_pos, sensor_heading)
                total_score += fov_area / 50000.0
        
        # 3. Add exploration reward (encourage visiting regions with insufficient observation)
        exploration_score = self._compute_exploration_score(sensor_pos, sensor_heading)
        total_score += exploration_score
        
        return total_score
    
    def _is_position_in_fov(self, position: np.ndarray, sensor_pos: np.ndarray, 
                           sensor_heading: float) -> bool:
        """Check if position is in sensor field of view"""
        # Compute position relative to sensor angle
        relative_pos = position - sensor_pos
        target_angle = np.arctan2(relative_pos[1], relative_pos[0])
        
        # Compute angle difference
        angle_diff = np.abs(target_angle - sensor_heading)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Handle angle wrapping
        
        # Check if within field of view angle
        fov_rad = np.deg2rad(SENSOR_FOV_DEGREES) / 2
        return angle_diff <= fov_rad
    
    def _compute_fov_area(self, sensor_pos: np.ndarray, sensor_heading: float) -> float:
        """Compute effective area of sensor field of view"""
        # Simplified calculation: sector area
        fov_rad = np.deg2rad(SENSOR_FOV_DEGREES)
        sector_area = 0.5 * SENSOR_RANGE**2 * fov_rad
        
        # Consider map boundary limit (simplified handling)
        boundary_factor = 1.0
        if sensor_pos[0] < SENSOR_RANGE or sensor_pos[0] > MAP_WIDTH - SENSOR_RANGE:
            boundary_factor *= 0.8
        if sensor_pos[1] < SENSOR_RANGE or sensor_pos[1] > MAP_HEIGHT - SENSOR_RANGE:
            boundary_factor *= 0.8
        
        return sector_area * boundary_factor
    
    def _compute_exploration_score(self, sensor_pos: np.ndarray, sensor_heading: float) -> float:
        """
        Compute exploration reward score
        Encourage agent to visit surveillance regions to obtain more information
        """
        # Modify strategy: encourage visiting surveillance regions instead of map edges
        region_A_center = np.array([(REGION_A[0] + REGION_A[1])/2, (REGION_A[2] + REGION_A[3])/2])
        region_B_center = np.array([(REGION_B[0] + REGION_B[1])/2, (REGION_B[2] + REGION_B[3])/2])
        
        dist_to_A = np.linalg.norm(sensor_pos - region_A_center)
        dist_to_B = np.linalg.norm(sensor_pos - region_B_center)
        
        # The closer to surveillance region, the higher the exploration reward
        max_dist = np.linalg.norm(region_A_center - region_B_center)
        
        # Significantly increase the weight of exploration reward in surveillance regions
        min_dist = min(dist_to_A, dist_to_B)
        exploration_score = max(0, (max_dist - min_dist) / max_dist) * 10.0  # Increase weight from 1.0 to 10.0
        
        # Additional reward: if heading towards surveillance region
        heading_vector = np.array([np.cos(sensor_heading), np.sin(sensor_heading)])
        
        # Check if heading towards the closer region
        if min_dist == dist_to_A:
            direction_to_region = (region_A_center - sensor_pos) / (dist_to_A + 1e-6)
        else:
            direction_to_region = (region_B_center - sensor_pos) / (dist_to_B + 1e-6)
        
        # Compute heading consistency, significantly increase weight
        heading_alignment = max(0, np.dot(heading_vector, direction_to_region))
        exploration_score += heading_alignment * 5.0   
        
        # Add forced surveillance region access reward
        # If close to surveillance region, give additional large reward
        if dist_to_A < 100.0:
            exploration_score += 15.0
        if dist_to_B < 100.0:
            exploration_score += 15.0
        
        return exploration_score
    
    def _compute_task_balance_score(self, sensor_pos: np.ndarray, pmbm_belief, ltl_state) -> float:
        """
        Compute task balance score - encourage visiting ignored regions
        """
        # Compute region existence probability
        map_hypothesis = pmbm_belief.get_map_hypothesis()
        if map_hypothesis is None:
            # If no target hypothesis, use default low probability and apply aggressive strategy
            p_exist_A = 0.33  # Default low probability
            p_exist_B = 0.33  # Default low probability
            
            # Continue executing the aggressive strategy code below
        else:
            # Compute region existence probability
            p_exist_A = self._prob_exist_in_region(map_hypothesis, REGION_A)
            p_exist_B = self._prob_exist_in_region(map_hypothesis, REGION_B)

        # Region center
        region_A_center = np.array([(REGION_A[0] + REGION_A[1])/2, (REGION_A[2] + REGION_A[3])/2])
        region_B_center = np.array([(REGION_B[0] + REGION_B[1])/2, (REGION_B[2] + REGION_B[3])/2])
        
        dist_to_A = np.linalg.norm(sensor_pos - region_A_center)
        dist_to_B = np.linalg.norm(sensor_pos - region_B_center)
        
        task_score = 0.0
        
        # Compute region urgency (the lower the probability, the higher the urgency)
        urgency_A = max(0, 0.7 - p_exist_A)  
        urgency_B = max(0, 0.7 - p_exist_B)
        
        # Forced balance: if region B is long ignored, give huge reward
        if p_exist_B < 0.4:  # Region B is severely ignored
            task_score += max(0, 50.0 - dist_to_B / 20.0)  # Huge reward to guide to B
        
        if p_exist_A < 0.4:  # Region A is severely ignored
            task_score += max(0, 50.0 - dist_to_A / 20.0)  # Huge reward to guide to A
        
        # If some region is urgent but not severely ignored, give moderate reward
        if urgency_A > 0 and p_exist_A >= 0.4:
            task_score += max(0, 30.0 - dist_to_A / 25.0) * (urgency_A + 0.5)
        
        if urgency_B > 0 and p_exist_B >= 0.4:
            task_score += max(0, 30.0 - dist_to_B / 25.0) * (urgency_B + 0.5)
        
        # If both regions have enough targets, force balance access
        if urgency_A == 0 and urgency_B == 0:
            # Force to visit the farther region
            if dist_to_A < dist_to_B:
                task_score += max(0, 25.0 - dist_to_B / 30.0)  # Increase reward to B
            else:
                task_score += max(0, 25.0 - dist_to_A / 30.0)  # Increase reward to A
        
        return task_score
    
    def _prob_exist_in_region(self, map_hypothesis, region) -> float:
        """Compute probability of target existence in region"""
        prob_no_targets = 1.0
        
        for component in map_hypothesis.bernoulli_components:
            if component.existence_prob > 0.1:  # Only consider meaningful existence probability
                # Compute probability of target in region
                pos_mean = component.get_position_mean()
                
                # Simplified calculation: if target mean is in region and existence probability is high, consider target in region
                if (region[0] <= pos_mean[0] <= region[1] and 
                    region[2] <= pos_mean[1] <= region[3]):
                    prob_no_targets *= (1.0 - component.existence_prob)
        
        return 1.0 - prob_no_targets
    
    def get_information_map(self, pmbm_belief, grid_size: int = 50) -> np.ndarray:
        """
        Generate information map for visualization
        Return information value of each grid point
        """
        info_map = np.zeros((grid_size, grid_size))
        
        x_coords = np.linspace(0, MAP_WIDTH, grid_size)
        y_coords = np.linspace(0, MAP_HEIGHT, grid_size)
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                pos = np.array([x, y])
                # Compute information score towards east as an example
                score = self._compute_fov_information_score(pos, 0.0, pmbm_belief)
                info_map[j, i] = score  # Note: j corresponds to y axis, i corresponds to x axis
        
        return info_map
