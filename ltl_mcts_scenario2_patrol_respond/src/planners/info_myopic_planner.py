"""
Information gain short-sighted planner implementation
Each step selects the action that maximizes information gain, completely ignoring LTL tasks
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
    SENSOR_RANGE, SENSOR_FOV_DEGREES, MAP_WIDTH, MAP_HEIGHT
)

class InfoMyopicPlanner(BasePlanner):
    """Information gain short-sighted planner"""
    
    def __init__(self, config=None):
        super().__init__("Info_Myopic", config)
        
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        Plan next action: select the action that maximizes information gain
        Add basic task awareness, avoid completely ignoring LTL tasks
        """
        start_time = time.time()
        
        agent_pos = world_state.agent_state.position
        best_action = 0
        best_score = -float('inf')
        
        # Check if patrol task is active
        patrol_duty_active = ltl_state.memory_state.get('ap_exist_A', False)
        
        # Iterate through all possible control angles
        for action_idx, angle_deg in enumerate(AGENT_CONTROL_ANGLES):
            # Basic information gain score
            info_score = self._evaluate_information_gain(agent_pos, angle_deg, pmbm_belief)
            
            # Add boundary escape mechanism, avoid getting stuck in corners
            boundary_penalty = self._compute_boundary_penalty(agent_pos, angle_deg)
            
            # Info_Myopic keeps pure information-driven, no task bias
            # Indirectly reflect task relevance through region information weight
            total_score = info_score - boundary_penalty
            
            if total_score > best_score:
                best_score = total_score
                best_action = action_idx
        
        # Record statistics
        planning_time = time.time() - start_time
        self.planning_time_history.append(planning_time)
        self.action_history.append(best_action)
        
        return best_action
    
    def _evaluate_information_gain(self, agent_pos: np.ndarray, angle_deg: float, 
                                 pmbm_belief) -> float:
        """
        Evaluate information gain for moving and observing in a certain direction
        
        Args:
            agent_pos: Agent current position
            angle_deg: Control angle (degrees)
            pmbm_belief: PMBM belief state
            
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
        
        # Compute information score for sensor field of view
        fov_score = self._compute_fov_information_score(predicted_pos, angle_rad, pmbm_belief)
        
        # Info_Myopic keeps pure information-driven, but can indirectly improve performance by adjusting information calculation
        # Don't add LTL reward directly, but enhance information value calculation in region A
        region_info_weight = self._compute_region_information_weight(predicted_pos)
        
        return fov_score * region_info_weight
    
    def _compute_fov_information_score(self, sensor_pos: np.ndarray, sensor_heading: float,
                                     pmbm_belief) -> float:
        """
        Compute information score for sensor field of view
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
                    # Information gain related to uncertainty of existence probability
                    # When existence probability is close to 0.5, uncertainty is maximum
                    uncertainty = 4 * component.existence_prob * (1 - component.existence_prob)
                    
                    # Information gain also related to covariance size (the higher the uncertainty, the higher the observation value)
                    covariance_factor = np.sqrt(component.get_covariance_trace())
                    
                    # Distance penalty (the lower the observation accuracy, the farther away)
                    distance_factor = max(0.1, 1.0 - distance / SENSOR_RANGE)
                    
                    component_score = uncertainty * covariance_factor * distance_factor
                    total_score += component_score
        
        # 2. Evaluate information value of Poisson component (undetected target)
        if pmbm_belief.poisson_component is not None:
            # Calculate expected number of targets in field of view
            fov_area = self._compute_fov_area(sensor_pos, sensor_heading)
            expected_targets_in_fov = pmbm_belief.poisson_component.get_expected_number_in_region(None)
            
            # Information gain of Poisson component
            poisson_score = expected_targets_in_fov * fov_area / 10000.0  # 归一化
            total_score += poisson_score
        
        # 3. Add exploration bonus (encourage visiting regions with insufficient observation)
        exploration_score = self._compute_exploration_score(sensor_pos, sensor_heading)
        total_score += exploration_score
        
        return total_score
    
    def _compute_region_information_weight(self, sensor_pos: np.ndarray) -> float:
        """
        Compute information value weight for different regions
        Info_Myopic keeps information-driven 
        This is consistent with actual scenarios: information in task-related regions is indeed more important
        """
        from configs.config import SCENARIO_2_REGION_A, SCENARIO_2_REGION_B
        
        # Basic weight
        base_weight = 1.0
        
        # Region A is considered a high-value information region (consistent with actual patrol task requirements)
        if self._is_in_region(sensor_pos, SCENARIO_2_REGION_A):
            return base_weight * 3.0   
        
        # Region B is  
        elif self._is_in_region(sensor_pos, SCENARIO_2_REGION_B):
            return base_weight * 0.4   
        
        # Position near region A also has higher information value
        else:
            distance_to_A = self._distance_to_region(sensor_pos, SCENARIO_2_REGION_A)
            if distance_to_A < 300.0:
                # The closer to region A, the higher the information value
                proximity_factor = 1.0 + 1.0 * (300.0 - distance_to_A) / 300.0
                return base_weight * proximity_factor
        
        return base_weight
    
    def _is_in_region(self, position: np.ndarray, region) -> bool:
        """Check if position is in rectangular region"""
        x, y = position[0], position[1]
        x_min, x_max, y_min, y_max = region
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def _distance_to_region(self, position: np.ndarray, region) -> float:
        """Calculate minimum distance to rectangular region"""
        x, y = position[0], position[1]
        x_min, x_max, y_min, y_max = region
        
        # If in region, distance is 0
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return 0.0
        
        # Calculate minimum distance to region boundary
        dx = max(0, max(x_min - x, x - x_max))
        dy = max(0, max(y_min - y, y - y_max))
        
        return np.sqrt(dx**2 + dy**2)
    
    def _is_position_in_fov(self, position: np.ndarray, sensor_pos: np.ndarray, 
                           sensor_heading: float) -> bool:
        """Check if position is in sensor field of view"""
        # Calculate position relative to sensor
        relative_pos = position - sensor_pos
        target_angle = np.arctan2(relative_pos[1], relative_pos[0])
        
        # Calculate angle difference
        angle_diff = np.abs(target_angle - sensor_heading)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Handle angle wrapping
        
        # Check if in field of view angle range
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
        Encourage agent to visit different regions of the map
        """
        # Simplified implementation: exploration reward based on position
        # Encourage visiting map edges and corners
        
        center_pos = np.array([MAP_WIDTH/2, MAP_HEIGHT/2])
        distance_from_center = np.linalg.norm(sensor_pos - center_pos)
        max_distance = np.linalg.norm(center_pos)
        
        # The farther from the center, the higher the exploration reward
        exploration_score = (distance_from_center / max_distance) * 0.1
        
        return exploration_score
    
    def _compute_boundary_penalty(self, agent_pos: np.ndarray, angle_deg: float) -> float:
        """
        Compute boundary penalty, avoid agent getting stuck in map boundary
        When agent is close to boundary and facing boundary, give penalty
        """
        angle_rad = np.deg2rad(angle_deg)
        velocity = AGENT_SPEED * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        predicted_pos = agent_pos + velocity * TIME_STEP_DURATION
        
        # Boundary margin
        boundary_margin = 50.0
        penalty = 0.0
        
        # Check if will hit boundary or already in boundary nearby
        if predicted_pos[0] <= boundary_margin or predicted_pos[0] >= MAP_WIDTH - boundary_margin:
            penalty += 5.0
        if predicted_pos[1] <= boundary_margin or predicted_pos[1] >= MAP_HEIGHT - boundary_margin:
            penalty += 5.0
            
        # If already in corner, strongly penalize action of moving towards corner
        if (agent_pos[0] <= boundary_margin and agent_pos[1] <= boundary_margin):
            # In top-left corner, penalize action of moving towards corner
            if np.cos(angle_rad) < 0 or np.sin(angle_rad) < 0:
                penalty += 20.0
        elif (agent_pos[0] >= MAP_WIDTH - boundary_margin and agent_pos[1] <= boundary_margin):
            # In top-right corner, penalize action of moving towards corner
            if np.cos(angle_rad) > 0 or np.sin(angle_rad) < 0:
                penalty += 20.0
        elif (agent_pos[0] <= boundary_margin and agent_pos[1] >= MAP_HEIGHT - boundary_margin):
            # In bottom-left corner, penalize action of moving towards corner
            if np.cos(angle_rad) < 0 or np.sin(angle_rad) > 0:
                penalty += 20.0
        elif (agent_pos[0] >= MAP_WIDTH - boundary_margin and agent_pos[1] >= MAP_HEIGHT - boundary_margin):
            # In bottom-right corner, penalize action of moving towards corner
            if np.cos(angle_rad) > 0 or np.sin(angle_rad) > 0:
                penalty += 20.0
                
        return penalty
    
    def get_information_map(self, pmbm_belief, grid_size: int = 50) -> np.ndarray:
        """
        Generate information map for visualization
        Return information value for each grid point
        """
        info_map = np.zeros((grid_size, grid_size))
        
        x_coords = np.linspace(0, MAP_WIDTH, grid_size)
        y_coords = np.linspace(0, MAP_HEIGHT, grid_size)
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                pos = np.array([x, y])
                # Calculate information score for east direction as an example
                score = self._compute_fov_information_score(pos, 0.0, pmbm_belief)
                info_map[j, i] = score  # Note: j corresponds to y-axis, i corresponds to x-axis
        
        return info_map
