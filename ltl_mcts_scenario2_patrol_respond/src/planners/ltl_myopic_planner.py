"""
LTL myopic planner implementation
Understand LTL task, but only make greedy decision for single step ahead
It is easy to be tempted by short-term gains, which reflects the limitations of the single-step greedy algorithm.
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
    MAP_WIDTH, MAP_HEIGHT, SCENARIO_2_REGION_A, SCENARIO_2_REGION_B
)

class LTLMyopicPlanner(BasePlanner):
    """ LTL myopic planner"""
    
    def __init__(self, config=None):
        super().__init__("LTL_Myopic", config)
        
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        Plan next action: select the action that is most beneficial for LTL task
        Use the same semantic-probability bridge as LTL-MCTS, but only make greedy choice
        """
        start_time = time.time()
        
        agent_pos = world_state.agent_state.position
        best_action = 0
        best_score = -float('inf')
        
        # Traverse all possible control angles
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
            agent_pos: agent current position
            angle_deg: control angle (degree)
            pmbm_belief: PMBM belief state
            ltl_state: LTL state
            
        Returns:
            float: LTL reward score
        """
        angle_rad = np.deg2rad(angle_deg)
        
        # Predict agent's position after moving
        velocity = AGENT_SPEED * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        predicted_pos = agent_pos + velocity * TIME_STEP_DURATION
        
        # Boundary check
        predicted_pos[0] = np.clip(predicted_pos[0], 0, MAP_WIDTH)
        predicted_pos[1] = np.clip(predicted_pos[1], 0, MAP_HEIGHT)
        
        # Calculate LTL reward
        ltl_reward = self._compute_ltl_reward(predicted_pos, angle_rad, pmbm_belief, ltl_state)
        
        # Add direction balance mechanism: prevent meaningless upward bias
        direction_penalty = self._compute_direction_penalty(angle_deg, agent_pos)
        
        return ltl_reward + direction_penalty
    
    def _compute_ltl_reward(self, sensor_pos: np.ndarray, sensor_heading: float,
                          pmbm_belief, ltl_state) -> float:
        """
        Calculate LTL task-related reward - simplified version, reflect short-sighted characteristic
        LTL_Myopic understands basic tasks but is easily tempted by short-term gains        """
        total_reward = 0.0
        
        # Get LTL state
        ap_exist_A = ltl_state.memory_state.get('ap_exist_A', False)
        ap_loc_any_A = ltl_state.memory_state.get('ap_loc_any_A', False)
        patrol_duty_active = ap_exist_A and not ap_loc_any_A
        
        # Basic task reward: weak task understanding, completely unable to resist temptation
        if patrol_duty_active:
            # When task is activated, weak bias towards region A (completely unable to resist temptation)
            if self._is_in_region(sensor_pos, SCENARIO_2_REGION_A):
                total_reward += 15.0  # Reward in region A (greatly削弱)
            else:
                # Reward for moving towards region A
                region_A_center = np.array([150.0, 500.0])  # Region A center
                to_region_A = region_A_center - sensor_pos
                
                if np.linalg.norm(to_region_A) > 10.0:
                    to_region_normalized = to_region_A / np.linalg.norm(to_region_A)
                    heading_vec = np.array([np.cos(sensor_heading), np.sin(sensor_heading)])
                    alignment = np.dot(heading_vec, to_region_normalized)
                    
                    # When task is activated, weak reward for moving towards region A (completely dominated)
                    if alignment > 0:
                        total_reward += alignment * 8.0  # Direction reward (greatly削弱)
                    else:
                        total_reward -= abs(alignment) * 0.5  # Almost no penalty
            
            # Almost no penalty in region B, completely unable to resist temptation
            if self._is_in_region(sensor_pos, SCENARIO_2_REGION_B):
                total_reward -= 1.0  # Almost no penalty, completely unable to resist temptation
        else:
            # Short-sighted exploration preference when there is no task:Easily attracted to "interesting" areas
            if self._is_in_region(sensor_pos, SCENARIO_2_REGION_A):
                # When in region A, give a small base reward, but easily attracted to other areas
                total_reward += 1.0  # Small base region A preference
                
                # Explicitly penalize upward movement
                angle_deg = np.rad2deg(sensor_heading)
                if 60 <= angle_deg <= 120:  # Towards north direction
                    height_factor = sensor_pos[1] / MAP_HEIGHT
                    if height_factor > 0.6:
                        total_reward -= 15.0
                    else:
                        total_reward -= 8.0
                        
                # Other boundary penalties
                if 240 <= angle_deg <= 300 and sensor_pos[1] < 200:
                    total_reward -= 8.0
                if (330 <= angle_deg <= 360 or 0 <= angle_deg <= 30) and sensor_pos[0] > MAP_WIDTH - 200:
                    total_reward -= 8.0
                if 150 <= angle_deg <= 210 and sensor_pos[0] < 200:
                    total_reward -= 8.0
            else:
                # Short-sighted algorithm's key flaw: Strong curiosity for "new" areas
                distance_to_B = self._distance_to_region(sensor_pos, SCENARIO_2_REGION_B)
                distance_to_A = self._distance_to_region(sensor_pos, SCENARIO_2_REGION_A)
                
                # Short-sighted algorithm's balance: basic task awareness but easily tempted
                
                # Strong basic attraction to region A (ensure task activation)
                region_A_center = np.array([150.0, 500.0])
                to_region_A = region_A_center - sensor_pos
                if np.linalg.norm(to_region_A) > 10.0:
                    to_region_normalized = to_region_A / np.linalg.norm(to_region_A)
                    heading_vec = np.array([np.cos(sensor_heading), np.sin(sensor_heading)])
                    alignment = np.dot(heading_vec, to_region_normalized)
                    if alignment > 0:
                        distance_factor = max(0, 1.0 - distance_to_A / 400.0)
                        total_reward += alignment * 15.0 * distance_factor  # Ensure reaching region A
        
        # Short-sighted algorithm's fatal flaw: "attracted" to B region at all times
        # 1. If see target, chase with extreme greed (short-sighted fatal flaw)
        # 2. If not see target, go to B region "explore" (wrong curiosity)
        
        has_visible_targets = False
        if pmbm_belief is not None:
            try:
                map_hyp = pmbm_belief.get_map_hypothesis()
                if map_hyp is not None:
                    for comp in map_hyp.bernoulli_components:
                        if comp.existence_prob > 0.1:  # Easily attracted to any target (short-sighted fatal flaw)
                            has_visible_targets = True
                            comp_pos = comp.get_position_mean()
                            comp_distance = np.linalg.norm(sensor_pos - comp_pos)
                            
                            # Strong attraction to all targets (short-sighted fatal flaw)
                            target_attraction = max(0, 500.0 - comp_distance / 1.0) * comp.existence_prob
                            
                            # B region target: overwhelming temptation (especially more dangerous when task is activated) (short-sighted fatal flaw)
                            if self._is_in_region(comp_pos, SCENARIO_2_REGION_B):
                                # Basic attraction
                                base_multiplier = 15.0
                                
                                # If task is activated, B region temptation is more dangerous (short-sighted fatal flaw)
                                if patrol_duty_active:
                                    base_multiplier = 50.0  # B region temptation greatly enhanced when task is activated
                                
                                target_attraction *= base_multiplier
                                
                                # Overwhelming direction reward
                                to_target = comp_pos - sensor_pos
                                if np.linalg.norm(to_target) > 10.0:
                                    to_target_normalized = to_target / np.linalg.norm(to_target)
                                    heading_vec = np.array([np.cos(sensor_heading), np.sin(sensor_heading)])
                                    alignment = np.dot(heading_vec, to_target_normalized)
                                    if alignment > 0:
                                        direction_reward = 300.0
                                        if patrol_duty_active:
                                            direction_reward = 800.0  # Direction reward greatly enhanced when task is activated
                                        total_reward += alignment * direction_reward
                            elif self._is_in_region(comp_pos, SCENARIO_2_REGION_A):
                                # A region target: strong attraction but less than B region
                                target_attraction *= 5.0
                                to_target = comp_pos - sensor_pos
                                if np.linalg.norm(to_target) > 10.0:
                                    to_target_normalized = to_target / np.linalg.norm(to_target)
                                    heading_vec = np.array([np.cos(sensor_heading), np.sin(sensor_heading)])
                                    alignment = np.dot(heading_vec, to_target_normalized)
                                    if alignment > 0:
                                        total_reward += alignment * 80.0
                            else:
                                target_attraction *= 3.0
                            
                            total_reward += target_attraction
            except:
                pass
        
        # Short-sighted algorithm's core flaw: wrong "patrol strategy"
        # Short-sighted algorithm thinks should "average time" in two regions (wrong intuition)
        if not has_visible_targets:
            distance_to_A = self._distance_to_region(sensor_pos, SCENARIO_2_REGION_A)
            distance_to_B = self._distance_to_region(sensor_pos, SCENARIO_2_REGION_B)
            
            # Short-sighted algorithm's wrong logic: think should "fairly" patrol two regions
            # This causes it to waste a lot of time in B region
            
            # If already spent some time in A region, want to go to B region "balance"
            if self._is_in_region(sensor_pos, SCENARIO_2_REGION_A):
                # Short-sighted "balance" idea: go to B region to see
                region_B_center = np.array([850.0, 500.0])
                to_B = region_B_center - sensor_pos
                if np.linalg.norm(to_B) > 10.0:
                    to_B_normalized = to_B / np.linalg.norm(to_B)
                    heading_vec = np.array([np.cos(sensor_heading), np.sin(sensor_heading)])
                    alignment = np.dot(heading_vec, to_B_normalized)
                    if alignment > 0:
                        total_reward += alignment * 25.0  # Moderate wrong "balance patrol" reward
            
            # Basic task understanding: should go to A region (but weakened by "balance" idea)
            elif distance_to_A < 400.0:
                region_A_center = np.array([150.0, 500.0])
                to_A = region_A_center - sensor_pos
                if np.linalg.norm(to_A) > 10.0:
                    to_A_normalized = to_A / np.linalg.norm(to_A)
                    heading_vec = np.array([np.cos(sensor_heading), np.sin(sensor_heading)])
                    alignment = np.dot(heading_vec, to_A_normalized)
                    if alignment > 0:
                        total_reward += alignment * 10.0  # Weakened A region exploration reward
            
            # Wrong "exploration" desire: want "fully understand" environment
            if distance_to_B < 500.0:
                exploration_reward = max(0, (500.0 - distance_to_B) / 500.0 * 15.0)
                total_reward += exploration_reward
                
                region_B_center = np.array([850.0, 500.0])
                to_B = region_B_center - sensor_pos
                if np.linalg.norm(to_B) > 10.0:
                    to_B_normalized = to_B / np.linalg.norm(to_B)
                    heading_vec = np.array([np.cos(sensor_heading), np.sin(sensor_heading)])
                    alignment = np.dot(heading_vec, to_B_normalized)
                    if alignment > 0:
                        total_reward += alignment * 18.0  # Moderate wrong exploration direction reward
        
        return total_reward
    
    def _is_in_region(self, position: np.ndarray, region: List[float]) -> bool:
        """Check if position is in rectangular region"""
        x, y = position[0], position[1]
        x_min, x_max, y_min, y_max = region
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def _distance_to_region(self, position: np.ndarray, region: List[float]) -> float:
        """Calculate the minimum distance from a point to a rectangular region"""
        x, y = position[0], position[1]
        x_min, x_max, y_min, y_max = region
        
        # If in region, distance is 0
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return 0.0
        
        # Calculate the minimum distance to the region boundary
        dx = max(0, max(x_min - x, x - x_max))
        dy = max(0, max(y_min - y, y - y_max))
        
        return np.sqrt(dx**2 + dy**2)
    
    
    def get_ltl_value_map(self, pmbm_belief, ltl_state, grid_size: int = 50) -> np.ndarray:
        """
        Generate LTL value map for visualization
        Return LTL reward value for each grid point
        """
        value_map = np.zeros((grid_size, grid_size))
        
        x_coords = np.linspace(0, MAP_WIDTH, grid_size)
        y_coords = np.linspace(0, MAP_HEIGHT, grid_size)
        
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                pos = np.array([x, y])
                # Calculate reward for multiple headings, take the maximum to get objective spatial attraction
                max_reward = float('-inf')
                test_headings = [0.0, np.pi/2, np.pi, 3*np.pi/2]  # East, North, West, South four directions
                for heading in test_headings:
                    reward = self._compute_ltl_reward(pos, heading, pmbm_belief, ltl_state)
                    max_reward = max(max_reward, reward)
                value_map[j, i] = max_reward  # Note: j corresponds to y axis, i corresponds to x axis
        
        return value_map
    
    
    def _compute_direction_penalty(self, angle_deg: float, agent_pos: np.ndarray) -> float:
        """
        Simple direction penalty, avoid meaningless boundary behavior
        """
        penalty = 0.0
        
        # Penalty for behavior too close to map boundary
        if agent_pos[1] > MAP_HEIGHT * 0.9 and 60 <= angle_deg <= 120:  # Close to upper boundary and up
            penalty -= 5.0
        if agent_pos[0] < 50.0 and 150 <= angle_deg <= 210:  # Close to left boundary and left
            penalty -= 5.0
        if agent_pos[0] > MAP_WIDTH - 50.0 and (330 <= angle_deg <= 360 or 0 <= angle_deg <= 30):  # Close to right boundary and right
            penalty -= 5.0
        if agent_pos[1] < 50.0 and 240 <= angle_deg <= 300:  # Close to lower boundary and down
            penalty -= 5.0
            
        return penalty
    
