"""
Simplified LTL-MCTS Planner for Scenario 2: Patrol and Respond
Focused implementation for G(ap_exist_A => F(ap_loc_any_A)) with specialized rewards
"""

import math
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.planners.base_planner import BasePlanner
from src.utils.data_structures import MCTSNode, LTLState, PMBMBelief, BernoulliComponent
from src.planners.scenario2_reward_functions import compute_scenario2_reward, get_scenario2_rollout_reward
from configs.config import *

class LTLMCTSPlanner(BasePlanner):
    """LTL-MCTS Planner for Patrol and Respond Task"""
    
    def __init__(self, name: str = "LTL_MCTS"):
        super().__init__(name)
        self.reset()
    
    def reset(self):
        """Reset planner state"""
        super().reset()
        self.mcts_root = None
        self.stats = {
            'total_planning_time': 0.0,
            'planning_calls': 0,
            'average_planning_time': 0.0
        }
        
    def plan_action(self, world_state, pmbm_belief: PMBMBelief, ltl_state: LTLState) -> int:
        """Plan action using LTL-MCTS"""
        start_time = time.time()
        
        # Create MCTS root node
        state_summary = self._create_belief_summary(pmbm_belief, ltl_state)
        self.mcts_root = MCTSNode(
            state_summary=state_summary,
            ltl_state=ltl_state,
            parent=None,
            action=None
        )
        
        # Run MCTS simulations
        for _ in range(MCTS_NUM_SIMULATIONS):
            self._mcts_simulation(self.mcts_root, world_state, pmbm_belief, ltl_state, 0)
        
        # Select best action
        best_action = self._select_best_action(self.mcts_root)
        
        # Update statistics
        planning_time = time.time() - start_time
        self.stats['total_planning_time'] += planning_time
        self.stats['planning_calls'] += 1
        self.stats['average_planning_time'] = self.stats['total_planning_time'] / self.stats['planning_calls']
        
        return best_action
    
    def _create_state_representation(self, world_state, pmbm_belief: PMBMBelief, ltl_state: LTLState) -> str:
        """Create enhanced state representation for MCTS (based on scenario 1 improvements)"""
        agent_pos = world_state.agent_state.position
        
        # Enhanced position discretization
        pos_str = f"pos_{int(agent_pos[0]/30)}_{int(agent_pos[1]/30)}"
        
        # Enhanced LTL state representation
        ap_exist_A = ltl_state.memory_state.get('ap_exist_A', False)
        ap_loc_any_A = ltl_state.memory_state.get('ap_loc_any_A', False)
        ltl_str = f"ltl_{ap_exist_A}_{ap_loc_any_A}"
        
        # Add patrol duty urgency (based on how long duty has been active)
        patrol_duty_active = ap_exist_A and not ap_loc_any_A
        urgency_str = f"urgent_{patrol_duty_active}"
        
        # Add region context
        in_region_A = self._is_in_region(agent_pos, SCENARIO_2_REGION_A)
        in_region_B = self._is_in_region(agent_pos, SCENARIO_2_REGION_B)
        region_str = f"reg_A{in_region_A}_B{in_region_B}"
        
        return f"{pos_str}_{ltl_str}_{urgency_str}_{region_str}"
    
    def _create_belief_summary(self, pmbm_belief: PMBMBelief, ltl_state: LTLState) -> np.ndarray:
        """Create LTL-guided belief summary (based on scenario 1 improvements)"""
        # Extract atomic proposition probabilities from LTL state
        ap_exist_A = 1.0 if ltl_state.memory_state.get('ap_exist_A', False) else 0.0
        ap_loc_any_A = 1.0 if ltl_state.memory_state.get('ap_loc_any_A', False) else 0.0
        
        # Add urgency indicator
        patrol_duty_active = ap_exist_A > 0.5 and ap_loc_any_A < 0.5
        urgency = 1.0 if patrol_duty_active else 0.0
        
        # Add belief quality indicators
        num_hypotheses = len(pmbm_belief.global_hypotheses) if pmbm_belief.global_hypotheses else 0
        belief_complexity = min(num_hypotheses / 10.0, 1.0)  # Normalized complexity
        
        # Create compact belief summary
        summary = np.array([
            ap_exist_A,
            ap_loc_any_A, 
            urgency,
            belief_complexity
        ])
        
        return summary
    
    def _mcts_simulation(self, node: MCTSNode, world_state, pmbm_belief: PMBMBelief, 
                        ltl_state: LTLState, depth: int) -> float:
        """Run one MCTS simulation"""
        if depth >= MCTS_MAX_DEPTH:
            return 0.0
        
        # Selection phase
        if node.is_fully_expanded(len(AGENT_CONTROL_ANGLES)) and node.children:
            # Select child using UCB1
            selected_child = self._ucb_select(node)
            return self._mcts_simulation(selected_child, world_state, pmbm_belief, ltl_state, depth + 1)
        
        # Expansion phase
        if not node.is_fully_expanded(len(AGENT_CONTROL_ANGLES)):
            action = self._get_next_untried_action(node)
            # Create child node with proper initialization
            child_state_summary = self._create_belief_summary(pmbm_belief, ltl_state)
            child_node = MCTSNode(
                state_summary=child_state_summary,
                ltl_state=ltl_state,
                parent=node,
                action=action
            )
            node.children[action] = child_node
            
            # Compute immediate reward for this action + rollout
            # Simulate the action effect first
            simulated_world_state = self._simulate_action_effect(world_state, action)
            info = self._create_mock_info(simulated_world_state, pmbm_belief, ltl_state)
            immediate_reward = compute_scenario2_reward(simulated_world_state, pmbm_belief, ltl_state, action, info)
            rollout_reward = self._simulate_rollout(child_node, simulated_world_state, pmbm_belief, ltl_state, depth + 1)
            reward = immediate_reward + rollout_reward
            
            # Backpropagation
            self._backpropagate(child_node, reward)
            return reward
        
        # If leaf node, simulate
        reward = self._simulate_rollout(node, world_state, pmbm_belief, ltl_state, depth)
        self._backpropagate(node, reward)
        return reward
    
    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """Select child using UCB1 formula"""
        best_child = None
        best_value = float('-inf')
        
        for child in node.children.values():
            if child.visits == 0:
                return child
            
            exploitation = child.total_reward / child.visits
            exploration = MCTS_EXPLORATION_CONSTANT * math.sqrt(
                math.log(node.visits) / child.visits
            )
            ucb_value = exploitation + exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        
        return best_child
    
    def _get_next_untried_action(self, node: MCTSNode) -> int:
        """Get next untried action for expansion"""
        tried_actions = set(node.children.keys())
        for action in range(len(AGENT_CONTROL_ANGLES)):
            if action not in tried_actions:
                return action
        return 0  # Fallback
    
    def _simulate_rollout(self, node: MCTSNode, world_state, pmbm_belief: PMBMBelief, 
                         ltl_state: LTLState, depth: int) -> float:
        """Enhanced rollout simulation (based on scenario 1 improvements)"""
        if depth >= MCTS_MAX_DEPTH:
            return 0.0
        
        # Enhanced rollout policy: task-aware actions
        total_reward = 0.0
        discount = 1.0
        
        # Simulate multiple steps with enhanced task-aware rollout
        for step in range(min(3, MCTS_MAX_DEPTH - depth)):  # Much shorter rollout for efficiency
            # Intelligent action selection based on current state
            action = self._select_rollout_action(world_state, ltl_state, step)
            
            # Update world state first, then compute reward
            world_state = self._simulate_action_effect(world_state, action)
            info = self._create_mock_info(world_state, pmbm_belief, ltl_state)
            reward = get_scenario2_rollout_reward(world_state, pmbm_belief, ltl_state, action, info)
            total_reward += discount * reward
            discount *= MCTS_DISCOUNT_FACTOR
        
        return total_reward
    
    def _select_rollout_action(self, world_state, ltl_state: LTLState, step: int) -> int:
        """Select intelligent action for rollout (task-aware policy)"""
        agent_pos = world_state.agent_state.position
        
        # Check if patrol duty is active (targets exist but not localized)
        ap_exist_A = ltl_state.memory_state.get('ap_exist_A', False)
        ap_loc_any_A = ltl_state.memory_state.get('ap_loc_any_A', False)
        patrol_duty_active = ap_exist_A and not ap_loc_any_A
        
        if patrol_duty_active:
            # During patrol duty: head towards region A with high certainty
            region_center = np.array([
                (SCENARIO_2_REGION_A[0] + SCENARIO_2_REGION_A[1]) / 2,  # x center
                (SCENARIO_2_REGION_A[2] + SCENARIO_2_REGION_A[3]) / 2   # y center
            ])
            direction = region_center - agent_pos
            target_angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
            target_angle = (target_angle + 360) % 360
            
            # Find closest control angle
            best_action = 0
            min_diff = float('inf')
            for i, angle in enumerate(AGENT_CONTROL_ANGLES):
                diff = min(abs(angle - target_angle), 360 - abs(angle - target_angle))
                if diff < min_diff:
                    min_diff = diff
                    best_action = i
            
            # 95% task-directed, 5% random for some exploration
            if np.random.random() < 0.95:
                return best_action
            else:
                return np.random.randint(len(AGENT_CONTROL_ANGLES))
        else:
            # No active duty: enhanced proactive patrol strategy
            distance_to_A = self._distance_to_region(agent_pos, SCENARIO_2_REGION_A)
            
            if self._is_in_region(agent_pos, SCENARIO_2_REGION_B):
                # If in region B, strongly move towards region A (100% directed)
                region_A_center = np.array([
                    (SCENARIO_2_REGION_A[0] + SCENARIO_2_REGION_A[1]) / 2,
                    (SCENARIO_2_REGION_A[2] + SCENARIO_2_REGION_A[3]) / 2
                ])
                direction = region_A_center - agent_pos
                target_angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
                target_angle = (target_angle + 360) % 360
                
                best_action = 0
                min_diff = float('inf')
                for i, angle in enumerate(AGENT_CONTROL_ANGLES):
                    diff = min(abs(angle - target_angle), 360 - abs(angle - target_angle))
                    if diff < min_diff:
                        min_diff = diff
                        best_action = i
                return best_action
            elif distance_to_A > 200.0:
                # If far from region A, move towards it (90% directed)
                region_A_center = np.array([
                    (SCENARIO_2_REGION_A[0] + SCENARIO_2_REGION_A[1]) / 2,
                    (SCENARIO_2_REGION_A[2] + SCENARIO_2_REGION_A[3]) / 2
                ])
                direction = region_A_center - agent_pos
                target_angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
                target_angle = (target_angle + 360) % 360
                
                best_action = 0
                min_diff = float('inf')
                for i, angle in enumerate(AGENT_CONTROL_ANGLES):
                    diff = min(abs(angle - target_angle), 360 - abs(angle - target_angle))
                    if diff < min_diff:
                        min_diff = diff
                        best_action = i
                
                if np.random.random() < 0.9:
                    return best_action
                else:
                    return np.random.randint(len(AGENT_CONTROL_ANGLES))
            elif self._is_in_region(agent_pos, SCENARIO_2_REGION_A):
                # If already in region A, patrol within it (50% directed, 50% random for exploration)
                if np.random.random() < 0.5:
                    # Move towards center of region A
                    region_A_center = np.array([
                        (SCENARIO_2_REGION_A[0] + SCENARIO_2_REGION_A[1]) / 2,
                        (SCENARIO_2_REGION_A[2] + SCENARIO_2_REGION_A[3]) / 2
                    ])
                    direction = region_A_center - agent_pos
                    target_angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
                    target_angle = (target_angle + 360) % 360
                    
                    best_action = 0
                    min_diff = float('inf')
                    for i, angle in enumerate(AGENT_CONTROL_ANGLES):
                        diff = min(abs(angle - target_angle), 360 - abs(angle - target_angle))
                        if diff < min_diff:
                            min_diff = diff
                            best_action = i
                    return best_action
                else:
                    # Random exploration within region A
                    return np.random.randint(len(AGENT_CONTROL_ANGLES))
            else:
                # Near region A but not inside: move towards it (80% directed)
                region_A_center = np.array([
                    (SCENARIO_2_REGION_A[0] + SCENARIO_2_REGION_A[1]) / 2,
                    (SCENARIO_2_REGION_A[2] + SCENARIO_2_REGION_A[3]) / 2
                ])
                direction = region_A_center - agent_pos
                target_angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
                target_angle = (target_angle + 360) % 360
                
                best_action = 0
                min_diff = float('inf')
                for i, angle in enumerate(AGENT_CONTROL_ANGLES):
                    diff = min(abs(angle - target_angle), 360 - abs(angle - target_angle))
                    if diff < min_diff:
                        min_diff = diff
                        best_action = i
                
                if np.random.random() < 0.8:
                    return best_action
                else:
                    return np.random.randint(len(AGENT_CONTROL_ANGLES))
    
    def _simulate_action_effect(self, world_state, action: int):
        """Simulate the effect of an action on world state (simplified)"""
        # Create a copy of world state for simulation
        new_world_state = type(world_state)(time_step=world_state.time_step + 1)
        
        # Update agent position based on action
        angle_deg = AGENT_CONTROL_ANGLES[action]
        angle_rad = np.radians(angle_deg)
        
        # Simple movement simulation
        movement = np.array([
            AGENT_SPEED * TIME_STEP_DURATION * np.cos(angle_rad),
            AGENT_SPEED * TIME_STEP_DURATION * np.sin(angle_rad)
        ])
        
        new_position = world_state.agent_state.position + movement
        
        # Boundary constraints
        new_position[0] = np.clip(new_position[0], 0, MAP_WIDTH)
        new_position[1] = np.clip(new_position[1], 0, MAP_HEIGHT)
        
        # Update agent state
        from src.utils.data_structures import AgentState
        new_world_state.agent_state = AgentState(
            position=new_position,
            velocity=movement / TIME_STEP_DURATION,
            heading=angle_deg
        )
        
        # Keep targets the same (simplified)
        new_world_state.targets = world_state.targets.copy() if hasattr(world_state, 'targets') else []
        
        return new_world_state
    
    def _simulate_ltl_update(self, world_state, ltl_state: LTLState) -> LTLState:
        """Simulate simplified LTL state update based on agent position"""
        # Create a copy of LTL state
        from src.utils.data_structures import LTLState
        new_ltl_state = LTLState(
            automaton_state=ltl_state.automaton_state,
            memory_state=ltl_state.memory_state.copy()
        )
        
        agent_pos = world_state.agent_state.position
        
        # Simplified AP probability estimation based on position
        # If agent is in region A, increase chance of localization
        if self._is_in_region(agent_pos, SCENARIO_2_REGION_A):
            # If there was an existing duty, there's a chance to complete it
            if new_ltl_state.memory_state.get('ap_exist_A', False):
                # Simulate successful localization with some probability
                if np.random.random() < 0.3:  # 30% chance per step in region A
                    new_ltl_state.memory_state['ap_loc_any_A'] = True
        else:
            # If outside region A, lose localization
            new_ltl_state.memory_state['ap_loc_any_A'] = False
        
        return new_ltl_state
    
    def _compute_rollout_reward(self, world_state, pmbm_belief: PMBMBelief, 
                               ltl_state: LTLState, action: int) -> float:
        """Compute simplified but effective rollout reward"""
        agent_pos = world_state.agent_state.position
        
        # Simulate action effect
        angle_deg = AGENT_CONTROL_ANGLES[action]
        angle_rad = np.deg2rad(angle_deg)
        velocity = AGENT_SPEED * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        next_pos = agent_pos + velocity * TIME_STEP_DURATION
        
        # Boundary constraints
        next_pos[0] = np.clip(next_pos[0], 0, MAP_WIDTH)
        next_pos[1] = np.clip(next_pos[1], 0, MAP_HEIGHT)
        
        # Check LTL state
        ap_exist_A = ltl_state.memory_state.get('ap_exist_A', False)
        ap_loc_any_A = ltl_state.memory_state.get('ap_loc_any_A', False)
        patrol_duty_active = ap_exist_A and not ap_loc_any_A
        
        reward = 0.0
        
        if patrol_duty_active:
            # High priority: complete patrol duty
            if self._is_in_region(next_pos, SCENARIO_2_REGION_A):
                reward += 25.0  # Strong reward for being in region A during duty
            else:
                # Reward for moving towards region A
                distance_to_A = self._distance_to_region(next_pos, SCENARIO_2_REGION_A)
                reward += max(0, 15.0 - distance_to_A * 0.03)
            
            # Strong penalty for temptation (moving to region B during duty)
            if self._is_in_region(next_pos, SCENARIO_2_REGION_B):
                reward -= 20.0
        else:
            # No active duty: moderate reward for staying ready
            if self._is_in_region(next_pos, SCENARIO_2_REGION_A):
                reward += 8.0  # Moderate reward for staying ready in region A
            else:
                # Small reward for moving towards region A
                distance_to_A = self._distance_to_region(next_pos, SCENARIO_2_REGION_A)
                reward += max(0, 5.0 - distance_to_A * 0.01)
        
        # Small step penalty to encourage efficiency
        reward -= 0.5
        
        return reward
    
    def _compute_immediate_reward(self, world_state, pmbm_belief: PMBMBelief, 
                                 ltl_state: LTLState, action: int) -> float:
        """Compute immediate reward similar to LTL_Myopic but stronger"""
        agent_pos = world_state.agent_state.position
        
        # Simulate action effect
        angle_deg = AGENT_CONTROL_ANGLES[action]
        angle_rad = np.deg2rad(angle_deg)
        velocity = AGENT_SPEED * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        next_pos = agent_pos + velocity * TIME_STEP_DURATION
        
        # Boundary constraints
        next_pos[0] = np.clip(next_pos[0], 0, MAP_WIDTH)
        next_pos[1] = np.clip(next_pos[1], 0, MAP_HEIGHT)
        
        # Check LTL state
        ap_exist_A = ltl_state.memory_state.get('ap_exist_A', False)
        ap_loc_any_A = ltl_state.memory_state.get('ap_loc_any_A', False)
        patrol_duty_active = ap_exist_A and not ap_loc_any_A
        
        reward = 0.0
        
        if patrol_duty_active:
            # High priority: complete patrol duty (similar to LTL_Myopic)
            if self._is_in_region(next_pos, SCENARIO_2_REGION_A):
                reward += 30.0  # Strong reward for being in region A during duty
            else:
                # Strong reward for moving towards region A
                distance_to_A = self._distance_to_region(next_pos, SCENARIO_2_REGION_A)
                reward += max(0, 25.0 - distance_to_A * 0.05)
            
            # Strong penalty for temptation (moving to region B during duty)
            if self._is_in_region(next_pos, SCENARIO_2_REGION_B):
                reward -= 30.0
        else:
            # No active duty: moderate reward for staying ready
            if self._is_in_region(next_pos, SCENARIO_2_REGION_A):
                reward += 10.0  # Moderate reward for staying ready in region A
            else:
                # Small reward for moving towards region A
                distance_to_A = self._distance_to_region(next_pos, SCENARIO_2_REGION_A)
                reward += max(0, 8.0 - distance_to_A * 0.02)
        
        return reward
    
    def _distance_to_region(self, position: np.ndarray, region: List[float]) -> float:
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
    
    def _create_mock_info(self, world_state, pmbm_belief: PMBMBelief, ltl_state: LTLState) -> Dict[str, Any]:
        """Create mock info dictionary for reward computation with enhanced clutter filtering"""
        agent_pos = world_state.agent_state.position
        
        # Use actual LTL memory state as ground truth for duty status
        ap_exist_A_memory = ltl_state.memory_state.get('ap_exist_A', False)
        ap_loc_any_A_memory = ltl_state.memory_state.get('ap_loc_any_A', False)
        
        # Enhanced AP estimation with clutter filtering for LTL_MCTS
        in_region_A = self._is_in_region(agent_pos, SCENARIO_2_REGION_A)
        
        # For LTL_MCTS, use more conservative estimation to avoid clutter-induced false positives
        ap_exist_A = self._compute_filtered_exist_prob(pmbm_belief, SCENARIO_2_REGION_A)
        ap_loc_any_A = self._compute_filtered_loc_prob(pmbm_belief, SCENARIO_2_REGION_A, agent_pos)
        
        # Fallback to memory-based estimation if filtered probabilities are too low
        if ap_exist_A < 0.1 and ap_exist_A_memory:
            ap_exist_A = 0.5  # Moderate confidence based on memory
        
        if ap_loc_any_A < 0.1 and in_region_A and ap_exist_A_memory:
            ap_loc_any_A = 0.7  # High confidence when in region A with memory
        
        # Duty status: use memory state primarily
        patrol_duty_active = ap_exist_A_memory and not ap_loc_any_A_memory
        
        return {
            'ap_exist_A': ap_exist_A,
            'ap_loc_any_A': ap_loc_any_A,
            'patrol_duty_active': patrol_duty_active,
            'duty_violation': False,
            'task_completed': ap_loc_any_A > LTL_LOC_THRESHOLD if patrol_duty_active else False
        }
    
    def _compute_filtered_exist_prob(self, belief: PMBMBelief, region: List[float]) -> float:
        """Compute existence probability with enhanced clutter filtering for LTL_MCTS"""
        map_hyp = belief.get_map_hypothesis()
        if map_hyp is None:
            return 0.0
        
        p_none = 1.0
        
        # Use higher threshold for LTL_MCTS to avoid clutter
        for comp in map_hyp.bernoulli_components:
            if comp.existence_prob > 0.75:  # High threshold for LTL_MCTS
                P_i_R = self._compute_gaussian_region_prob(comp, region)
                if P_i_R > 1e-8:
                    p_none *= (1.0 - float(comp.existence_prob) * P_i_R)
        
        # Reduced PPP contribution for LTL_MCTS
        lam0 = float(PPP_DENSITY) * 0.1  # Further reduce PPP influence
        area_R = float((region[1] - region[0]) * (region[3] - region[2]))
        Lambda_R = max(0.0, lam0 * area_R)
        
        p_exist = 1.0 - np.exp(-Lambda_R) * p_none
        return float(np.clip(p_exist, 0.0, 1.0))
    
    def _compute_filtered_loc_prob(self, belief: PMBMBelief, region: List[float], agent_pos: np.ndarray) -> float:
        """Compute localization probability with enhanced clutter filtering for LTL_MCTS"""
        map_hyp = belief.get_map_hypothesis()
        if map_hyp is None:
            return 0.0
        
        max_prob = 0.0
        
        # Use higher threshold for LTL_MCTS
        for comp in map_hyp.bernoulli_components:
            if comp.existence_prob > 0.75:  # High threshold for LTL_MCTS
                P_i_R = self._compute_gaussian_region_prob(comp, region)
                localization_prob = comp.existence_prob * P_i_R
                max_prob = max(max_prob, localization_prob)
        
        return float(np.clip(max_prob, 0.0, 1.0))
    
    def _compute_gaussian_region_prob(self, bernoulli_comp, region: List[float]) -> float:
        """Calculate probability integral P_{i,R} of Gaussian distribution in rectangular region"""
        try:
            from scipy.stats import norm
            
            # Get mean and covariance
            mean = bernoulli_comp.get_position_mean()
            cov = bernoulli_comp.get_position_cov()
            
            # Extract marginal distribution parameters for X and Y
            mu_x, mu_y = mean[0], mean[1]
            sigma_x = np.sqrt(cov[0, 0])
            sigma_y = np.sqrt(cov[1, 1])
            
            # Calculate probability integral in X direction
            prob_x = norm.cdf(region[1], mu_x, sigma_x) - norm.cdf(region[0], mu_x, sigma_x)
            
            # Calculate probability integral in Y direction  
            prob_y = norm.cdf(region[3], mu_y, sigma_y) - norm.cdf(region[2], mu_y, sigma_y)
            
            # Assuming X and Y independent, total probability integral is the product
            prob_integral = prob_x * prob_y
            
            return min(max(prob_integral, 0.0), 1.0)  # Limit to [0,1] range
            
        except Exception:
            # Simplified calculation: check if mean is in region
            mean = bernoulli_comp.get_position_mean()
            in_region = (region[0] <= mean[0] <= region[1] and 
                        region[2] <= mean[1] <= region[3])
            return 1.0 if in_region else 0.0
    
    def _is_in_region(self, position: np.ndarray, region: List[float]) -> bool:
        """Check if position is inside rectangular region"""
        x, y = position[0], position[1]
        x_min, x_max, y_min, y_max = region
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward through the tree"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def _select_best_action(self, root: MCTSNode) -> int:
        """Select best action from root node"""
        if not root.children:
            return 0  # Default action
        
        # Select action with highest average reward (more robust than visit count)
        best_action = max(root.children.keys(), 
                         key=lambda a: (root.children[a].total_reward / root.children[a].visits) 
                                     if root.children[a].visits > 0 else float('-inf'))
        return best_action
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics"""
        return {
            'name': self.name,
            'total_planning_calls': self.stats['planning_calls'],
            'average_planning_time': self.stats['average_planning_time'],
            'total_planning_time': self.stats['total_planning_time']
        }