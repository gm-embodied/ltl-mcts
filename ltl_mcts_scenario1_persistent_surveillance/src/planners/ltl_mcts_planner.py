"""
LTL-MCTS planner implementation - based on original paper verification project
Adapted for Scenario 1: Persistent surveillance task
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
from configs.config import *

@dataclass
class MCTSParams:
    """MCTS parameter configuration"""
    num_simulations: int = MCTS_NUM_SIMULATIONS
    max_depth: int = MCTS_MAX_DEPTH
    gamma: float = MCTS_DISCOUNT_FACTOR
    ucb_c: float = MCTS_EXPLORATION_CONSTANT
    progressive_widening_c: float = 1.5
    max_actions_per_node: int = 6

class LTLMCTSPlanner(BasePlanner):
    """LTL-MCTS planner - core contribution of the paper"""
    
    def __init__(self, config=None):
        super().__init__("LTL_MCTS", config)
        self.params = MCTSParams()
        if config:
            for key, value in config.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
        
        # cache recent belief summary for debugging
        self.last_belief_summary = None
        self.last_ltl_state = None
    
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        Plan next action using LTL-MCTS algorithm
        """
        start_time = time.time()
        
        try:
            # 1. Create LTL belief summary
            belief_summary = self._create_belief_summary(pmbm_belief, ltl_state, world_state)
            self.last_belief_summary = belief_summary
            self.last_ltl_state = ltl_state
            
            # 2. Run MCTS search
            best_action = self._mcts_search(world_state, pmbm_belief, ltl_state, belief_summary)
            
        except Exception as e:
            # Record error information, but do not interrupt experiment
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"LTL-MCTS planning failed: {str(e)}, using random action")
            
            # If error, use random action
            best_action = np.random.randint(0, len(AGENT_CONTROL_ANGLES))
        
        # 3. Record statistics information
        planning_time = time.time() - start_time
        self.planning_time_history.append(planning_time)
        self.action_history.append(best_action)
        
        return best_action
    
    def _create_belief_summary(self, pmbm_belief: PMBMBelief, ltl_state: LTLState, world_state=None) -> np.ndarray:
        """
        Create LTL belief summary b^φ - contains position and navigation information
        Summary vector: [P_exist(A), P_exist(B), ltl_state_A, ltl_state_B, urgency_A, urgency_B, agent_x, agent_y]
        """
        try:
            # Compute region existence probability
            p_exist_A = self._compute_region_exist_prob(pmbm_belief, REGION_A)
            p_exist_B = self._compute_region_exist_prob(pmbm_belief, REGION_B)
            
            # LTL state information
            ltl_state_A = 1.0 if ltl_state.memory_state.get('ap_exist_A', False) else 0.0
            ltl_state_B = 1.0 if ltl_state.memory_state.get('ap_exist_B', False) else 0.0
            
            # Compute urgency (consistent with LTL_Myopic)
            urgency_A = max(0, 0.5 - p_exist_A)  # 存在概率越低越紧急
            urgency_B = max(0, 0.5 - p_exist_B)
            
            # Agent position information (normalized to [0,1])
            if world_state is not None:
                agent_pos = world_state.agent_state.position
                agent_x = agent_pos[0] / MAP_WIDTH
                agent_y = agent_pos[1] / MAP_HEIGHT
            else:
                agent_x = 0.5
                agent_y = 0.5
            
            # Build belief summary vector
            summary = np.array([p_exist_A, p_exist_B, ltl_state_A, ltl_state_B, urgency_A, urgency_B, agent_x, agent_y])
            
            return summary
        except Exception as e:
            # If error, return default summary
            return np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    
    def _compute_region_exist_prob(self, pmbm_belief: PMBMBelief, region: List[float]) -> float:
        """
        Compute region existence probability - use exact Gaussian integral calculation
        P_exist(R) = 1 - [exp(-Lambda_R) * ∏(1 - r_i * P_{i,R})]
        """
        try:
            # Compute region area and PPP contribution
            x_min, x_max, y_min, y_max = region
            area_R = (x_max - x_min) * (y_max - y_min)
            lambda_R = PPP_DENSITY * area_R
            
            # Get MAP hypothesis
            map_hyp = pmbm_belief.get_map_hypothesis()
            if not map_hyp or not hasattr(map_hyp, 'bernoulli_components'):
                # Only PPP contribution
                return 1.0 - np.exp(-lambda_R)
            
            # Compute Bernoulli component contribution
            bernoulli_product = 1.0
            for comp in map_hyp.bernoulli_components:
                r_i = float(comp.existence_prob)
                # Use exact Gaussian integral calculation for P_{i,R}
                P_i_R = self._compute_gaussian_region_integral(comp, region)
                bernoulli_product *= (1.0 - r_i * P_i_R)
            
            # Compute final existence probability
            p_exist = 1.0 - (np.exp(-lambda_R) * bernoulli_product)
            return float(np.clip(p_exist, 0.0, 1.0))
        
        except Exception as e:
            return 0.0
    
    def _compute_gaussian_region_integral(self, bernoulli_comp, region: List[float]) -> float:
        """Compute exact probability integral of Gaussian distribution in rectangular region"""
        try:
            from scipy.stats import norm
            
            # Get mean and covariance
            mean = bernoulli_comp.get_position_mean()
            cov = bernoulli_comp.get_position_cov()
            
            # Extract marginal distribution parameters
            mu_x, mu_y = mean[0], mean[1]
            sigma_x = np.sqrt(max(cov[0, 0], 1e-6))  # Prevent numerical issues
            sigma_y = np.sqrt(max(cov[1, 1], 1e-6))
            
            # Compute exact probability integral
            prob_x = norm.cdf(region[1], mu_x, sigma_x) - norm.cdf(region[0], mu_x, sigma_x)
            prob_y = norm.cdf(region[3], mu_y, sigma_y) - norm.cdf(region[2], mu_y, sigma_y)
            
            prob_integral = prob_x * prob_y
            return float(np.clip(prob_integral, 0.0, 1.0))
            
        except Exception:
            # Fall back to simplified calculation
            mean = bernoulli_comp.get_position_mean()
            x_min, x_max, y_min, y_max = region
            if x_min <= mean[0] <= x_max and y_min <= mean[1] <= y_max:
                return 0.8
            else:
                return 0.1
    
    def _simulate_action_effect(self, current_summary: np.ndarray, action: int) -> np.ndarray:
        """
        Simulate state change after action execution (mainly position change)
        """
        try:
            new_summary = current_summary.copy()
            
            # Get current agent position
            agent_x = current_summary[6] if len(current_summary) > 6 else 0.5
            agent_y = current_summary[7] if len(current_summary) > 7 else 0.5
            
            # Restore actual position
            agent_pos = np.array([agent_x * MAP_WIDTH, agent_y * MAP_HEIGHT])
            
            # Compute position after action
            angle_deg = AGENT_CONTROL_ANGLES[action]
            angle_rad = np.deg2rad(angle_deg)
            velocity = AGENT_SPEED * np.array([np.cos(angle_rad), np.sin(angle_rad)])
            new_pos = agent_pos + velocity * TIME_STEP_DURATION
            
            # Boundary check
            new_pos[0] = np.clip(new_pos[0], 0, MAP_WIDTH)
            new_pos[1] = np.clip(new_pos[1], 0, MAP_HEIGHT)
            
            # Update position information in summary
            if len(new_summary) > 6:
                new_summary[6] = new_pos[0] / MAP_WIDTH
            if len(new_summary) > 7:
                new_summary[7] = new_pos[1] / MAP_HEIGHT
            
            return new_summary
            
        except Exception as e:
            return current_summary.copy()
        
    def _mcts_search(self, world_state, pmbm_belief, ltl_state, belief_summary) -> int:
        """
        Execute MCTS search
        """
        try:
            # Create root node
            root = MCTSNode(
                state_summary=belief_summary,
                ltl_state=ltl_state,
                parent=None,
                action=None
            )
            
            # Execute MCTS simulation
            for sim_idx in range(self.params.num_simulations):
                # Select and expand
                leaf_node = self._select_and_expand(root)
                
                # Simulation (rollout)
                reward = self._simulate_rollout(leaf_node)
                
                # Backpropagate update
                self._backpropagate(leaf_node, reward)
            
            # Select best action
            best_action, best_child = root.get_best_child()
            if best_action is None:
                # If no child nodes, randomly select
                return np.random.randint(0, len(AGENT_CONTROL_ANGLES))
            
            return best_action
        
        except Exception as e:
            return np.random.randint(0, len(AGENT_CONTROL_ANGLES))
    
    def _is_node_fully_expanded(self, node: MCTSNode) -> bool:
        """Check if node is fully expanded according to Progressive Widening"""
        max_actions = min(
            len(AGENT_CONTROL_ANGLES),
            max(1, int(self.params.progressive_widening_c * np.sqrt(node.visits + 1)))
        )
        max_actions = min(max_actions, self.params.max_actions_per_node)
        return len(node.children) >= max_actions
    
    def _select_and_expand(self, root: MCTSNode) -> MCTSNode:
        """
        Select and expand stage
        """
        current_node = root
        
        # Progress along UCB path until finding an unfully expanded node
        while self._is_node_fully_expanded(current_node) and current_node.children:
            # UCB1 select best action
            best_action = self._ucb1_select(current_node)
            # Move to child node
            current_node = current_node.children[best_action]
        
        # Expand new action
        max_actions = min(
            len(AGENT_CONTROL_ANGLES),
            max(1, int(self.params.progressive_widening_c * np.sqrt(current_node.visits + 1)))
        )
        max_actions = min(max_actions, self.params.max_actions_per_node)
        
        if len(current_node.children) < max_actions:
            # Select an unexplored action
            unexplored_actions = [a for a in range(len(AGENT_CONTROL_ANGLES)) 
                                if a not in current_node.children]
            if unexplored_actions:
                new_action = np.random.choice(unexplored_actions)
                
                # Simulate state change after action execution
                new_summary = self._simulate_action_effect(current_node.state_summary, new_action)
                
                # Create new child node
                child_node = MCTSNode(
                    state_summary=new_summary,
                    ltl_state=current_node.ltl_state,  # LTL state temporarily remain unchanged
                    parent=current_node,
                    action=new_action
                )
                
                current_node.children[new_action] = child_node
                return child_node
        
        return current_node
    
    def _ucb1_select(self, node: MCTSNode) -> int:
        """UCB1 select"""
        if not node.children:
            return 0
        
        best_action = None
        best_value = float('-inf')
        
        for action, child in node.children.items():
            if child.visits == 0:
                return action  # Prioritize selecting unvisited child nodes
            
            # UCB1 formula
            exploitation = child.total_reward / child.visits
            exploration = self.params.ucb_c * np.sqrt(np.log(node.visits) / child.visits)
            ucb_value = exploitation + exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action
        
        return best_action if best_action is not None else 0
    
    def _simulate_rollout(self, node: MCTSNode) -> float:
        """
        Improved rollout - use the same reward logic as LTL_Myopic
        """
        try:
            summary = node.state_summary
            p_exist_A = summary[0]
            p_exist_B = summary[1]
            ltl_state_A = summary[2] if len(summary) > 2 else 0.0
            ltl_state_B = summary[3] if len(summary) > 3 else 0.0
            urgency_A = summary[4] if len(summary) > 4 else 0.0
            urgency_B = summary[5] if len(summary) > 5 else 0.0
            agent_x = summary[6] if len(summary) > 6 else 0.5
            agent_y = summary[7] if len(summary) > 7 else 0.5
            
            # Restore agent position
            agent_pos = np.array([agent_x * MAP_WIDTH, agent_y * MAP_HEIGHT])
            
            # Region center position
            region_A_center = np.array([(REGION_A[0] + REGION_A[1])/2, (REGION_A[2] + REGION_A[3])/2])
            region_B_center = np.array([(REGION_B[0] + REGION_B[1])/2, (REGION_B[2] + REGION_B[3])/2])
            
            dist_to_A = np.linalg.norm(agent_pos - region_A_center)
            dist_to_B = np.linalg.norm(agent_pos - region_B_center)
            
            total_reward = 0.0
            
            # 1. Distance reward: encourage moving towards urgent region (consistent with LTL_Myopic)
            if urgency_A > 0:
                reward_A = max(0, 5.0 - dist_to_A / 100.0) * (urgency_A + 1.0)
                total_reward += reward_A
                
            if urgency_B > 0:
                reward_B = max(0, 5.0 - dist_to_B / 100.0) * (urgency_B + 1.0)
                total_reward += reward_B
            
            # 2. Maintain reward: if both regions have targets, give maintain reward
            if p_exist_A > 0.4 and p_exist_B > 0.4:
                total_reward += 10.0
            
            # 3. Balance reward: encourage balancing attention to two regions
            if p_exist_A > 0.3 and p_exist_B > 0.3:
                balance_bonus = 5.0 * (1.0 - abs(p_exist_A - p_exist_B))
                total_reward += balance_bonus
            
            # 4. Basic exploration reward: if no clear target, move towards the nearest region
            if urgency_A == 0 and urgency_B == 0:
                if dist_to_A < dist_to_B:
                    total_reward += max(0, 3.0 - dist_to_A / 150.0)
                else:
                    total_reward += max(0, 3.0 - dist_to_B / 150.0)
            
            # 5. LTL state completion reward
            if ltl_state_A > 0.5 and ltl_state_B > 0.5:
                total_reward += 15.0  # Big reward, encourage completing LTL task
            
            return total_reward
        
        except Exception as e:
            return 0.0
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate update"""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics information"""
        if len(self.planning_time_history) == 0:
            return {
                'total_planning_time': 0.0,
                'average_planning_time': 0.0,
                'planning_calls': 0
            }
        
        return {
            'total_planning_time': sum(self.planning_time_history),
            'average_planning_time': np.mean(self.planning_time_history),
            'planning_calls': len(self.planning_time_history)
        }