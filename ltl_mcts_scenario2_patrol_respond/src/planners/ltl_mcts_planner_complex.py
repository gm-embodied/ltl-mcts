"""
LTL-MCTS planner implementation
Based on existing MCTS code, adapted to new experimental framework
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
from src.planners.ltl_myopic_planner import LTLMyopicPlanner
from src.utils.data_structures import MCTSNode, LTLState, PMBMBelief, BernoulliComponent
from src.utils.scenario_adapter import get_scenario_adapter, set_current_scenario, get_scenario_config
from src.utils.ltl_window import WindowState, update_window_state

# Import basic configuration (parameters independent of scenario)
from configs.scenario1_config import (
    AGENT_CONTROL_ANGLES, AGENT_SPEED, TIME_STEP_DURATION,
    SENSOR_RANGE, SENSOR_FOV_DEGREES, MAP_WIDTH, MAP_HEIGHT,
    MCTS_NUM_SIMULATIONS, MCTS_MAX_DEPTH, MCTS_EXPLORATION_CONSTANT,
    MCTS_DISCOUNT_FACTOR, AP_PROBABILITY_UPPER_THRESHOLD, AP_PROBABILITY_LOWER_THRESHOLD,
    R_CONFIRM, R_TIMEOUT, R_STEP,
    TARGET_SURVIVAL_PROBABILITY, TARGET_PROCESS_NOISE_COV
)

# Scenario 2 shaping (compatible with scene 1, no impact on scene 1's new reward)
REWARD_WEIGHT_DISTANCE = -0.02
REWARD_WEIGHT_EXISTENCE = 2.0

# Try to import scenario 2 reward function
try:
    from src.planners.scenario2_reward_functions import compute_scenario2_reward, get_scenario2_rollout_reward
    SCENARIO2_AVAILABLE = True
except ImportError:
    SCENARIO2_AVAILABLE = False

def compute_step_reward(prev_ws: WindowState, curr_ws: WindowState,
                        resetA: bool, resetB: bool, violate: bool,
                        agent_pos: np.ndarray = None) -> float:
    """
    Differential reward function - press worst area + suppress jitter
    r_k = α(max(t_{k-1}) - max(t_k)) - λ|t^A_k - t^B_k| + β·reset - δ·delta
    """
    # Parameter setting
    alpha = 1.0      # Press worst area
    lambda_balance = 0.4  # Balance penalty
    beta = 0.5       # Cross delay reward
    delta = 1.0      # Strong penalty for window violation
    
    # Terminal violation large negative reward
    if violate:
        return R_TIMEOUT  # -2.0
    
    reward = 0.0
    W = get_scenario_config('LTL_WINDOW_STEPS')
    
    # 1. Press worst area: reduce differential reward for maximum timer
    prev_max_timer = max(prev_ws.t_A, prev_ws.t_B)
    curr_max_timer = max(curr_ws.t_A, curr_ws.t_B)
    timer_improvement = prev_max_timer - curr_max_timer
    reward += alpha * timer_improvement
    
    # 2. Balance penalty: suppress large difference between two area timers
    timer_imbalance = abs(curr_ws.t_A - curr_ws.t_B)
    reward -= lambda_balance * timer_imbalance
    
    # 3. Cross delay reward (OFF→ON conversion)
    if resetA or resetB:
        reward += beta
    
    # 4. Strong penalty for window violation (prevent violation)
    if curr_ws.t_A >= W or curr_ws.t_B >= W:
        reward -= delta
    
    # 5. Basic step reward (keep exploration)
    reward += R_STEP  # -0.003
    
    return reward

@dataclass
class MCTSParams:
    """MCTS parameter configuration"""
    num_simulations: int = 500  # Increase search 
    max_depth: int = 15   # Medium search depth
    gamma: float = 0.95   # Medium discount factor
    ucb_c: float = 2.0    # Increase exploration
    progressive_widening_c: float = 1.5  # More conservative Progressive Widening
    max_actions_per_node: int = 6  # Increase action exploration space
    hysteresis_upper: float = AP_PROBABILITY_UPPER_THRESHOLD
    hysteresis_lower: float = AP_PROBABILITY_LOWER_THRESHOLD

class LTLMCTSPlanner(BasePlanner):
    """LTL-MCTS planner - core contribution of the paper"""
    
    def __init__(self, config=None):
        super().__init__("LTL_MCTS", config)
        self.params = MCTSParams()
        if config:
            for key, value in config.items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
        
        # MCTS statistics
        self.Ns: Dict[Tuple, int] = {}          # Node visits
        self.Nsa: Dict[Tuple, Dict[int, int]] = {}  # State-action visits
        self.Qsa: Dict[Tuple, Dict[int, float]] = {}  # State-action Q values
        
        # Cache recent belief summary for debugging
        self.last_belief_summary = None
        self.last_ltl_state = None
        
        # Scenario adapter
        self.adapter = get_scenario_adapter()
        self.scenario_type = None  # Will be detected on first call
        
        # Intelligent rollout strategy: use LTL_Myopic as "old soldier"
        self.rollout_planner = LTLMyopicPlanner(config)
    
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        Use LTL-MCTS algorithm to plan next action
        """
        start_time = time.time()
        
        # Scenario detection (on first call)
        if self.scenario_type is None:
            self.scenario_type = self.adapter.detect_scenario_from_ltl_state(ltl_state)
            self.adapter.set_scenario_type(self.scenario_type)
        
        # Get scenario-related configuration
        debug_mcts = get_scenario_config('DEBUG_MCTS')
        
        if debug_mcts:
            print(f"--- LTL_MCTS_Planner: plan_action START (Scenario: {self.scenario_type}) ---")
        
        # 1. Create LTL belief summary
        belief_summary = self._create_belief_summary(pmbm_belief, ltl_state)
        self.last_belief_summary = belief_summary
        self.last_ltl_state = ltl_state
        
        # 2. Run MCTS search
        best_action = self._mcts_search(world_state, pmbm_belief, ltl_state, belief_summary)
        
        # 3. Record statistics
        planning_time = time.time() - start_time
        self.planning_time_history.append(planning_time)
        self.action_history.append(best_action)
        if debug_mcts:
            print(f"--- LTL_MCTS_Planner: plan_action END ---")
            print(f"Actual planning time: {planning_time * 1000:.2f} ms")
        
        return best_action
    
    def _detect_scenario_type(self, ltl_state: LTLState) -> str:
        """Detect current scenario type"""
        if hasattr(ltl_state, 'scenario_type') and ltl_state.scenario_type:
            return ltl_state.scenario_type
        
        # Determine scenario type based on LTL state's memory_state
        if hasattr(ltl_state, 'memory_state') and ltl_state.memory_state:
            if 'ap_exist_A' in ltl_state.memory_state:
                return 'scenario2'
        
        return 'scenario1'  # Default to scenario 1
    
    def _create_belief_summary(self, pmbm_belief: PMBMBelief, ltl_state: LTLState) -> np.ndarray:
        """
        Create LTL belief summary b^φ - optimized for patrol task
        Summary vector: [P_exist(A), P_exist(B), t_A, t_B]
        """
        # Get current scenario's region definition
        region_a, region_b = self.adapter.get_regions()
        
        # Compute region existence probability (consistent with environment calculation)
        p_exist_A = self._compute_region_exist_prob(pmbm_belief, region_a)
        p_exist_B = self._compute_region_exist_prob(pmbm_belief, region_b)
        
        # Get window timer from LTL state (if available)
        t_A = 0.0
        t_B = 0.0
        if hasattr(ltl_state, 'memory_state') and ltl_state.memory_state:
            # Try to get window state from memory_state
            if 'window_state' in ltl_state.memory_state:
                ws = ltl_state.memory_state['window_state']
                if hasattr(ws, 't_A') and hasattr(ws, 't_B'):
                    t_A = float(ws.t_A)
                    t_B = float(ws.t_B)
        
        # Build belief summary vector: [P_exist(A), P_exist(B), t_A, t_B]
        summary = np.array([p_exist_A, p_exist_B, t_A, t_B])
        
        return summary
    
    def _compute_region_exist_prob(self, pmbm_belief: PMBMBelief, region: Tuple[float, float, float, float]) -> float:
        """
        Compute region existence probability - consistent with environment implementation
        P_exist(R) = 1 - [exp(-Lambda_R) * ∏(1 - r_i * P_{i,R})]
        """
        # Get configuration parameters
        ppp_density = float(get_scenario_config('PPP_DENSITY'))
        
        # Compute region area and PPP contribution
        x_min, x_max, y_min, y_max = region
        area_R = (x_max - x_min) * (y_max - y_min)
        lambda_R = ppp_density * area_R
        
        # Get MAP hypothesis
        map_hyp = pmbm_belief.get_map_hypothesis()
        if not map_hyp or not hasattr(map_hyp, 'bernoulli_components'):
            # Only PPP contribution
            return 1.0 - np.exp(-lambda_R)
        
        # Compute Bernoulli component contribution
        bernoulli_product = 1.0
        for comp in map_hyp.bernoulli_components:
            r_i = float(comp.existence_prob)
            # Compute Gaussian probability in rectangular region (simplified center point check)
            comp_pos = comp.get_position_mean()
            if x_min <= comp_pos[0] <= x_max and y_min <= comp_pos[1] <= y_max:
                P_i_R = 0.8  # Simplified: if center in region, probability set to 0.8
            else:
                P_i_R = 0.1  # Simplified: if center in region, probability set to 0.1
            
            bernoulli_product *= (1.0 - r_i * P_i_R)
        
        # Compute final existence probability
        p_exist = 1.0 - (np.exp(-lambda_R) * bernoulli_product)
        return float(np.clip(p_exist, 0.0, 1.0))
    
    def _compute_atomic_proposition_probabilities(self, pmbm_belief: PMBMBelief) -> Dict[str, float]:
        """
        Compute atomic proposition satisfaction probability
        Implement semantic-probability bridge in the paper
        """
        probs = {}
        
        # Get MAP hypothesis
        map_targets = pmbm_belief.get_map_targets()
        
        # ap_confirm_any: any target is confirmed
        confirm_prob = 0.0
        for target in map_targets:
            if target.existence_prob > 0.5:
                trace = target.get_covariance_trace()
                # Stable sigmoid mapping, avoid exp overflow
                alpha = 0.1
                x = alpha * (trace - 50.0)
                if x >= 0:
                    ex = np.exp(-x)
                    confirm_prob_single = 1.0 / (1.0 + ex)
                else:
                    ex = np.exp(x)
                    confirm_prob_single = ex / (1.0 + ex)
                confirm_prob = max(confirm_prob, confirm_prob_single)
        
        probs['ap_confirm_any'] = confirm_prob
        
        # Scenario 2 atomic proposition (if needed)
        if hasattr(self, 'scenario_type') and self.scenario_type == 'scenario2':
            probs.update(self._compute_scenario2_ap_probabilities(pmbm_belief))
        
        return probs
    
    def _compute_scenario2_ap_probabilities(self, pmbm_belief: PMBMBelief) -> Dict[str, float]:
        """Compute scenario 2 atomic proposition probability"""
        probs = {}
        
        # ap_exist_A: A region exists target
        # Simplified implementation: check if any target's position estimate is in A region
        region_A = [0.0, 300.0, 0.0, 1000.0]  # [xmin, xmax, ymin, ymax]
        
        exist_A_prob = 0.0
        map_targets = pmbm_belief.get_map_targets()
        
        for target in map_targets:
            pos = target.get_position_mean()
            if (region_A[0] <= pos[0] <= region_A[1] and 
                region_A[2] <= pos[1] <= region_A[3]):
                exist_A_prob = max(exist_A_prob, target.existence_prob)
        
        probs['ap_exist_A'] = exist_A_prob
        
        # ap_loc_any_A: any target is located in A region
        loc_A_prob = 0.0
        for target in map_targets:
            pos = target.get_position_mean()
            if (region_A[0] <= pos[0] <= region_A[1] and 
                region_A[2] <= pos[1] <= region_A[3]):
                # Consider existence probability and position uncertainty
                pos_cov = target.get_position_cov()
                uncertainty_factor = 1.0 / (1.0 + np.trace(pos_cov) / 100.0)
                loc_A_prob = max(loc_A_prob, target.existence_prob * uncertainty_factor)
        
        probs['ap_loc_any_A'] = loc_A_prob
        
        return probs
    
    def _mcts_search(self, world_state, pmbm_belief, ltl_state, belief_summary) -> int:
        """
        Execute MCTS search
        """
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
            leaf_node, leaf_belief = self._select_and_expand(
                root, world_state, pmbm_belief, ltl_state
            )
            
            # Simulation (rollout)
            reward = self._simulate_rollout(leaf_node, leaf_belief, world_state)
            
            # Backpropagate update
            self._backpropagate(leaf_node, reward)
            import sys
            module = sys.modules[__name__]
            debug_mcts = getattr(module, 'DEBUG_MCTS', False)
            if debug_mcts and sim_idx % max(1, self.params.num_simulations // 5) == 0:
                print(f"MCTS sim {sim_idx}/{self.params.num_simulations}")
        
        # Select best action
        best_action, best_child = root.get_best_child()
        if best_action is None:
            # If no children, randomly select
            import sys
            module = sys.modules[__name__]
            debug_mcts = getattr(module, 'DEBUG_MCTS', False)
            if debug_mcts:
                print("WARNING: No children found, using random action")
            return np.random.randint(0, len(AGENT_CONTROL_ANGLES))
        
        # Debug information
        import sys
        module = sys.modules[__name__]
        debug_mcts = getattr(module, 'DEBUG_MCTS', False)
        if debug_mcts:
            print(f"Root children: {len(root.children)}")
            for action, child in root.children.items():
                avg_value = child.total_reward / max(1, child.visits)
                print(f"  Action {action}: visits={child.visits}, avg_value={avg_value:.3f}")
            print(f"Selected action: {best_action} (visits={best_child.visits})")
        
        return best_action
    
    def _is_node_fully_expanded(self, node: MCTSNode) -> bool:
        """Check if node is fully expanded according to Progressive Widening"""
        max_actions = min(
            len(AGENT_CONTROL_ANGLES),
            max(1, int(self.params.progressive_widening_c * np.sqrt(node.visits + 1)))
        )
        max_actions = min(max_actions, self.params.max_actions_per_node)
        return len(node.children) >= max_actions
    
    def _select_and_expand(self, root: MCTSNode, world_state, pmbm_belief, ltl_state) -> Tuple[MCTSNode, Any]:
        """
        Selection and expansion phase - fix UCB path propagation problem
        """
        current_node = root
        current_belief = pmbm_belief
        current_world = world_state
        current_ltl = ltl_state
        
        # Propagate state along UCB path until finding an unexpanded node
        while self._is_node_fully_expanded(current_node) and current_node.children:
            # UCB1 select best action
            best_action = self._ucb1_select(current_node)
            
            # Key fix: propagate world/belief/ltl state along path
            current_world, _, current_belief, current_ltl = self._simulate_action(
                current_world, current_belief, current_ltl, best_action
            )
            
            # Move to child node
            current_node = current_node.children[best_action]
        
        # Now on an unexpanded node, expand a new action
        # Progressive Widening: limit action expansion based on visits
        max_actions = min(
            len(AGENT_CONTROL_ANGLES),
            max(1, int(self.params.progressive_widening_c * np.sqrt(current_node.visits + 1)))
        )
        max_actions = min(max_actions, self.params.max_actions_per_node)
        
        if len(current_node.children) < max_actions:
            # Expand new action - randomly select instead of always selecting first
            unexplored_actions = [a for a in range(len(AGENT_CONTROL_ANGLES)) 
                                if a not in current_node.children]
            action = np.random.choice(unexplored_actions)
            
            # Simulation execute action
            next_world, next_observation, next_belief, next_ltl_state = self._simulate_action(
                current_world, current_belief, current_ltl, action
            )
            
            # Create new belief summary
            next_summary = self._create_belief_summary(next_belief, next_ltl_state)
            
            # Create child node
            child_node = MCTSNode(
                state_summary=next_summary,
                ltl_state=next_ltl_state,
                parent=current_node,
                action=action
            )
            
            current_node.children[action] = child_node
            return child_node, next_belief
        
        # If fully expanded and no children, return current node
        return current_node, current_belief
    
    def _ucb1_select(self, node: MCTSNode) -> int:
        """
        Use UCB1 formula to select action
        """
        best_score = -float('inf')
        best_action = None
        
        for action, child in node.children.items():
            score = child.get_ucb1_score(self.params.ucb_c)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _simulate_action(self, world_state, pmbm_belief, ltl_state, action: int) -> Tuple[Any, Any, Any, Any]:
        """
        Simulation execute an action
        Return: (next_world_state, observation, next_belief, next_ltl_state)
        """
        # Lightweight generative model: approximate belief update based on FoV
        from copy import deepcopy
        angle_rad = np.deg2rad(AGENT_CONTROL_ANGLES[action])
        velocity = AGENT_SPEED * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        new_position = world_state.agent_state.position + velocity * TIME_STEP_DURATION
        new_position[0] = np.clip(new_position[0], 0, MAP_WIDTH)
        new_position[1] = np.clip(new_position[1], 0, MAP_HEIGHT)

        next_world = deepcopy(world_state)
        next_world.agent_state.position = new_position
        next_world.agent_state.heading = angle_rad

        # Lightweight belief update - avoid deep copy of entire belief
        next_belief = deepcopy(pmbm_belief)  
        hyp = next_belief.get_map_hypothesis()
        if hyp is not None:
            # Directly update components in place, avoid repeated deep copy
            for comp in hyp.bernoulli_components:
                # Prediction step:  
                if hasattr(comp, 'state_cov'):
                    # Only update trace of position covariance (scalar approximation)
                    pos_trace = np.trace(comp.state_cov[:2, :2])
                    comp.state_cov[:2, :2] += np.eye(2) * (pos_trace * 0.1)   
                
                comp.existence_prob *= TARGET_SURVIVAL_PROBABILITY

                # FoV observation update 
                pos_mean = comp.get_position_mean()
                if (
                    self._is_in_fov(pos_mean, new_position, angle_rad)
                    and (np.linalg.norm(pos_mean - new_position) <= SENSOR_RANGE)
                ):
                    #  observation update
                    comp.existence_prob = min(0.99, comp.existence_prob + 0.15)
                    if hasattr(comp, 'state_cov'):
                        comp.state_cov[:2, :2] *= 0.7  

        # Propagate LTL state according to updated belief (with lag threshold) - consistent with environment
        ap_probs = self._compute_atomic_proposition_probabilities(next_belief)
        next_ltl_state = LTLState(ltl_state.automaton_state, ltl_state.memory_state)
        for ap_name, prob in ap_probs.items():
            cur = next_ltl_state.memory_state.get(ap_name, False)
            ap_exist_prob_on = get_scenario_config('AP_EXIST_PROB_ON')
            ap_exist_prob_off = get_scenario_config('AP_EXIST_PROB_OFF')
            if not cur and prob > ap_exist_prob_on:
                next_ltl_state.memory_state[ap_name] = True
            elif cur and prob < ap_exist_prob_off:
                next_ltl_state.memory_state[ap_name] = False
        # Simplify LDBA: scenario 1 reaches confirmation state
        if hasattr(next_ltl_state, 'scenario_type') and next_ltl_state.scenario_type == 'scenario2':
            pass
        else:
            if next_ltl_state.memory_state.get('ap_confirm_any', False):
                next_ltl_state.automaton_state = 1
        observation = None
        return next_world, observation, next_belief, next_ltl_state

    def _is_in_fov(self, target_pos: np.ndarray, agent_pos: np.ndarray, agent_heading: float) -> bool:
        rel = target_pos - agent_pos
        ang = np.arctan2(rel[1], rel[0])
        diff = abs(ang - agent_heading)
        diff = min(diff, 2 * np.pi - diff)
        return diff <= (np.deg2rad(SENSOR_FOV_DEGREES) / 2.0)
    
    def _simulate_rollout(self, node: MCTSNode, belief, world_state) -> float:
        """Scenario-adaptive rollout simulation"""
        if self.scenario_type == 'scenario2' and SCENARIO2_AVAILABLE:
            return self._simulate_scenario2_rollout(node, belief, world_state)
        else:
            return self._simulate_scenario1_rollout(node, belief, world_state)
    
    def _simulate_scenario2_rollout(self, node: MCTSNode, belief, world_state) -> float:
        """Scenario 2"""
        total_reward = 0.0
        discount = 1.0
        current_depth = 0
        
        # Extract current LTL state information
        ltl_state = node.ltl_state
        duty_triggered = getattr(ltl_state, 'duty_to_patrol_triggered', False) if hasattr(ltl_state, 'duty_to_patrol_triggered') else False
        steps_since_duty = getattr(ltl_state, 'steps_since_duty_triggered', 0) if hasattr(ltl_state, 'steps_since_duty_triggered') else 0
        
        # rollout loop
        while current_depth < self.params.max_depth:
            # Scenario 2 intelligent strategy:
            # 1. If duty is triggered, prioritize towards A region
            # 2. Otherwise, patrol near A region
            
            # Get current scenario region definition
            region_a, region_b = self.adapter.get_regions()
            
            if duty_triggered:
                # Duty is triggered, must towards A region
                target_region = region_a
                cx = (target_region[0] + target_region[1]) / 2.0
                cy = (target_region[2] + target_region[3]) / 2.0
            else:
                # Normal patrol, prioritize A region
                agent_pos = world_state.agent_state.position if world_state.agent_state else np.array([500.0, 500.0])
                
                # Calculate distance to A and B regions
                center_A = np.array([(region_a[0] + region_a[1]) / 2.0, 
                                   (region_a[2] + region_a[3]) / 2.0])
                
                # Towards A region center (preventive patrol)
                cx, cy = center_A[0], center_A[1]
            
            action = self._choose_action_towards(world_state, np.array([cx, cy]))
            
            # Forward simulate one step
            next_world, _obs, next_belief, next_ltl_state = self._simulate_action(
                world_state, belief, node.ltl_state, action
            )
            
            # Build state information for reward calculation
            current_state = {
                'agent_position': world_state.agent_state.position if world_state.agent_state else np.array([500.0, 500.0]),
                'duty_triggered': duty_triggered,
                'duty_completed': False,   
                'steps_since_duty': steps_since_duty
            }
            
            next_state = {
                'agent_position': next_world.agent_state.position if next_world.agent_state else np.array([500.0, 500.0]),
                'duty_triggered': duty_triggered,   
                'duty_completed': False,
                'steps_since_duty': steps_since_duty + 1 if duty_triggered else 0,
                'b_region_has_target': False   
            
            # Use scenario 2 
            step_reward = compute_scenario2_reward(current_state, action, next_state, ltl_violation=False)
            total_reward += discount * step_reward
            
            # Propagate state
            world_state = next_world
            belief = next_belief
            node.ltl_state = next_ltl_state
            
            current_depth += 1
            discount *= self.params.gamma
        
        return total_reward
    
    def _simulate_scenario1_rollout(self, node: MCTSNode, belief, world_state) -> float:
        """Scenario 1 """
        total_reward = 0.0
        discount = 1.0
        current_depth = 0

        # Get current scenario region definition and parameters
        region_a, region_b = self.adapter.get_regions()
        ap_exist_prob_on = get_scenario_config('AP_EXIST_PROB_ON')
        
        # Estimate initial region existence probability based on current belief, initialize window state
        pA0 = self._prob_exist_in_region(belief, region_a)
        pB0 = self._prob_exist_in_region(belief, region_b)
        ws = WindowState(
            t_A=0,
            t_B=0,
            apA_on=(pA0 >= ap_exist_prob_on),
            apB_on=(pB0 >= ap_exist_prob_on),
            confirm_count_A=0,
            confirm_count_B=0,
            armed_A=(pA0 >= ap_exist_prob_on),
            armed_B=(pB0 >= ap_exist_prob_on)
        )

        # rollout loop  
        while current_depth < self.params.max_depth:
            # Simple strategy: towards region with higher timer
            if ws.t_A > ws.t_B + 2:  # A region more urgent
                cx = (region_a[0] + region_a[1]) / 2.0
                cy = (region_a[2] + region_a[3]) / 2.0
            elif ws.t_B > ws.t_A + 2:  # B region more urgent
                cx = (region_b[0] + region_b[1]) / 2.0
                cy = (region_b[2] + region_b[3]) / 2.0
            else:
                # When timer is approaching, randomly select or towards farther region
                agent_pos = world_state.agent_state.position if world_state.agent_state else np.array([500.0, 500.0])
                center_A = np.array([(region_a[0] + region_a[1]) / 2.0, (region_a[2] + region_a[3]) / 2.0])
                center_B = np.array([(region_b[0] + region_b[1]) / 2.0, (region_b[2] + region_b[3]) / 2.0])
                
                dist_A = np.linalg.norm(agent_pos - center_A)
                dist_B = np.linalg.norm(agent_pos - center_B)
                
                if dist_A > dist_B:  # Towards farther region (balanced patrol)
                    cx, cy = center_A[0], center_A[1]
                else:
                    cx, cy = center_B[0], center_B[1]
            
            action = self._choose_action_towards(world_state, np.array([cx, cy]))

            # Forward simulate one step
            next_world, _obs, next_belief, next_ltl_state = self._simulate_action(
                world_state, belief, node.ltl_state, action
            )

            # Calculate next region existence probability
            pA = self._prob_exist_in_region(next_belief, region_a)
            pB = self._prob_exist_in_region(next_belief, region_b)

            # Update window state and calculate reward - use consistent threshold with environment
            prev_ws = WindowState(**vars(ws))
            ltl_window_steps = get_scenario_config('LTL_WINDOW_STEPS')
            ap_exist_prob_off = get_scenario_config('AP_EXIST_PROB_OFF')
            ws, resetA, resetB, violate = update_window_state(ws, pA, pB,
                                                              ltl_window_steps,
                                                              ap_exist_prob_on,
                                                              ap_exist_prob_off)
            
            # Get agent position for reward calculation
            agent_pos = next_world.agent_state.position if next_world.agent_state else None
            step_reward = compute_step_reward(prev_ws, ws, resetA, resetB, violate, agent_pos)
            total_reward += discount * step_reward

            # Termination condition: window violation
            if violate:
                break

            # Propagate state
            world_state = next_world
            belief = next_belief
            node.ltl_state = next_ltl_state

            current_depth += 1
            discount *= self.params.gamma

        return total_reward

    def _compute_eta(self, agent_pos: np.ndarray, region: Tuple[float, float, float, float], current_timer: int) -> float:
        """
        Calculate expected recovery time (ETA)
        ETA_R = ceil(max(0, dist_to_visible(R) - FoV_margin) / v) + t_confirm
        """
        # Region center
        center_x = (region[0] + region[1]) / 2.0
        center_y = (region[2] + region[3]) / 2.0
        region_center = np.array([center_x, center_y])
        
        # Distance to region center
        dist_to_center = np.linalg.norm(agent_pos - region_center)
        
        # FoV coverage margin (simplified: assume need to be within SENSOR_RANGE distance from region edge)
        fov_margin = SENSOR_RANGE * 0.8  # Conservative estimate
        
        # Moving time estimate
        dist_to_visible = max(0.0, dist_to_center - fov_margin)
        move_time = np.ceil(dist_to_visible / AGENT_SPEED) if AGENT_SPEED > 0 else 0
        
        # Confirm time (assume need 2-3 steps to stabilize observation and trigger ON)
        confirm_time = 3
        
        # Total ETA = current timer + moving time + confirm time
        eta = current_timer + move_time + confirm_time
        
        return float(eta)

    def _choose_action_towards(self, world_state, target_xy: np.ndarray) -> int:
        agent_pos = world_state.agent_state.position
        vec = target_xy - agent_pos
        desired = np.arctan2(vec[1], vec[0])
        best_idx = 0
        best_diff = float('inf')
        for idx, angle_deg in enumerate(AGENT_CONTROL_ANGLES):
            ang = np.deg2rad(angle_deg)
            diff = abs(ang - desired)
            diff = min(diff, 2 * np.pi - diff)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        return best_idx
    def _prob_exist_in_region(self, pmbm_belief: PMBMBelief, region: list) -> float:
        map_hyp = pmbm_belief.get_map_hypothesis()
        if map_hyp is None:
            return 0.0
        p_none = 1.0
        for comp in map_hyp.bernoulli_components:
            pos = comp.get_position_mean()
            in_R = (region[0] <= pos[0] <= region[1]) and (region[2] <= pos[1] <= region[3])
            if in_R:
                p_none *= (1.0 - float(comp.existence_prob))
        # PPP项：Λ_R ≈ λ0 * Area(R)
        try:
            lam0 = pmbm_belief.poisson_component.intensity_function(None) if pmbm_belief.poisson_component else 0.0
        except Exception:
            lam0 = 0.0
        area_R = float((region[1] - region[0]) * (region[3] - region[2]))
        Lambda_R = max(0.0, float(lam0) * area_R)
        p_exist = 1.0 - np.exp(-Lambda_R) * p_none
        return float(np.clip(p_exist, 0.0, 1.0))
    
    def get_shaping_reward(self, current_sim_state: dict) -> float:
        """Calculate shaping reward for MCTS (only for scenario 2)"""
        # Check if scenario 2 configuration is available
        if not SCENARIO2_AVAILABLE:
            return 0.0
            
        ltl_state = current_sim_state.get('ltl_state')
        
        # 1. Precondition check: duty is triggered?
        if not hasattr(ltl_state, 'duty_is_active') or not ltl_state.duty_is_active:
            return 0.0  # When duty is not triggered, no guidance is provided, reward is 0
        
        # 2. Duty is triggered, start providing "breadcrumbs" guidance
        shaping_reward = 0.0
        
        # Get current scenario region definition
        region_a, region_b = self.adapter.get_regions()
        
        # Guidance 1 
        agent_state = current_sim_state.get('agent_state')
        if agent_state and hasattr(agent_state, 'position'):
            agent_pos = agent_state.position
            distance_to_A = self._calculate_distance_to_region(agent_pos, region_a)
            shaping_reward += REWARD_WEIGHT_DISTANCE * distance_to_A
        
        # Guidance 2:  
        belief = current_sim_state.get('belief')
        if belief:
            target_in_A = self._find_best_target_in_region(belief, region_a)
            
            if target_in_A is not None:
                existence_prob = target_in_A.existence_prob
                shaping_reward += REWARD_WEIGHT_EXISTENCE * existence_prob
                
        return shaping_reward
    
    def _calculate_distance_to_region(self, point: np.ndarray, region: list) -> float:
        """Calculate the shortest distance from a point to a rectangular region. If the point is in the rectangle, return 0.
        region format: [xmin, xmax, ymin, ymax]
        """
        x, y = point[0], point[1]
        xmin, xmax, ymin, ymax = region
        
        # If point is in rectangle, distance is 0
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return 0.0
        
        # Calculate shortest distance to rectangle boundary
        dx = max(0, max(xmin - x, x - xmax))
        dy = max(0, max(ymin - y, y - ymax))
        
        return np.sqrt(dx**2 + dy**2)
    
    def _find_best_target_in_region(self, belief, region: list):
        """Traverse all tracked targets in belief (Bernoulli components),
        return the one with the highest existence probability in the region.
        If there is no target in the region, return None.
        """
        if not belief:
            return None
            
        map_targets = belief.get_map_targets()
        if not map_targets:
            return None
        
        best_target = None
        best_prob = 0.0
        
        xmin, xmax, ymin, ymax = region
        
        for target in map_targets:
            pos = target.get_position_mean()
            x, y = pos[0], pos[1]
            
            # Check if target is in region
            if xmin <= x <= xmax and ymin <= y <= ymax:
                if target.existence_prob > best_prob:
                    best_prob = target.existence_prob
                    best_target = target
        
        return best_target
    
    def _compute_reward(self, ltl_state: LTLState, belief_summary: np.ndarray) -> float:
        """
        Calculate reward function - fixed version
        Based on LTL automaton state and belief summary
        """
        # Basic reward: based on LTL automaton state
        if ltl_state.automaton_state == 1:  # Accepting state
            return 100.0   
        
        # Potential reward: based on atomic proposition probability
        potential_reward = 0.0
        if belief_summary is not None and len(belief_summary) > 0:
            # Scenario 1: encourage confirm probability increase
            confirm_prob = belief_summary[0]
            if confirm_prob > 0.01:
                potential_reward += confirm_prob * 20.0   
            
            # Scenario 2: extra consideration for patrol duty
            if len(belief_summary) > 1:
                potential_reward += belief_summary[1] * 15.0
        
        # Base exploration reward
        base_reward = 1.0   
        return potential_reward + base_reward
    
    def _is_terminal_state(self, ltl_state: LTLState) -> bool:
        """
        Check if it is a terminal state
        """
        return ltl_state.automaton_state == 1  # Accepting state
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagate update node statistics
        """
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent
    
    def reset(self):
        """Reset planner state"""
        super().reset()
        self.Ns.clear()
        self.Nsa.clear()
        self.Qsa.clear()
        self.last_belief_summary = None
        self.last_ltl_state = None
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information"""
        return {
            'last_belief_summary': self.last_belief_summary.tolist() if self.last_belief_summary is not None else None,
            'last_ltl_state': {
                'automaton_state': self.last_ltl_state.automaton_state if self.last_ltl_state else None,
                'memory_state': self.last_ltl_state.memory_state if self.last_ltl_state else None
            },
            'tree_size': len(self.Ns),
            'mcts_params': {
                'num_simulations': self.params.num_simulations,
                'max_depth': self.params.max_depth,
                'gamma': self.params.gamma,
                'ucb_c': self.params.ucb_c
            }
        }
