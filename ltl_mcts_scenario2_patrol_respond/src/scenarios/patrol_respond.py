"""
Scenario 2: Patrol and Respond Task
LTL Formula: G(ap_exist_A => F(ap_loc_any_A))
Implementation of patrol and respond simulation environment with scripted events
"""

import sys
import numpy as np
import random
from typing import List, Tuple, Dict, Any
import logging
from dataclasses import dataclass

# Import configuration and data structures
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from configs.config import *
from src.filters.proper_pmbm_filter import ProperPMBMFilter
from src.utils.data_structures import (
    AgentState, TargetState, WorldState, Observation, Measurement,
    PMBMBelief, AtomicProposition, LTLState, create_initial_pmbm_belief
)

class PatrolRespondEnvironment:
    """Scenario 2: Patrol and Respond Task Simulation Environment"""
    
    def __init__(self, config=None, random_seed=None):
        self.config = config or {}
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(LOG_LEVEL)
        
        # Environment state
        self.world_state = None
        self.pmbm_belief = None
        self.current_ltl_state = None
        
        # Patrol and respond task state
        self.patrol_duty_active = False
        self.patrol_duty_start_time = None
        self.patrol_duty_completed = False
        self.ltl_violation = False
        
        # Scripted events tracking
        self.target_a_spawned = False
        self.target_b_spawned = False
        
        # Statistics
        self.stats = {
            'task_completed': False,
            'completion_time': None,
            'ltl_violation': False,
            'violation_time': None,
            'patrol_efficiency': 0,
            'duty_activations': 0,
            'duty_completions': 0,
            'temptation_resistance': 0,
            'total_detections': 0,
            'total_false_alarms': 0,
            'trajectory': [],
            'belief_history': [],
            'ap_probability_history': []
        }
        
        # PMBM filter
        self.pmbm_filter = None
    
    def initialize(self) -> Tuple[WorldState, PMBMBelief, LTLState]:
        """Initialize environment"""
        self.logger.info("Initializing patrol and respond scenario...")
        
        # Initialize agent state
        agent_state = AgentState(
            position=np.array(AGENT_INITIAL_POSITION, dtype=float),
            velocity=np.array([0.0, 0.0]),
            heading=0.0
        )
        
        # Initialize empty target list (targets will be spawned by scripted events)
        targets = []
        
        # Initialize world state
        self.world_state = WorldState(time_step=0)
        self.world_state.agent_state = agent_state
        self.world_state.targets = targets
        
        # Initialize PMBM filter
        self.pmbm_filter = ProperPMBMFilter()
        # Set motion model
        self.pmbm_filter.set_motion_model(
            TARGET_TRANSITION_MATRIX, 
            TARGET_PROCESS_NOISE_COV, 
            SENSOR_MEASUREMENT_NOISE_COV
        )
        
        # Initialize PMBM belief
        self.pmbm_belief = create_initial_pmbm_belief()
        
        # Initialize LTL state
        self.current_ltl_state = LTLState(
            automaton_state=0, 
            memory_state={'ap_exist_A': False, 'ap_loc_any_A': False}
        )
        
        # Reset patrol duty state
        self.patrol_duty_active = False
        self.patrol_duty_start_time = None
        self.patrol_duty_completed = False
        self.ltl_violation = False
        
        # Reset scripted events
        self.target_a_spawned = False
        self.target_b_spawned = False
        
        self.logger.info(f"Scenario initialization complete - Agent position: {agent_state.position}")
        return self.world_state, self.pmbm_belief, self.current_ltl_state
    
    def step(self, action: int, planner_name: str = "Unknown") -> Tuple[WorldState, Observation, PMBMBelief, LTLState, bool, Dict[str, Any]]:
        """Execute one simulation step"""
        if self.ltl_violation:
            return self.world_state, Observation([]), self.pmbm_belief, self.current_ltl_state, True, {
                'patrol_duty_active': False, 'duty_violation': True, 'task_completed': False
            }
        
        current_step = self.world_state.time_step
        
        # 1. Execute agent action
        self._execute_agent_action(action)
        
        # 2. Handle scripted events (target spawning)
        self._handle_scripted_events()
        
        # 3. Update target states
        self._update_targets()
        
        # 4. Generate observation
        observation = self._generate_observation()
        
        # 5. Update PMBM belief
        # First predict
        self.pmbm_belief = self.pmbm_filter.predict(self.pmbm_belief)
        # Then update
        self.pmbm_belief = self.pmbm_filter.update(self.pmbm_belief, observation)
        
        # 6. Calculate atomic proposition probabilities
        ap_probs = self._compute_atomic_proposition_probabilities()
        
        # 7. Update patrol duty state and check LTL
        duty_status = self._update_patrol_duty_state(ap_probs)
        
        # 8. Check task completion and violation
        task_completed, ltl_violation = self._check_task_completion()
        
        # 9. Determine if simulation should end
        is_done = (current_step >= SIMULATION_TIME_STEPS - 1) or task_completed or ltl_violation
        
        # 10. Check temptation resistance (agent going to region B when duty is active)
        self._check_temptation_resistance(action, duty_status)
        
        # 11. Update statistics
        self._update_statistics(observation, ap_probs, duty_status)
        
        # 11. Update LTL state memory
        self.current_ltl_state.memory_state['ap_exist_A'] = ap_probs['ap_exist_A'] > LTL_EXIST_THRESHOLD
        self.current_ltl_state.memory_state['ap_loc_any_A'] = ap_probs['ap_loc_any_A'] > LTL_LOC_THRESHOLD
        
        # 12. Increment time step
        self.world_state.time_step += 1
        
        # 13. Prepare info dictionary
        info = {
            'ap_exist_A': ap_probs['ap_exist_A'],
            'ap_loc_any_A': ap_probs['ap_loc_any_A'],
            'patrol_duty_active': self.patrol_duty_active,
            'duty_violation': ltl_violation,
            'task_completed': task_completed,
            'current_step': current_step
        }
        
        return self.world_state, observation, self.pmbm_belief, self.current_ltl_state, is_done, info
    
    def _execute_agent_action(self, action: int):
        """Execute agent action - strictly according to motion model"""
        # Get control angle
        control_angle = AGENT_CONTROL_ANGLES[action]
        control_angle_rad = np.deg2rad(control_angle)
        
        # Update heading (normalize to [-π, π])
        self.world_state.agent_state.heading = self._normalize_angle(control_angle_rad)
        
        # Calculate velocity vector (based on new heading)
        velocity = AGENT_SPEED * np.array([
            np.cos(self.world_state.agent_state.heading),
            np.sin(self.world_state.agent_state.heading)
        ])
        
        # Update position (constant velocity integration)
        new_position = self.world_state.agent_state.position + velocity * TIME_STEP_DURATION
        
        # Boundary check
        new_position[0] = np.clip(new_position[0], 0, MAP_WIDTH)
        new_position[1] = np.clip(new_position[1], 0, MAP_HEIGHT)
        
        # Update agent state
        self.world_state.agent_state.position = new_position
        self.world_state.agent_state.velocity = velocity
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _handle_scripted_events(self):
        """Handle scripted target spawning events"""
        current_step = self.world_state.time_step
        
        # Spawn target in region A at designated time
        if current_step == SCENARIO_2_TARGET_A_SPAWN_TIME and not self.target_a_spawned:
            target_a = TargetState(
                id="scripted_target_A",
                is_alive=True,
                position=np.array(SCENARIO_2_TARGET_A_SPAWN_POS, dtype=float),
                velocity=np.array(SCENARIO_2_TARGET_A_VELOCITY, dtype=float),
                birth_time=current_step
            )
            self.world_state.targets.append(target_a)
            self.target_a_spawned = True
            self.logger.info(f"Step {current_step}: Target spawned in region A at {SCENARIO_2_TARGET_A_SPAWN_POS}")
        
        # Spawn distraction target in region B at designated time
        if current_step == SCENARIO_2_TARGET_B_SPAWN_TIME and not self.target_b_spawned:
            target_b = TargetState(
                id="scripted_target_B",
                is_alive=True,
                position=np.array(SCENARIO_2_TARGET_B_SPAWN_POS, dtype=float),
                velocity=np.array(SCENARIO_2_TARGET_B_VELOCITY, dtype=float),
                birth_time=current_step
            )
            self.world_state.targets.append(target_b)
            self.target_b_spawned = True
            self.logger.info(f"Step {current_step}: Distraction target spawned in region B at {SCENARIO_2_TARGET_B_SPAWN_POS}")
        
        # Handle additional scripted spawns
        from configs.config import SCENARIO_2_ADDITIONAL_SPAWNS
        for spawn_event in SCENARIO_2_ADDITIONAL_SPAWNS:
            if current_step == spawn_event['time']:
                target_id = f"additional_target_{spawn_event['region']}_{current_step}"
                # Check if this target was already spawned
                existing_target = any(t.id == target_id for t in self.world_state.targets)
                if not existing_target:
                    new_target = TargetState(
                        id=target_id,
                        is_alive=True,
                        position=np.array(spawn_event['pos'], dtype=float),
                        velocity=np.array(spawn_event['vel'], dtype=float),
                        birth_time=current_step
                    )
                    self.world_state.targets.append(new_target)
                    self.logger.info(f"Step {current_step}: Additional target spawned in region {spawn_event['region']} at {spawn_event['pos']}")
    
    def _update_targets(self):
        """Update target states"""
        # Update existing target motion and survival
        surviving_targets = []
        for target in self.world_state.targets:
            # Survival check
            if np.random.random() < TARGET_SURVIVAL_PROBABILITY:
                # Construct current state vector [x, y, vx, vy]
                current_state = np.concatenate([target.position, target.velocity])
                
                # Update target state (motion model)
                new_state = TARGET_TRANSITION_MATRIX @ current_state
                # Add process noise
                noise = np.random.multivariate_normal(
                    np.zeros(TARGET_STATE_DIM), TARGET_PROCESS_NOISE_COV
                )
                new_state += noise
                
                # Boundary check
                new_state[0] = np.clip(new_state[0], 0, MAP_WIDTH)
                new_state[1] = np.clip(new_state[1], 0, MAP_HEIGHT)
                
                # Update target position and velocity
                target.position = new_state[:2]
                target.velocity = new_state[2:]
                surviving_targets.append(target)
        
        self.world_state.targets = surviving_targets
    
    def _generate_observation(self) -> Observation:
        """Generate observation"""
        measurements = []
        agent_pos = self.world_state.agent_state.position
        agent_orientation = self.world_state.agent_state.heading
        
        # Detect real targets
        for target in self.world_state.targets:
            if not target.is_alive:  # Skip dead targets
                continue
                
            if self._is_target_detectable(target, agent_pos, agent_orientation):
                if np.random.random() < SENSOR_PROB_DETECTION:
                    # Generate noisy measurement
                    true_pos = target.position
                    noise = np.random.multivariate_normal(
                        np.zeros(2), SENSOR_MEASUREMENT_NOISE_COV
                    )
                    measured_pos = true_pos + noise
                    
                    measurement = Measurement(
                        value=measured_pos,
                        is_clutter=False,
                        source_target_id=target.id
                    )
                    measurements.append(measurement)
        
        # Generate clutter measurements (limited by FoV, uniformly distributed in sector area; count scaled by FoV area)
        alpha_rad = np.deg2rad(SENSOR_FOV_DEGREES)
        mean_clutter = SENSOR_CLUTTER_RATE * (alpha_rad / (2.0 * np.pi))
        num_clutter = np.random.poisson(mean_clutter)
        
        fov_half_rad = np.deg2rad(SENSOR_FOV_DEGREES) / 2.0
        for _ in range(num_clutter):
            # Angle within current FoV
            angle_offset = np.random.uniform(-fov_half_rad, fov_half_rad)
            angle = agent_orientation + angle_offset
            # Radius uniformly distributed by area: r = R*sqrt(u)
            distance = SENSOR_RANGE * np.sqrt(np.random.uniform(0.0, 1.0))

            clutter_pos = agent_pos + distance * np.array([
                np.cos(angle), np.sin(angle)
            ])

            if 0 <= clutter_pos[0] <= MAP_WIDTH and 0 <= clutter_pos[1] <= MAP_HEIGHT:
                measurement = Measurement(
                    value=clutter_pos,
                    is_clutter=True,
                    source_target_id=None
                )
                measurements.append(measurement)
        
        return Observation(measurements)
    
    def _is_target_detectable(self, target: TargetState, agent_pos: np.ndarray, agent_orientation: float) -> bool:
        """Check if target is detectable"""
        # Distance check
        distance = np.linalg.norm(target.position - agent_pos)
        if distance > SENSOR_RANGE:
            return False
        
        # Field of view check
        target_direction = target.position - agent_pos
        target_angle = np.arctan2(target_direction[1], target_direction[0])
        angle_diff = abs(target_angle - agent_orientation)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Take minimum angle difference
        
        fov_rad = np.deg2rad(SENSOR_FOV_DEGREES / 2)
        return angle_diff <= fov_rad
    
    def _compute_atomic_proposition_probabilities(self) -> Dict[str, float]:
        """Calculate atomic proposition satisfaction probabilities"""
        probs = {}
        
        # ap_exist_A: At least one target exists in patrol region A
        probs['ap_exist_A'] = self._prob_exist_in_region(self.pmbm_belief, SCENARIO_2_REGION_A)
        
        # ap_loc_any_A: Any tracked target position estimate is in patrol region A
        probs['ap_loc_any_A'] = self._prob_loc_any_in_region(self.pmbm_belief, SCENARIO_2_REGION_A)
        
        # Debug logging
        current_step = self.world_state.time_step
        if current_step <= 10 or current_step % 10 == 0:
            self.logger.info(f"[BRIDGE] t={current_step} ap_exist_A={probs['ap_exist_A']:.4f} ap_loc_any_A={probs['ap_loc_any_A']:.4f}")
        
        return probs
    
    def _prob_exist_in_region(self, belief: PMBMBelief, region: List[float]) -> float:
        """Calculate probability that at least one target exists in the region"""
        map_hyp = belief.get_map_hypothesis()
        if map_hyp is None:
            return 0.0
        
        # Get current agent state for FoV check
        agent_pos = self.world_state.agent_state.position
        agent_orientation = self.world_state.agent_state.heading
        
        p_none = 1.0
        
        for comp in map_hyp.bernoulli_components:
            # Calculate probability integral P_{i,R} of this Bernoulli component in region R
            P_i_R = self._compute_gaussian_region_prob(comp, region)
            if P_i_R > 1e-8:  # Lower threshold to allow more components to contribute
                p_none *= (1.0 - float(comp.existence_prob) * P_i_R)
        
        # PPP term: prior existence intensity of region
        lam0 = float(PPP_DENSITY)
        # Calculate region area
        area_R = float((region[1] - region[0]) * (region[3] - region[2]))
        Lambda_R = max(0.0, lam0 * area_R)
        
        # Final probability
        p_exist = 1.0 - np.exp(-Lambda_R) * p_none
        
        return float(np.clip(p_exist, 0.0, 1.0))
    
    def _prob_loc_any_in_region(self, belief: PMBMBelief, region: List[float]) -> float:
        """Calculate probability that any tracked target's position estimate is in the region"""
        map_hyp = belief.get_map_hypothesis()
        if map_hyp is None:
            return 0.0
        
        max_prob = 0.0
        
        # Find the maximum localization probability among all components in the region
        for comp in map_hyp.bernoulli_components:
            if comp.existence_prob > 0.3:  # Only consider moderate-confidence components
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
    
    def _update_patrol_duty_state(self, ap_probs: Dict[str, float]) -> Dict[str, Any]:
        """Update patrol duty state and check for violations"""
        current_step = self.world_state.time_step
        
        # Check if patrol duty should be activated
        if not self.patrol_duty_active and ap_probs['ap_exist_A'] > LTL_EXIST_THRESHOLD:
            self.patrol_duty_active = True
            self.patrol_duty_start_time = current_step
            self.stats['duty_activations'] += 1
            self.logger.info(f"Patrol duty activated at step {current_step}! P(exist_A) = {ap_probs['ap_exist_A']:.3f}")
        
        # Check if patrol duty is completed
        if self.patrol_duty_active and not self.patrol_duty_completed:
            if ap_probs['ap_loc_any_A'] > LTL_LOC_THRESHOLD:
                self.patrol_duty_completed = True
                self.stats['duty_completions'] += 1
                completion_time = current_step - self.patrol_duty_start_time
                self.logger.info(f"Patrol duty completed at step {current_step}! P(loc_A) = {ap_probs['ap_loc_any_A']:.3f}")
                
                # Reset for next duty cycle
                self.patrol_duty_active = False
                self.patrol_duty_completed = False
                self.patrol_duty_start_time = None
        
        # Check for LTL violation (timeout)
        if self.patrol_duty_active and not self.patrol_duty_completed:
            duty_duration = current_step - self.patrol_duty_start_time
            if duty_duration >= LTL_FINALLY_GRACE_PERIOD:
                self.ltl_violation = True
                self.stats['ltl_violation'] = True
                self.stats['violation_time'] = current_step
                self.logger.info(f"LTL violation! Patrol duty timed out after {duty_duration} steps")
        
        return {
            'patrol_duty_active': self.patrol_duty_active,
            'duty_start_time': self.patrol_duty_start_time,
            'duty_completed': self.patrol_duty_completed,
            'ltl_violation': self.ltl_violation
        }
    
    def _check_task_completion(self) -> Tuple[bool, bool]:
        """Check if task is completed or violated with stricter success criteria"""
        current_step = self.world_state.time_step
        
        # Task fails if LTL is violated
        if self.ltl_violation:
            self.stats['task_completed'] = False
            return False, True
        
        # Task succeeds if we reach the end without violation AND meet quality criteria
        if current_step >= SIMULATION_TIME_STEPS - 1:
            if not self.ltl_violation:
                # 分层成功标准：基础要求 + 优秀表现
                duty_completions = self.stats.get('duty_completions', 0)
                patrol_efficiency = self.stats.get('patrol_efficiency', 0.0)
                
                # 基础成功标准：至少完成5次任务，效率75%以上
                basic_min_completions = 5
                basic_min_efficiency = 0.75
                
                # 优秀成功标准：至少完成12次任务，效率90%以上  
                excellent_min_completions = 12
                excellent_min_efficiency = 0.90
                
                if (duty_completions >= excellent_min_completions and 
                    patrol_efficiency >= excellent_min_efficiency):
                    self.stats['task_completed'] = True
                    self.stats['completion_time'] = current_step
                    self.stats['performance_level'] = 'excellent'
                    self.logger.info(f"Patrol and respond task excellently completed! "
                                   f"Completions: {duty_completions}, Efficiency: {patrol_efficiency:.3f}")
                    return True, False
                elif (duty_completions >= basic_min_completions and 
                      patrol_efficiency >= basic_min_efficiency):
                    self.stats['task_completed'] = True
                    self.stats['completion_time'] = current_step
                    self.stats['performance_level'] = 'basic'
                    self.logger.info(f"Patrol and respond task basically completed! "
                                   f"Completions: {duty_completions}, Efficiency: {patrol_efficiency:.3f}")
                    return True, False
                else:
                    self.stats['task_completed'] = False
                    self.stats['performance_level'] = 'failed'
                    self.logger.info(f"Task failed quality check - Completions: {duty_completions}, "
                                   f"Efficiency: {patrol_efficiency:.3f}")
                    return False, False
        
        return False, False
    
    def _check_temptation_resistance(self, action: int, duty_status: Dict[str, Any]):
        """Check if agent resists temptation to go to region B when duty is active"""
        if not duty_status['patrol_duty_active']:
            return  # No duty active, no temptation to resist
        
        # Calculate agent's intended direction based on action
        agent_pos = self.world_state.agent_state.position
        action_angle = AGENT_CONTROL_ANGLES[action]
        intended_direction = np.array([np.cos(np.radians(action_angle)), 
                                     np.sin(np.radians(action_angle))])
        
        # Check if agent is moving towards region B (right side of map)
        region_b_center = np.array([850.0, 500.0])  # Center of region B
        to_region_b = region_b_center - agent_pos
        to_region_b_normalized = to_region_b / (np.linalg.norm(to_region_b) + 1e-6)
        
        # Calculate dot product to see if moving towards B
        dot_product = np.dot(intended_direction, to_region_b_normalized)
        
        # If moving towards B (dot product > 0.5 means roughly same direction)
        if dot_product > 0.5:
            self.stats['temptation_resistance'] += 1
            self.logger.debug(f"Temptation detected: agent moving towards region B during active duty (dot={dot_product:.3f})")
    
    def _update_statistics(self, observation: Observation, ap_probs: Dict[str, float], duty_status: Dict[str, Any]):
        """Update statistics"""
        # Record trajectory
        self.stats['trajectory'].append({
            'time_step': self.world_state.time_step,
            'agent_position': self.world_state.agent_state.position.copy(),
            'num_targets': len(self.world_state.targets),
            'num_measurements': len(observation.measurements)
        })
        
        # Record atomic proposition probability history
        self.stats['ap_probability_history'].append({
            'time_step': self.world_state.time_step,
            'ap_exist_A': ap_probs['ap_exist_A'],
            'ap_loc_any_A': ap_probs['ap_loc_any_A'],
            'patrol_duty_active': duty_status['patrol_duty_active']
        })
        
        # Record detection statistics
        self.stats['total_detections'] += len([m for m in observation.measurements if not m.is_clutter])
        self.stats['total_false_alarms'] += len([m for m in observation.measurements if m.is_clutter])
        
        # Update patrol efficiency (completion rate)
        if self.stats['duty_activations'] > 0:
            self.stats['patrol_efficiency'] = self.stats['duty_completions'] / self.stats['duty_activations']
        else:
            self.stats['patrol_efficiency'] = 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        return self.stats.copy()
    
    def reset(self):
        """Reset environment"""
        self.world_state = None
        self.pmbm_belief = None
        self.current_ltl_state = None
        
        # Reset patrol duty state
        self.patrol_duty_active = False
        self.patrol_duty_start_time = None
        self.patrol_duty_completed = False
        self.ltl_violation = False
        
        # Reset scripted events
        self.target_a_spawned = False
        self.target_b_spawned = False
        
        # Reset statistics
        self.stats = {
            'task_completed': False,
            'completion_time': None,
            'performance_level': 'failed',
            'ltl_violation': False,
            'violation_time': None,
            'patrol_efficiency': 0,
            'duty_activations': 0,
            'duty_completions': 0,
            'temptation_resistance': 0,
            'total_detections': 0,
            'total_false_alarms': 0,
            'trajectory': [],
            'belief_history': [],
            'ap_probability_history': []
        }
