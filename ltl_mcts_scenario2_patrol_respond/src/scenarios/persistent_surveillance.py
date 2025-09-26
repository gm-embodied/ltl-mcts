"""
Scenario 1: Persistent Surveillance Task
LTL Formula: G(F(ap_exist_A)) & G(F(ap_exist_B))
Implementation of complete simulation environment, timer mechanism and main simulation loop
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
from src.utils.ltl_window import WindowState, update_window_state
from src.filters.proper_pmbm_filter import ProperPMBMFilter
from src.utils.data_structures import (
    AgentState, TargetState, WorldState, Observation, Measurement,
    PMBMBelief, AtomicProposition, LTLState, create_initial_pmbm_belief
)

class PersistentSurveillanceEnvironment:
    """Scenario 1: Persistent Surveillance Task Simulation Environment"""
    
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
        
        # Persistent surveillance task timers
        self.timer_A = 0  # Region A timer
        self.timer_B = 0  # Region B timer
        self.ltl_violation = False  # LTL violation flag
        
        # Statistics
        self.stats = {
            'task_completed': False,
            'completion_time': None,
            'ltl_violation': False,
            'violation_time': None,
            'patrol_efficiency': 0,  # LTL refresh count
            'total_detections': 0,
            'total_false_alarms': 0,
            'trajectory': [],
            'belief_history': [],
            'ap_probability_history': [],
            'timer_A_history': [],
            'timer_B_history': []
        }
        
        # PMBM filter
        self.pmbm_filter = None
    
    def initialize(self) -> Tuple[WorldState, PMBMBelief, LTLState]:
        """Initialize environment"""
        self.logger.info("Initializing persistent surveillance scenario...")
        
        # Initialize agent state
        agent_state = AgentState(
            position=np.array(AGENT_INITIAL_POSITION, dtype=float),
            velocity=np.array([0.0, 0.0]),
            heading=0.0
        )
        
        # Initialize target list (fix zero target dilemma)
        targets = []
        
        # Fix zero target dilemma: place initial targets according to configuration
        initial_count = TARGET_INITIAL_COUNT
        placement_strategy = INITIAL_TARGET_PLACEMENT
        
        if initial_count > 0 and placement_strategy == 'regions':
            # Place half targets in each region
            targets_per_region = max(1, initial_count // 2)
            
            # Place targets in region A
            for i in range(targets_per_region):
                x = np.random.uniform(REGION_A[0], REGION_A[1])
                y = np.random.uniform(REGION_A[2], REGION_A[3])
                vx = np.random.normal(0, 2.0)
                vy = np.random.normal(0, 2.0)
                targets.append(TargetState(
                    id=f"init_A_{i}", is_alive=True,
                    position=np.array([x, y]),
                    velocity=np.array([vx, vy]), birth_time=0
                ))
            
            # Place targets in region B
            for i in range(initial_count - targets_per_region):
                x = np.random.uniform(REGION_B[0], REGION_B[1])
                y = np.random.uniform(REGION_B[2], REGION_B[3])
                vx = np.random.normal(0, 2.0)
                vy = np.random.normal(0, 2.0)
                targets.append(TargetState(
                    id=f"init_B_{i}", is_alive=True,
                    position=np.array([x, y]),
                    velocity=np.array([vx, vy]), birth_time=0
                ))
            
            self.logger.info(f"[INIT] Placed {len(targets)} initial targets in A/B regions to avoid zero target dilemma")
        
        elif initial_count > 0 and placement_strategy == 'random':
            # Randomly place initial targets
            for i in range(initial_count):
                x = np.random.uniform(0, MAP_WIDTH)
                y = np.random.uniform(0, MAP_HEIGHT)
                vx = np.random.normal(0, 2.0)
                vy = np.random.normal(0, 2.0)
                targets.append(TargetState(
                    id=f"init_rand_{i}", is_alive=True,
                    position=np.array([x, y]),
                    velocity=np.array([vx, vy]), birth_time=0
                ))
            self.logger.info(f"[INIT] Randomly placed {initial_count} initial targets")
        
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
            memory_state={'ap_exist_A': False, 'ap_exist_B': False}
        )
        
        # ========== Windowed persistent surveillance parameters and state ==========
        self.W = LTL_WINDOW_STEPS
        self.on_th = AP_EXIST_PROB_ON
        self.off_th = AP_EXIST_PROB_OFF
        
        # Ensure initial state: all ap OFF, not armed, timers at 0
        self.ws = WindowState(t_A=0, t_B=0, apA_on=False, apB_on=False,
                              confirm_count_A=0, confirm_count_B=0,
                              armed_A=False, armed_B=False)
        
        self.logger.info(f"[INIT] Windowed parameters: W={self.W}, on_th={self.on_th:.2f}, off_th={self.off_th:.2f}")
        self.logger.info(f"[INIT] Initial state: apA/apB={self.ws.apA_on}/{self.ws.apB_on}, armed_A/B={self.ws.armed_A}/{self.ws.armed_B}")
        self.max_interval_A = 0
        self.max_interval_B = 0
        self.last_reset_step_A = -1  # Fix: initialize to -1 so first reset can correctly calculate interval
        self.last_reset_step_B = -1
        self.timeout_steps = 0
        
        # Reset timers and statistics (compatible with old fields, but no longer used for criteria)
        self.timer_A = 0
        self.timer_B = 0
        self.ltl_violation = False
        
        self.logger.info(f"Scenario initialization complete - Agent position: {agent_state.position}")
        return self.world_state, self.pmbm_belief, self.current_ltl_state
    
    def step(self, action: int, planner_name: str = "Unknown") -> Tuple[WorldState, Observation, PMBMBelief, LTLState, bool, Dict[str, Any]]:
        """Execute one simulation step"""
        if self.ltl_violation:
            return self.world_state, Observation([]), self.pmbm_belief, self.current_ltl_state, True, {
                'p_exist_A': 0.0, 'p_exist_B': 0.0, 'resetA': False, 'resetB': False,
                't_A': self.ws.t_A, 't_B': self.ws.t_B, 'apA_on': self.ws.apA_on,
                'apB_on': self.ws.apB_on, 'violate': True
            }
        
        # 1. Execute agent action
        self._execute_agent_action(action)
        
        # 2. Update target states (including new targets)
        self._update_targets()
        
        # 3. Generate observation
        observation = self._generate_observation()
        
        # 4. Update PMBM belief
        # First predict
        self.pmbm_belief = self.pmbm_filter.predict(self.pmbm_belief)
        # Then update
        self.pmbm_belief = self.pmbm_filter.update(self.pmbm_belief, observation)
        
        # 5. Calculate regional existence probabilities (RFS bridge)
        pA = self._prob_exist_in_region(self.pmbm_belief, REGION_A)
        pB = self._prob_exist_in_region(self.pmbm_belief, REGION_B)
        
        # 6. Update windowed state (copy previous state for interval statistics)
        current_step = self.world_state.time_step
        prev_ws = WindowState(**vars(self.ws))
        prev_apA_on = prev_ws.apA_on
        prev_apB_on = prev_ws.apB_on
        
        self.ws, resetA, resetB, violate = update_window_state(
            self.ws, pA, pB, self.W, self.on_th, self.off_th
        )
        
        # Key debug log - detailed print for first 5 steps
        if current_step <= 5 or DEBUG_BRIDGE:
            self.logger.info(f"[CHECK] k={current_step} P_A={pA:.2f} P_B={pB:.2f} "
                           f"prev_on(A/B)={prev_apA_on}/{prev_apB_on} -> new_on(A/B)={self.ws.apA_on}/{self.ws.apB_on} "
                           f"armed(A/B)={self.ws.armed_A}/{self.ws.armed_B} tA/tB={self.ws.t_A}/{self.ws.t_B} violate={violate}")
        
        # Debug output - print LTL status every 10 steps
        elif current_step % 10 == 0:
            self.logger.info(f"[LTL] step={current_step} pA/pB={pA:.2f}/{pB:.2f} "
                           f"apA/apB={self.ws.apA_on}/{self.ws.apB_on} "
                           f"tA/tB={self.ws.t_A}/{self.ws.t_B}")
        
        # Debug output - print immediately on violation
        if violate:
            self.logger.info(f"[VIOLATE] step={current_step} violate={violate}")
        
        # Sync ap existence state to LTL memory (for planner to read)
        self.current_ltl_state.memory_state['ap_exist_A'] = self.ws.apA_on
        self.current_ltl_state.memory_state['ap_exist_B'] = self.ws.apB_on
        
        # 7. Record maximum confirmation interval (based on reset events)
        # Note: current_step is the time step before this step execution, actual reset occurs at current_step+1
        actual_step = current_step + 1  # Fix timing: actual step when reset occurs
        if resetA:
            # Fix: check last_reset_step value before update
            old_last_reset_A = self.last_reset_step_A
            if old_last_reset_A >= 0:  
                # Not the first reset, calculate interval from last reset
                interval = actual_step - old_last_reset_A
                self.max_interval_A = max(self.max_interval_A, interval)
            else:
                # First reset, calculate interval from episode start
                interval = actual_step  # actual_step is the interval from step 0
                self.max_interval_A = max(self.max_interval_A, interval)
            self.last_reset_step_A = actual_step
        if resetB:
            # Fix: check last_reset_step value before update
            old_last_reset_B = self.last_reset_step_B
            if old_last_reset_B >= 0:  
                # Not the first reset, calculate interval from last reset
                interval = actual_step - old_last_reset_B
                self.max_interval_B = max(self.max_interval_B, interval)
            else:
                # First reset, calculate interval from episode start
                interval = actual_step  # actual_step is the interval from step 0
                self.max_interval_B = max(self.max_interval_B, interval)
            self.last_reset_step_B = actual_step
        
        # 8. Violation and termination
        if violate:
            self.timeout_steps += 1
            self.ltl_violation = True
            self.stats['ltl_violation'] = True
            self.stats['violation_time'] = current_step
            is_done = True
            task_completed = False
        else:
            is_done = (current_step >= SIMULATION_TIME_STEPS - 1)
            task_completed = (is_done and not violate)
        
        # 9. Update statistics
        self._update_statistics_window(observation, pA, pB)
        
        # 10. If terminated, write completion flag and time
        if is_done:
            self.stats['task_completed'] = bool(task_completed)
            if task_completed:
                self.stats['completion_time'] = current_step

        # 11. Increment time step
        self.world_state.time_step += 1
        
        # 12. info return
        info = {
            'p_exist_A': pA,
            'p_exist_B': pB,
            'resetA': resetA,
            'resetB': resetB,
            't_A': self.ws.t_A,
            't_B': self.ws.t_B,
            'apA_on': self.ws.apA_on,
            'apB_on': self.ws.apB_on,
            'violate': violate,
            # Debug information
            'lambda_A': getattr(self, '_lambda_A', 0.0),
            'lambda_B': getattr(self, '_lambda_B', 0.0),
            'bern_prod_A': getattr(self, '_bern_prod_A', 1.0),
            'bern_prod_B': getattr(self, '_bern_prod_B', 1.0),
            'fov_comps_A': getattr(self, '_fov_comps_A', 'N/A'),
            'fov_comps_B': getattr(self, '_fov_comps_B', 'N/A')
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
    
    def _update_targets(self):
        """Update target states, including new targets and existing target motion"""
        # 1. Existing target motion and survival
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
        
        # 2. New targets (regional birth)
        nA = np.random.poisson(REGIONAL_BIRTHS_PER_STEP_A)
        nB = np.random.poisson(REGIONAL_BIRTHS_PER_STEP_B)
        nBG = np.random.poisson(BACKGROUND_BIRTHS_PER_STEP)
        
        # Debug output - print regional birth counts every 10 steps
        if self.world_state.time_step % 10 == 0:
            self.logger.info(f"[BIRTH] step={self.world_state.time_step} A={nA} B={nB} BG={nBG}")
        
        for _ in range(nA):
            self._spawn_target_in_rect(REGION_A)
        for _ in range(nB):
            self._spawn_target_in_rect(REGION_B)
        for _ in range(nBG):
            self._spawn_target_in_background()
    
    def _spawn_target_in_rect(self, region: List[float]):
        x = np.random.uniform(region[0], region[1])
        y = np.random.uniform(region[2], region[3])
        vx = np.random.normal(0, 2.0)
        vy = np.random.normal(0, 2.0)
        new_target = TargetState(
            id=f"target_{self.world_state.time_step}_{len(self.world_state.targets)}",
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            is_alive=True,
            birth_time=self.world_state.time_step
        )
        self.world_state.targets.append(new_target)

    def _spawn_target_in_background(self):
        while True:
            x = np.random.uniform(0, MAP_WIDTH)
            y = np.random.uniform(0, MAP_HEIGHT)
            inA = (REGION_A[0] <= x <= REGION_A[1]) and (REGION_A[2] <= y <= REGION_A[3])
            inB = (REGION_B[0] <= x <= REGION_B[1]) and (REGION_B[2] <= y <= REGION_B[3])
            if not (inA or inB):
                break
        vx = np.random.normal(0, 2.0)
        vy = np.random.normal(0, 2.0)
        new_target = TargetState(
            id=f"bg_{self.world_state.time_step}_{len(self.world_state.targets)}",
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            is_alive=True,
            birth_time=self.world_state.time_step
        )
        self.world_state.targets.append(new_target)
    
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
    
    def _prob_exist_in_region(self, belief: PMBMBelief, region: List[float]) -> float:
        """Return joint existence probability of any target in rectangular region (0~1)
        Using independence approximation: P_exist(R) = 1 - exp(-Λ_R) * Π_i (1 - r_i * P_{i,R} * FoV_i)
        where P_{i,R} is the probability integral of i-th Bernoulli component in region R, FoV_i is FoV gating term
        """
        map_hyp = belief.get_map_hypothesis()
        if map_hyp is None:
            return 0.0
        
        # Get current agent state for FoV check
        agent_pos = self.world_state.agent_state.position
        agent_orientation = self.world_state.agent_state.heading
        
        p_none = 1.0
        fov_components = 0
        total_components = 0
        
        for comp in map_hyp.bernoulli_components:
            total_components += 1
            
            # Check if this component is in FoV (for statistics only)
            comp_pos = comp.get_position_mean()
            in_fov = self._is_position_in_fov(comp_pos, agent_pos, agent_orientation)
            
            if in_fov:
                fov_components += 1
            
            # Fix: P_exist calculation should consider all Bernoulli components, not just those in FoV
            # Calculate probability integral P_{i,R} of this Bernoulli component in region R
            P_i_R = self._compute_gaussian_region_prob(comp, region)
            if P_i_R > 1e-8:  # Lower threshold to allow more components to contribute
                p_none *= (1.0 - float(comp.existence_prob) * P_i_R)
        
        # PPP term: prior existence intensity of region
        lam0 = float(PPP_DENSITY)
        # Calculate region area
        area_R = float((region[1] - region[0]) * (region[3] - region[2]))
        # PPP intensity should not be scaled by FoV - this is prior probability, not observation probability
        Lambda_R = max(0.0, lam0 * area_R)
        
        # Final probability
        p_exist = 1.0 - np.exp(-Lambda_R) * p_none
        
        # Store Lambda values for debugging (distinguish A and B regions)
        if region == REGION_A:
            self._lambda_A = Lambda_R
            self._bern_prod_A = p_none
            self._fov_comps_A = f"{fov_components}/{total_components}"
        elif region == REGION_B:
            self._lambda_B = Lambda_R
            self._bern_prod_B = p_none
            self._fov_comps_B = f"{fov_components}/{total_components}"
        
        # Debug output - print P(exist) components for first 5 steps or every 10 steps
        if self.world_state.time_step <= 5 or self.world_state.time_step % 10 == 0:
            region_name = "A" if region == REGION_A else "B"
            self.logger.info(f"[PEXIST] step={self.world_state.time_step} "
                           f"Λ{region_name}={Lambda_R:.4f} "
                           f"BernProd{region_name}={p_none:.3f} "
                           f"FoV_comps={fov_components}/{total_components} "
                           f"P_exist{region_name}={p_exist:.4f}")
        
        return float(np.clip(p_exist, 0.0, 1.0))
    
    def _is_position_in_fov(self, target_pos: np.ndarray, agent_pos: np.ndarray, agent_orientation: float) -> bool:
        """Check if position is in FoV"""
        # Distance check
        distance = np.linalg.norm(target_pos - agent_pos)
        if distance > SENSOR_RANGE:
            return False
        
        # Field of view check
        target_direction = target_pos - agent_pos
        target_angle = np.arctan2(target_direction[1], target_direction[0])
        angle_diff = abs(target_angle - agent_orientation)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Take minimum angle difference
        
        fov_rad = np.deg2rad(SENSOR_FOV_DEGREES / 2)
        return angle_diff <= fov_rad
    
    def _compute_gaussian_region_prob(self, bernoulli_comp, region: List[float]) -> float:
        """Calculate probability integral P_{i,R} of Gaussian distribution in rectangular region"""
        try:
            from scipy.stats import norm
            
            # Get mean and covariance
            mean = bernoulli_comp.get_position_mean()
            cov = bernoulli_comp.get_position_cov()
            
            # Fix: use correct probability integral calculation
            # For rectangular region, probability integral = P(x_min < X < x_max) * P(y_min < Y < y_max)
            # Assume X and Y independent (diagonal covariance matrix)
            
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
    
    def _update_statistics_window(self, observation: Observation, pA: float, pB: float):
        """Minimal change to record windowed metrics and general statistics"""
        # General trajectory and measurement statistics
        self.stats['trajectory'].append({
            'time_step': self.world_state.time_step,
            'agent_position': self.world_state.agent_state.position.copy(),
            'num_targets': len(self.world_state.targets),
            'num_measurements': len(observation.measurements)
        })
        self.stats['total_detections'] += len([m for m in observation.measurements if not m.is_clutter])
        self.stats['total_false_alarms'] += len([m for m in observation.measurements if m.is_clutter])
        
        # Probability and window timer history
        if 'ap_probability_history' in self.stats:
            self.stats['ap_probability_history'].append({
                'time_step': self.world_state.time_step,
                'ap_exist_A': pA,
                'ap_exist_B': pB
            })
        if 'timer_A_history' in self.stats:
            self.stats['timer_A_history'].append(self.ws.t_A)
        if 'timer_B_history' in self.stats:
            self.stats['timer_B_history'].append(self.ws.t_B)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        # Supplement windowed key metrics
        self.stats.update({
            'patrol_efficiency': self.ws.confirm_count_A + self.ws.confirm_count_B,
            'timeout_steps': self.timeout_steps,
            'max_interval_A': self.max_interval_A,
            'max_interval_B': self.max_interval_B
        })
        return self.stats.copy()
    
    def reset(self):
        """Reset environment"""
        self.world_state = None
        self.pmbm_belief = None
        self.current_ltl_state = None
        self.timer_A = 0
        self.timer_B = 0
        self.ltl_violation = False
        
        # Reset statistics
        self.stats = {
            'task_completed': False,
            'completion_time': None,
            'ltl_violation': False,
            'violation_time': None,
            'patrol_efficiency': 0,
            'total_detections': 0,
            'total_false_alarms': 0,
            'trajectory': [],
            'belief_history': [],
            'ap_probability_history': [],
            'timer_A_history': [],
            'timer_B_history': []
        }
