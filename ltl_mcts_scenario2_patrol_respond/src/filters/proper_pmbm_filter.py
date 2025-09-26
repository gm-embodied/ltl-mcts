"""
Proper PMBM filter implementation
Key: correct covariance convergence process, avoid critical effect
"""

import numpy as np
from typing import List, Optional, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.data_structures import BernoulliComponent, GlobalHypothesis, PMBMBelief, Observation
# Get DEBUG_BRIDGE configuration
def _get_debug_bridge():
    try:
        from configs.config import DEBUG_BRIDGE
        return DEBUG_BRIDGE
    except ImportError:
        return False

class ProperPMBMFilter:
    """Proper PMBM filter implementation"""
    
    def __init__(self):
        # Motion model parameters
        self.F = None  # 状态转移矩阵
        self.Q = None  # 过程噪声协方差
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 观测矩阵
        self.R = None  # 观测噪声协方差
        
        # PMBM parameters (optimize tracking performance, reduce OSPA)
        self.survival_prob = 0.995   # 提高存活概率，维持航迹
        self.detection_prob = 0.9   # 提高检测概率，匹配配置
        self.gating_threshold = 50.0  # 放宽门控，避免丢失关联
        self.prune_threshold = 0.3  # 降低剪枝阈值，保留更多航迹
        self.max_components = 100    # 增加分量数，保持更多跟踪
        
        # Newborn target parameters
        self.birth_weight = 0.01  # 降低新生目标权重，减少假目标
        self.clutter_rate = 0.5   # 匹配配置文件中的杂波率
        
    def set_motion_model(self, F: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """Set motion model parameters"""
        self.F = F
        self.Q = Q
        self.R = R
        if _get_debug_bridge():
            print(f"[PMBM] Set motion model: F={F.shape}, Q={Q.shape}, R={R.shape}")
    
    def predict(self, belief: PMBMBelief) -> PMBMBelief:
        """PMBM prediction step"""
        new_belief = PMBMBelief()
        new_belief.timestamp = belief.timestamp + 1
        new_belief.poisson_component = belief.poisson_component
        
        # Get current MAP hypothesis
        current_hyp = belief.get_map_hypothesis()
        if current_hyp is None:
            # No existing hypothesis, create empty hypothesis
            empty_hyp = GlobalHypothesis(weight=1.0, bernoulli_components=[])
            new_belief.global_hypotheses = [empty_hyp]
            return new_belief
        
        # Predict each Bernoulli component
        predicted_components = []
        for comp in current_hyp.bernoulli_components:
            if comp.existence_prob < self.prune_threshold:
                continue
                
            # Predict state distribution (standard Kalman prediction)
            predicted_mean = self.F @ comp.state_mean
            predicted_cov = self.F @ comp.state_cov @ self.F.T + self.Q
            
            # Predict existence probability (consider survival probability)
            predicted_existence = self.survival_prob * comp.existence_prob
            
            predicted_comp = BernoulliComponent(
                existence_prob=predicted_existence,
                state_mean=predicted_mean,
                state_cov=predicted_cov,
                label=comp.label
            )
            predicted_components.append(predicted_comp)
        
        # Create predicted hypothesis
        predicted_hyp = GlobalHypothesis(
            weight=1.0,
            bernoulli_components=predicted_components
        )
        
        new_belief.global_hypotheses = [predicted_hyp]
        return new_belief
    
    def update(self, belief: PMBMBelief, observation: Observation) -> PMBMBelief:
        """PMBM update step"""
        updated_belief = PMBMBelief()
        updated_belief.timestamp = belief.timestamp
        updated_belief.poisson_component = belief.poisson_component
        
        # Get current hypothesis
        current_hyp = belief.get_map_hypothesis()
        if current_hyp is None:
            current_components = []
        else:
            current_components = current_hyp.bernoulli_components.copy()
        
        # Process observation
        measurements_array = observation.get_measurement_array()
        
        if len(measurements_array) == 0:
            # No observation, only update existence probability (consider missed detection)
            for comp in current_components:
                comp.existence_prob *= (1.0 - self.detection_prob)
        else:
            # Implement improved PMBM data association
            current_components = self._improved_pmbm_data_association(current_components, measurements_array)
        
        # Prune and limit
        current_components = self._prune_and_limit_components(current_components)
        
        # Create updated hypothesis
        updated_hyp = GlobalHypothesis(
            weight=1.0,
            bernoulli_components=current_components
        )
        
        updated_belief.global_hypotheses = [updated_hyp]
        return updated_belief
    
    def _improved_pmbm_data_association(self, components: List[BernoulliComponent], 
                                       measurements: np.ndarray) -> List[BernoulliComponent]:
        """Improved PMBM data association (more theoretical version)"""
        updated_components = []
        
        # Compute association probability matrix for all components and measurements
        association_matrix = self._compute_association_matrix(components, measurements)
        
        # Use Hungarian algorithm or greedy algorithm for optimal assignment
        assignments = self._solve_assignment_problem(association_matrix, components, measurements)
        
        # Process assignment results
        assigned_measurements = set()
        for comp_idx, meas_idx in assignments:
            if meas_idx >= 0:  # Valid assignment
                meas = measurements[meas_idx]
                assigned_measurements.add(meas_idx)
                updated_comp = self._update_component_with_measurement(components[comp_idx], meas)
                updated_components.append(updated_comp)
            else:  # Missed detection
                missed_comp = self._update_component_missed_detection(components[comp_idx])
                if missed_comp.existence_prob > self.prune_threshold:
                    updated_components.append(missed_comp)
        
        # Create new components for unassigned measurements
        for i, meas in enumerate(measurements):
            if i not in assigned_measurements:
                new_comp = self._create_new_component_proper(meas)
                updated_components.append(new_comp)
        
        return updated_components
    
    def _compute_association_matrix(self, components: List[BernoulliComponent], 
                                  measurements: np.ndarray) -> np.ndarray:
        """Compute association probability matrix"""
        n_components = len(components)
        n_measurements = len(measurements)
        
        if n_components == 0 or n_measurements == 0:
            return np.array([])
        
        # Association log likelihood matrix
        log_likelihood_matrix = np.full((n_components, n_measurements), -np.inf)
        
        for i, comp in enumerate(components):
            for j, meas in enumerate(measurements):
                log_likelihood = self._compute_association_likelihood(comp, meas)
                # Gating check
                if log_likelihood > -self.gating_threshold:
                    log_likelihood_matrix[i, j] = log_likelihood
        
        return log_likelihood_matrix
    
    def _solve_assignment_problem(self, log_likelihood_matrix: np.ndarray, 
                                components: List[BernoulliComponent], 
                                measurements: np.ndarray) -> List[Tuple[int, int]]:
        """Solve assignment problem (simplified greedy method)"""
        if log_likelihood_matrix.size == 0:
            return [(i, -1) for i in range(len(components))]  # All components missed detection
        
        assignments = []
        used_measurements = set()
        
        # Sort components by existence probability, prioritize high probability components
        sorted_indices = sorted(range(len(components)), 
                              key=lambda i: components[i].existence_prob, reverse=True)
        
        for comp_idx in sorted_indices:
            best_meas_idx = -1
            best_log_likelihood = -np.inf
            
            for meas_idx in range(log_likelihood_matrix.shape[1]):
                if meas_idx in used_measurements:
                    continue
                
                log_likelihood = log_likelihood_matrix[comp_idx, meas_idx]
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_meas_idx = meas_idx
            
            if best_meas_idx >= 0 and best_log_likelihood > -np.inf:
                assignments.append((comp_idx, best_meas_idx))
                used_measurements.add(best_meas_idx)
            else:
                assignments.append((comp_idx, -1))  # Missed detection
        
        return assignments
    
    def _compute_association_likelihood(self, component: BernoulliComponent, 
                                      measurement: np.ndarray) -> float:
        """Compute association log likelihood between measurement and component"""
        try:
            # Predict measurement
            predicted_meas = self.H @ component.state_mean
            innovation = measurement - predicted_meas
            
            # Innovation covariance
            S = self.H @ component.state_cov @ self.H.T + self.R
            
            # Compute Mahalanobis distance
            mahal_dist = innovation.T @ np.linalg.inv(S) @ innovation
            
            # Log likelihood (half of negative Mahalanobis distance)
            log_likelihood = -0.5 * mahal_dist
            
            return log_likelihood
            
        except np.linalg.LinAlgError:
            return -np.inf
    
    def _update_component_with_measurement(self, component: BernoulliComponent, 
                                         measurement: np.ndarray) -> BernoulliComponent:
        """Update Bernoulli component with measurement - use standard PMBM Bayesian update"""
        # Create new component to avoid modifying original component
        updated_comp = BernoulliComponent(
            existence_prob=component.existence_prob,
            state_mean=component.state_mean.copy(),
            state_cov=component.state_cov.copy(),
            label=component.label
        )
        
        # Standard Kalman update
        predicted_meas = self.H @ updated_comp.state_mean
        innovation = measurement - predicted_meas
        
        # Innovation covariance
        S = self.H @ updated_comp.state_cov @ self.H.T + self.R
        
        try:
            # Kalman gain
            K = updated_comp.state_cov @ self.H.T @ np.linalg.inv(S)
            
            # Record covariance trace before update
            old_trace = np.trace(updated_comp.state_cov[:2, :2])
            
            # Update state estimate
            updated_comp.state_mean = updated_comp.state_mean + K @ innovation
            
            # Use Joseph form to ensure numerical stability and better convergence
            I_KH = np.eye(4) - K @ self.H
            updated_comp.state_cov = I_KH @ updated_comp.state_cov @ I_KH.T + K @ self.R @ K.T
            
            # Record updated covariance trace
            new_trace = np.trace(updated_comp.state_cov[:2, :2])
            
            # Debug output: show covariance convergence process (controlled by DEBUG switch)
            if _get_debug_bridge():
                print(f"[PMBM DEBUG] 卡尔曼更新: {updated_comp.label} 协方差迹 {old_trace:.1f} -> {new_trace:.1f}")
            
            # Improved existence probability update, enhance tracking stability
            likelihood = self._compute_gaussian_likelihood(innovation, S)
            
            # Use smaller surveillance area to calculate clutter density, improve detection update effect
            # Use sensor coverage area instead of the whole map
            sensor_area = np.pi * (220.0 ** 2) * (60.0 / 360.0)  # 传感器扇形区域
            clutter_density = self.clutter_rate / sensor_area
            
            # Standard Bayesian update: P(existence|detection) 
            r = updated_comp.existence_prob
            numerator = r * self.detection_prob * likelihood
            denominator = numerator + (1 - r) * clutter_density
            
            if denominator > 0:
                # More aggressive existence probability update,有助于稳定跟踪
                updated_prob = numerator / denominator
                updated_comp.existence_prob = min(0.995, updated_prob)
            else:
                # Numerical stability: if denominator is 0,适度提升概率
                updated_comp.existence_prob = min(0.95, r * 1.1)
            
        except np.linalg.LinAlgError:
            # Numerical problem, keep original state but reduce existence probability
            updated_comp.existence_prob *= 0.5
        
        # Ensure covariance is positive definite
        updated_comp.state_cov = self._ensure_positive_definite(updated_comp.state_cov)
        
        return updated_comp
    
    def _update_component_missed_detection(self, component: BernoulliComponent) -> BernoulliComponent:
        """Process missed detection component - use standard PMBM Bayesian update"""
        # Standard PMBM missed detection update: P(existence|undetected)
        r = component.existence_prob
        
        # Bayesian update formula
        numerator = r * (1 - self.detection_prob)
        denominator = numerator + (1 - r)  # (1-r) * P(undetected|existence) = (1-r) * 1
        
        if denominator > 0:
            new_existence_prob = numerator / denominator
        else:
            # Numerical stability
            new_existence_prob = r * 0.5
        
        missed_comp = BernoulliComponent(
            existence_prob=max(0.01, new_existence_prob),  # Set minimum existence probability to prevent complete disappearance
            state_mean=component.state_mean.copy(),
            state_cov=component.state_cov.copy(),
            label=component.label
        )
        return missed_comp
    
    def _create_new_component_proper(self, measurement: np.ndarray) -> BernoulliComponent:
        """Create new Bernoulli component (correct initial uncertainty, accelerate convergence)"""
        # Initial state: position from measurement, velocity is zero
        initial_mean = np.array([measurement[0], measurement[1], 0.0, 0.0])
        
        # Fix: use reasonable initial uncertainty, ensure fast convergence below confirmation threshold
        measurement_var = np.diag(self.R)[0]  # Measurement variance
        
        # Adjust initial position covariance: use smaller multiplier 
        # Target: after 2-3 observations, covariance trace can 
        initial_position_var = measurement_var * 5.0   
        initial_velocity_var = 25.0   
        
        initial_cov = np.diag([
            initial_position_var,   # Var(x) = 5 * 100 = 500
            initial_position_var,   # Var(y) = 5 * 100 = 500  
            initial_velocity_var,   # Var(vx) = 25
            initial_velocity_var    # Var(vy) = 25
        ])
        
        # Debug information (controlled by DEBUG switch)
        if _get_debug_bridge():
            initial_trace = initial_position_var * 2
            print(f"[PMBM DEBUG] New component: measurement={measurement}, initial position variance={initial_position_var}, initial trace={initial_trace:.1f}")
        
        return BernoulliComponent(
            existence_prob=0.9,  # Further increase initial existence probability, ensure tracking can be maintained
            state_mean=initial_mean,
            state_cov=initial_cov,
            label=f"track_{np.random.randint(100000)}"
        )
    
    def _compute_gaussian_likelihood(self, innovation: np.ndarray, S: np.ndarray) -> float:
        """Compute standard Gaussian likelihood density"""
        try:
            det_S = np.linalg.det(S)
            if det_S <= 0:
                return 1e-10  # Return very small positive value instead of 1.0
            
            # Standard Gaussian likelihood density  
            mahal_dist = innovation.T @ np.linalg.inv(S) @ innovation
            likelihood = np.exp(-0.5 * mahal_dist) / np.sqrt((2*np.pi)**len(innovation) * det_S)
            
            #  
            return max(1e-10, likelihood)
            
        except (np.linalg.LinAlgError, OverflowError):
            return 1e-10
    
    def _prune_and_limit_components(self, components: List[BernoulliComponent]) -> List[BernoulliComponent]:
        """Prune and limit component数量"""
        # Prune low existence probability components
        pruned = [comp for comp in components if comp.existence_prob > self.prune_threshold]
        
        # Sort by existence probability and limit number
        if len(pruned) > self.max_components:
            pruned.sort(key=lambda x: x.existence_prob, reverse=True)
            pruned = pruned[:self.max_components]
        
        return pruned
    
    def _ensure_positive_definite(self, cov: np.ndarray, min_eigenval: float = 1e-6) -> np.ndarray:
        """Ensure covariance matrix is positive definite"""
        try:
            # Eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            
            # Set negative eigenvalues to minimum value
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct covariance matrix
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
        except np.linalg.LinAlgError:
            # If decomposition fails, return diagonal matrix
            return np.eye(cov.shape[0]) * min_eigenval
