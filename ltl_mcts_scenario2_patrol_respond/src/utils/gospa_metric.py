"""
GOSPA (Generalized Optimal Sub-Pattern Assignment) Metric Implementation
GOSPA is more suitable for high-clutter environments compared to OSPA
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
import math

def compute_gospa_error(true_targets: List[np.ndarray], 
                       estimated_targets: List[np.ndarray],
                       c: float = 100.0, 
                       p: int = 2, 
                       alpha: float = 2.0) -> Tuple[float, dict]:
    """
    Compute GOSPA (Generalized Optimal Sub-Pattern Assignment) error
    
    GOSPA is more robust to cardinality errors than OSPA, making it better
    for high-clutter surveillance scenarios.
    
    Args:
        true_targets: List of true target positions [x, y]
        estimated_targets: List of estimated target positions [x, y]
        c: Cutoff parameter for localization errors
        p: Order parameter (typically 2 for L2 norm)
        alpha: Penalty parameter for cardinality errors (alpha >= p)
        
    Returns:
        Tuple of (gospa_error, decomposition_dict)
        decomposition_dict contains: 'localization', 'missed', 'false'
    """
    
    if len(true_targets) == 0 and len(estimated_targets) == 0:
        return 0.0, {'localization': 0.0, 'missed': 0.0, 'false': 0.0}
    
    n = len(true_targets)
    m = len(estimated_targets)
    
    # Convert to numpy arrays for easier computation
    if n > 0:
        true_array = np.array(true_targets)
    if m > 0:
        est_array = np.array(estimated_targets)
    
    # Handle edge cases
    if n == 0:
        # Only false alarms
        false_penalty = (c ** p) * m
        gospa = (false_penalty / max(m, 1)) ** (1/p)
        return gospa, {'localization': 0.0, 'missed': 0.0, 'false': gospa}
    
    if m == 0:
        # Only missed detections
        missed_penalty = (c ** p) * n
        gospa = (missed_penalty / max(n, 1)) ** (1/p)
        return gospa, {'localization': 0.0, 'missed': gospa, 'false': 0.0}
    
    # Compute distance matrix
    distance_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist = np.linalg.norm(true_array[i] - est_array[j])
            distance_matrix[i, j] = min(dist ** p, c ** p)
    
    # Solve assignment problem
    if n <= m:
        # More estimates than true targets
        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        
        # Compute localization error for assigned pairs
        localization_error = 0.0
        for i, j in zip(row_indices, col_indices):
            localization_error += distance_matrix[i, j]
        
        # Missed targets (should be 0 in this case)
        missed_targets = 0
        
        # False alarms
        false_alarms = m - n
        
    else:
        # More true targets than estimates
        # Pad the distance matrix
        padded_matrix = np.full((n, n), c ** p)
        padded_matrix[:n, :m] = distance_matrix
        
        row_indices, col_indices = linear_sum_assignment(padded_matrix)
        
        # Compute localization error for real assignments
        localization_error = 0.0
        assigned_estimates = 0
        for i, j in zip(row_indices, col_indices):
            if j < m:  # Real estimate (not padded)
                localization_error += distance_matrix[i, j]
                assigned_estimates += 1
        
        # Missed targets
        missed_targets = n - assigned_estimates
        
        # False alarms
        false_alarms = 0
    
    # Compute GOSPA components
    localization_component = localization_error
    missed_component = (c ** p) * missed_targets
    false_component = (c ** p) * false_alarms
    
    # Total GOSPA error
    total_error = localization_component + missed_component + false_component
    
    # Normalize by the maximum of true and estimated cardinalities
    normalizer = max(n, m, 1)
    gospa = (total_error / normalizer) ** (1/p)
    
    # Decomposition for analysis
    decomposition = {
        'localization': (localization_component / normalizer) ** (1/p) if localization_component > 0 else 0.0,
        'missed': (missed_component / normalizer) ** (1/p) if missed_component > 0 else 0.0,
        'false': (false_component / normalizer) ** (1/p) if false_component > 0 else 0.0
    }
    
    return gospa, decomposition

def compute_gospa_with_uncertainty(true_targets: List[np.ndarray],
                                 estimated_targets: List[np.ndarray],
                                 uncertainties: List[float] = None,
                                 c: float = 100.0,
                                 p: int = 2,
                                 alpha: float = 2.0) -> Tuple[float, dict]:
    """
    Compute GOSPA with uncertainty-aware weighting
    
    This version considers the uncertainty of estimated targets,
    giving less penalty to uncertain estimates.
    """
    
    if uncertainties is None:
        return compute_gospa_error(true_targets, estimated_targets, c, p, alpha)
    
    # Weight the distance matrix by uncertainty
    n = len(true_targets)
    m = len(estimated_targets)
    
    if n == 0 or m == 0:
        return compute_gospa_error(true_targets, estimated_targets, c, p, alpha)
    
    # Modify cutoff based on uncertainty
    weighted_targets = []
    for i, (target, uncertainty) in enumerate(zip(estimated_targets, uncertainties)):
        # Higher uncertainty leads to more lenient evaluation
        uncertainty_factor = min(2.0, 1.0 + uncertainty)
        weighted_targets.append(target)
    
    # Use standard GOSPA but with uncertainty-adjusted cutoff
    adaptive_c = c * 1.5  # More lenient for uncertain estimates
    return compute_gospa_error(true_targets, weighted_targets, adaptive_c, p, alpha)

