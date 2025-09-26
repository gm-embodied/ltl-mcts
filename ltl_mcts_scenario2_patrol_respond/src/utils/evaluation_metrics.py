"""
Evaluation metrics calculation tool
Include OSPA error, success rate, efficiency metrics, etc.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.optimize import linear_sum_assignment
import math

def compute_ospa_error(true_targets: List[np.ndarray], estimated_targets: List[np.ndarray], 
                      c: float = 100.0, p: int = 2) -> float:
    """
    Calculate OSPA (Optimal Sub-Pattern Assignment) error
    
    Args:
        true_targets: True target position list, each element is [x, y]
        estimated_targets: Estimated target position list, each element is [x, y]
        c: Cutoff parameter (cutoff parameter)
        p: Distance index parameter
        
    Returns:
        float: OSPA error value
    """
    if len(true_targets) == 0 and len(estimated_targets) == 0:
        return 0.0
    
    n = len(true_targets)
    m = len(estimated_targets)
    
    if n == 0:
        # Only false alarms, no true targets
        return c * (m ** (1/p))
    
    if m == 0:
        # Only missed detections, no estimated targets
        return c * (n ** (1/p))
    
    # Build distance matrix
    distance_matrix = np.zeros((max(n, m), max(n, m)))
    
    for i in range(n):
        for j in range(m):
            dist = np.linalg.norm(np.array(true_targets[i]) - np.array(estimated_targets[j]))
            distance_matrix[i, j] = min(dist, c) ** p
    
    # Handle unbalanced cases
    if n > m:
        # More true targets, add virtual estimated targets
        for i in range(n):
            for j in range(m, n):
                distance_matrix[i, j] = c ** p
    elif m > n:
        # More estimated targets, add virtual true targets
        for i in range(n, m):
            for j in range(m):
                distance_matrix[i, j] = c ** p
    
    # Use Hungarian algorithm to solve optimal assignment
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    
    # Calculate OSPA error
    total_cost = distance_matrix[row_indices, col_indices].sum()
    ospa_error = (total_cost / max(n, m)) ** (1/p)
    
    # Assert protection: OSPA should not exceed cutoff parameter c
    assert ospa_error <= c + 1e-6, f"OSPA {ospa_error} exceeded cutoff {c}"
    
    return float(ospa_error)

def compute_success_rate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate success rate statistics
    
    Args:
        results: Experiment result list
        
    Returns:
        Dict: Contains success rate, confidence interval, etc. statistics
    """
    if not results:
        return {'success_rate': 0.0, 'confidence_interval': (0.0, 0.0), 'num_runs': 0}
    
    successes = sum(1 for r in results if r.get('task_completed', False))
    total_runs = len(results)
    success_rate = successes / total_runs
    
    # Calculate 95% confidence interval (using normal approximation)
    if total_runs > 1:
        std_error = math.sqrt(success_rate * (1 - success_rate) / total_runs)
        margin_error = 1.96 * std_error  # 95% confidence interval
        ci_lower = max(0.0, success_rate - margin_error)
        ci_upper = min(1.0, success_rate + margin_error)
    else:
        ci_lower = ci_upper = success_rate
    
    return {
        'success_rate': success_rate,
        'confidence_interval': (ci_lower, ci_upper),
        'num_runs': total_runs,
        'num_successes': successes
    }

def compute_completion_time_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate completion time statistics (only successful experiments)
    
    Args:
        results: Experiment result list
        
    Returns:
        Dict: Completion time statistics
    """
    successful_results = [r for r in results if r.get('task_completed', False)]
    
    if not successful_results:
        return {
            'mean_completion_time': float('inf'),
            'std_completion_time': 0.0,
            'median_completion_time': float('inf'),
            'min_completion_time': float('inf'),
            'max_completion_time': float('inf'),
            'num_successful_runs': 0
        }
    
    completion_times = [r.get('completion_time', float('inf')) for r in successful_results]
    
    return {
        'mean_completion_time': np.mean(completion_times),
        'std_completion_time': np.std(completion_times),
        'median_completion_time': np.median(completion_times),
        'min_completion_time': np.min(completion_times),
        'max_completion_time': np.max(completion_times),
        'num_successful_runs': len(successful_results)
    }

def compute_ospa_stats(ospa_history: List[float]) -> Dict[str, float]:
    """
    Calculate OSPA error statistics
    
    Args:
        ospa_history: OSPA error history
        
    Returns:
        Dict: OSPA statistics
    """
    if not ospa_history:
        return {
            'mean_ospa': float('inf'),
            'std_ospa': 0.0,
            'median_ospa': float('inf'),
            'min_ospa': float('inf'),
            'max_ospa': float('inf'),
            'final_ospa': float('inf')
        }
    
    return {
        'mean_ospa': np.mean(ospa_history),
        'std_ospa': np.std(ospa_history),
        'median_ospa': np.median(ospa_history),
        'min_ospa': np.min(ospa_history),
        'max_ospa': np.max(ospa_history),
        'final_ospa': ospa_history[-1] if ospa_history else float('inf')
    }

def compute_planning_time_stats(planning_times: List[float]) -> Dict[str, float]:
    """
    Calculate planning time statistics
    
    Args:
        planning_times: Planning time list (seconds)
        
    Returns:
        Dict: Planning time statistics
    """
    if not planning_times:
        return {
            'mean_planning_time': 0.0,
            'std_planning_time': 0.0,
            'median_planning_time': 0.0,
            'min_planning_time': 0.0,
            'max_planning_time': 0.0,
            'mean_planning_time_ms': 0.0
        }
    
    # Convert to milliseconds
    planning_times_ms = [t * 1000 for t in planning_times]
    
    return {
        'mean_planning_time': np.mean(planning_times),
        'std_planning_time': np.std(planning_times),
        'median_planning_time': np.median(planning_times),
        'min_planning_time': np.min(planning_times),
        'max_planning_time': np.max(planning_times),
        'mean_planning_time_ms': np.mean(planning_times_ms)
    }

def compute_comprehensive_metrics(experiment_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comprehensive evaluation metrics for all planners
    
    Args:
        experiment_results: Experiment result dictionary, key is planner name, value is result list
        
    Returns:
        Dict: Comprehensive evaluation metrics
    """
    comprehensive_metrics = {}
    
    for planner_name, results in experiment_results.items():
        if not results:
            continue
        
        # 1. Success rate metrics
        success_stats = compute_success_rate(results)
        
        # 2. Completion time metrics
        completion_stats = compute_completion_time_stats(results)
        
        # 3. OSPA error metrics
        all_ospa_errors = []
        for result in results:
            if 'ospa_history' in result:
                all_ospa_errors.extend(result['ospa_history'])
        ospa_stats = compute_ospa_stats(all_ospa_errors)
        
        # 4. Planning time metrics
        all_planning_times = []
        for result in results:
            if 'planning_times' in result:
                all_planning_times.extend(result['planning_times'])
        planning_stats = compute_planning_time_stats(all_planning_times)
        
        # 5. Additional performance metrics
        additional_stats = compute_additional_performance_metrics(results)
        
        # Merge all metrics
        comprehensive_metrics[planner_name] = {
            'success_metrics': success_stats,
            'efficiency_metrics': completion_stats,
            'accuracy_metrics': ospa_stats,
            'computational_metrics': planning_stats,
            'additional_metrics': additional_stats
        }
    
    return comprehensive_metrics

def compute_additional_performance_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate additional performance metrics
    
    Args:
        results: Experiment result list
        
    Returns:
        Dict: Additional performance metrics
    """
    if not results:
        return {}
    
    # Detection and false alarm statistics
    total_detections = sum(r.get('total_detections', 0) for r in results)
    total_false_alarms = sum(r.get('total_false_alarms', 0) for r in results)
    
    # Scenario 2 specific metrics
    ltl_violations = sum(1 for r in results if r.get('ltl_violation', False))
    duty_triggers = sum(1 for r in results if r.get('duty_trigger_time') is not None)
    
    # Calculate average detection rate and false alarm rate
    num_runs = len(results)
    avg_detections_per_run = total_detections / num_runs if num_runs > 0 else 0
    avg_false_alarms_per_run = total_false_alarms / num_runs if num_runs > 0 else 0
    
    additional_metrics = {
        'avg_detections_per_run': avg_detections_per_run,
        'avg_false_alarms_per_run': avg_false_alarms_per_run,
        'total_detections': total_detections,
        'total_false_alarms': total_false_alarms,
        'ltl_violation_rate': ltl_violations / num_runs if num_runs > 0 else 0,
        'duty_trigger_rate': duty_triggers / num_runs if num_runs > 0 else 0
    }
    
    return additional_metrics

def create_performance_summary_table(comprehensive_metrics: Dict[str, Dict[str, Any]]) -> str:
    """
    Create performance summary table (for paper)
    
    Args:
        comprehensive_metrics: Comprehensive evaluation metrics
        
    Returns:
        str: Formatted table string
    """
    if not comprehensive_metrics or len(comprehensive_metrics) == 0:
        return "No metrics available."
    
    # Table title
    table_lines = [
        "Performance Summary Table",
        "=" * 80,
        f"{'Method':<15} | {'Success Rate':<12} | {'Avg Time':<10} | {'OSPA Error':<12} | {'Planning Time':<15}",
        "-" * 80
    ]
    
    # Add data for each planner
    for planner_name, metrics in comprehensive_metrics.items():
        success_rate = metrics['success_metrics']['success_rate']
        avg_time = metrics['efficiency_metrics']['mean_completion_time']
        ospa_error = metrics['accuracy_metrics']['mean_ospa']
        planning_time = metrics['computational_metrics']['mean_planning_time_ms']
        
        # Format values
        success_str = f"{success_rate:.3f}"
        time_str = f"{avg_time:.1f}" if avg_time != float('inf') else "∞"
        ospa_str = f"{ospa_error:.2f}" if ospa_error != float('inf') else "∞"
        planning_str = f"{planning_time:.2f}ms"
        
        table_lines.append(
            f"{planner_name:<15} | {success_str:<12} | {time_str:<10} | {ospa_str:<12} | {planning_str:<15}"
        )
    
    table_lines.append("=" * 80)
    
    return "\n".join(table_lines)

def extract_trajectory_for_visualization(stats: Dict[str, Any], 
                                       target_positions: Dict[int, List[np.ndarray]]) -> Dict[str, Any]:
    """
    Extract trajectory data for visualization
    
    Args:
        stats: Experiment statistics data
        target_positions: True target position history {time_step: [pos1, pos2, ...]}
        
    Returns:
        Dict: Visualization data
    """
    trajectory_data = {
        'agent_trajectory': [],
        'target_trajectories': {},
        'critical_events': [],
        'ltl_state_changes': []
    }
    
    # Extract agent trajectory
    for traj_point in stats.get('trajectory', []):
        trajectory_data['agent_trajectory'].append({
            'time_step': traj_point['time_step'],
            'position': traj_point['agent_position'],
            'heading': traj_point.get('agent_heading', 0),
            'planner': traj_point.get('planner', 'unknown')
        })
    
    # Extract target trajectory
    for time_step, positions in target_positions.items():
        for i, pos in enumerate(positions):
            target_id = f"target_{i}"
            if target_id not in trajectory_data['target_trajectories']:
                trajectory_data['target_trajectories'][target_id] = []
            
            trajectory_data['target_trajectories'][target_id].append({
                'time_step': time_step,
                'position': pos.tolist()
            })
    
    # Extract critical events
    trajectory_data['critical_events'] = stats.get('critical_decisions', [])
    
    # Extract LTL state changes
    trajectory_data['ltl_state_changes'] = stats.get('ltl_state_history', [])
    
    return trajectory_data
