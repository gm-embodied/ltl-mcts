"""
Evaluation metrics calculation tools
Contains OSPA error, success rate, efficiency metrics, etc.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.optimize import linear_sum_assignment
import math

def compute_ospa_error(true_targets: List[np.ndarray], estimated_targets: List[np.ndarray], 
                      c: float = 100.0, p: int = 2) -> float:
    """
    Compute OSPA (Optimal Sub-Pattern Assignment) error
    
    Args:
        true_targets: list of true target positions, each element is [x, y]
        estimated_targets: list of estimated target positions, each element is [x, y]
        c: cutoff parameter
        p: distance order parameter
        
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
    
    # Compute OSPA error
    total_cost = distance_matrix[row_indices, col_indices].sum()
    ospa_error = (total_cost / max(n, m)) ** (1/p)
    
    # Assert protection: OSPA should not exceed cutoff parameter c
    assert ospa_error <= c + 1e-6, f"OSPA {ospa_error} exceeded cutoff {c}"
    
    return float(ospa_error)

def compute_success_rate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute success rate statistics
    
    Args:
        results: Experiment result list
        
    Returns:
        Dict: Contains success rate, confidence interval, etc.
    """
    if not results:
        return {'success_rate': 0.0, 'confidence_interval': (0.0, 0.0), 'num_runs': 0}
    
    successes = sum(1 for r in results if r.get('task_completed', False))
    total_runs = len(results)
    success_rate = successes / total_runs
    
    # Compute 95% confidence interval (using normal approximation)
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
    Compute completion time statistics (only successful experiments)
    
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
    Compute OSPA error statistics
    
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
    Compute planning time statistics
    
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

def compute_comprehensive_metrics(experiment_results: Dict[str, List[Dict[str, Any]]], 
                                max_steps: int = 50, threshold_tau: int = 20) -> Dict[str, Dict[str, Any]]:
    """
    Compute comprehensive evaluation metrics for all planners
    
    Args:
        experiment_results: Experiment result dictionary, key is planner name, value is result list
        max_steps: Maximum simulation steps
        threshold_tau: Violation threshold
        
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
            if 'gospa_history' in result:  # Use GOSPA instead of OSPA
                all_ospa_errors.extend(result['gospa_history'])
        ospa_stats = compute_ospa_stats(all_ospa_errors)
        
        # 4. Planning time metrics
        all_planning_times = []
        for result in results:
            if 'planning_times' in result:
                all_planning_times.extend(result['planning_times'])
        planning_stats = compute_planning_time_stats(all_planning_times)
        
        # 5. New performance metrics
        # LTL satisfaction time (TTS)
        tts_stats = compute_time_to_satisfaction(results, max_steps)
        
        # Task violation rate (Spec Violation Rate)
        violation_stats = compute_spec_violation_rate(results, threshold_tau)
        
        # 95th percentile revisit interval (Revisit 95th)
        revisit_stats = compute_revisit_95th_percentile(results)
        
        # 6. Additional performance metrics
        additional_stats = compute_additional_performance_metrics(results)
        
        # Merge all metrics
        comprehensive_metrics[planner_name] = {
            'success_metrics': success_stats,
            'efficiency_metrics': completion_stats,
            'accuracy_metrics': ospa_stats,
            'computational_metrics': planning_stats,
            'tts_metrics': tts_stats,   
            'violation_metrics': violation_stats,   
            'revisit_metrics': revisit_stats,  
            'additional_metrics': additional_stats
        }
    
    return comprehensive_metrics

def compute_time_to_satisfaction(results: List[Dict[str, Any]], max_steps: int) -> Dict[str, float]:
    """
    Compute LTL satisfaction time (Time-to-Satisfaction, TTS) metrics
    
    Args:
        results: Experiment result list
        max_steps: Maximum simulation steps
        
    Returns:
        Dict: TTS statistics
    """
    if not results:
        return {
            'mean_tts': float('inf'),
            'std_tts': 0.0,
            'median_tts': float('inf'),
            'min_tts': float('inf'),
            'max_tts': float('inf'),
            'success_rate': 0.0
        }
    
    # Collect the time when LTL is first satisfied
    tts_values = []
    successful_runs = 0
    
    for result in results:
        if result.get('task_completed', False):
            # Successful experiment, use completion time
            completion_time = result.get('completion_time', max_steps)
            tts_values.append(completion_time)
            successful_runs += 1
        else:
            # Failed experiment, set to max_steps
            tts_values.append(max_steps)
    
    if not tts_values:
        return {
            'mean_tts': float('inf'),
            'std_tts': 0.0,
            'median_tts': float('inf'),
            'min_tts': float('inf'),
            'max_tts': float('inf'),
            'success_rate': 0.0
        }
    
    return {
        'mean_tts': np.mean(tts_values),
        'std_tts': np.std(tts_values),
        'median_tts': np.median(tts_values),
        'min_tts': np.min(tts_values),
        'max_tts': np.max(tts_values),
        'success_rate': successful_runs / len(results) if results else 0.0
    }

def compute_spec_violation_rate(results: List[Dict[str, Any]], threshold_tau: int = 20) -> Dict[str, float]:
    """
    Compute task violation rate (Spec Violation Rate) metrics
    Based on the proportion of revisit intervals exceeding the threshold τ
    
    Args:
        results: Experiment result list
        threshold_tau: Violation threshold (steps)
        
    Returns:
        Dict: Violation rate statistics
    """
    if not results:
        return {
            'gap_over_tau_rate_A': 0.0,
            'gap_over_tau_rate_B': 0.0,
            'gap_over_tau_rate_avg': 0.0,
            'on_time_patrol_rate_A': 1.0,
            'on_time_patrol_rate_B': 1.0,
            'on_time_patrol_rate_avg': 1.0
        }
    
    total_gaps_A = []
    total_gaps_B = []
    
    for result in results:
        # Extract revisit intervals from timer history
        timer_A_history = result.get('timer_A_history', [])
        timer_B_history = result.get('timer_B_history', [])
        
        # Compute the revisit interval of region A (when timer is reset)
        if timer_A_history:
            gaps_A = []
            current_gap = 0
            for timer_val in timer_A_history:
                if timer_val == 0 and current_gap > 0:
                    gaps_A.append(current_gap)
                    current_gap = 0
                else:
                    current_gap = timer_val
            if current_gap > 0:   
                gaps_A.append(current_gap)
            total_gaps_A.extend(gaps_A)
        
        # Compute the revisit interval of region B
        if timer_B_history:
            gaps_B = []
            current_gap = 0
            for timer_val in timer_B_history:
                if timer_val == 0 and current_gap > 0:
                    gaps_B.append(current_gap)
                    current_gap = 0
                else:
                    current_gap = timer_val
            if current_gap > 0:   
                gaps_B.append(current_gap)
            total_gaps_B.extend(gaps_B)
    
    # Compute the violation rate
    violations_A = sum(1 for gap in total_gaps_A if gap > threshold_tau)
    violations_B = sum(1 for gap in total_gaps_B if gap > threshold_tau)
    
    gap_over_tau_rate_A = violations_A / len(total_gaps_A) if total_gaps_A else 0.0
    gap_over_tau_rate_B = violations_B / len(total_gaps_B) if total_gaps_B else 0.0
    gap_over_tau_rate_avg = (gap_over_tau_rate_A + gap_over_tau_rate_B) / 2
    
    return {
        'gap_over_tau_rate_A': gap_over_tau_rate_A,
        'gap_over_tau_rate_B': gap_over_tau_rate_B,
        'gap_over_tau_rate_avg': gap_over_tau_rate_avg,
        'on_time_patrol_rate_A': 1.0 - gap_over_tau_rate_A,
        'on_time_patrol_rate_B': 1.0 - gap_over_tau_rate_B,
        'on_time_patrol_rate_avg': 1.0 - gap_over_tau_rate_avg
    }

def compute_revisit_95th_percentile(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute robust uniformity: 95th percentile revisit interval (Revisit 95th) metrics
    
    Args:
        results: Experiment result list
        
    Returns:
        Dict: 95th percentile revisit interval statistics
    """
    if not results:
        return {
            'revisit_95th_A': float('inf'),
            'revisit_95th_B': float('inf'),
            'revisit_95th_avg': float('inf'),
            'revisit_mean_A': float('inf'),
            'revisit_mean_B': float('inf')
        }
    
    total_gaps_A = []
    total_gaps_B = []
    
    for result in results:
        # Extract revisit intervals from timer history
        timer_A_history = result.get('timer_A_history', [])
        timer_B_history = result.get('timer_B_history', [])
        
        # Compute the revisit interval of region A
        if timer_A_history:
            gaps_A = []
            current_gap = 0
            for timer_val in timer_A_history:
                if timer_val == 0 and current_gap > 0:
                    gaps_A.append(current_gap)
                    current_gap = 0
                else:
                    current_gap = timer_val
            if current_gap > 0:
                gaps_A.append(current_gap)
            total_gaps_A.extend(gaps_A)
        
        # Compute the revisit interval of region B
        if timer_B_history:
            gaps_B = []
            current_gap = 0
            for timer_val in timer_B_history:
                if timer_val == 0 and current_gap > 0:
                    gaps_B.append(current_gap)
                    current_gap = 0
                else:
                    current_gap = timer_val
            if current_gap > 0:
                gaps_B.append(current_gap)
            total_gaps_B.extend(gaps_B)
    
    # Compute the 95th percentile
    revisit_95th_A = np.percentile(total_gaps_A, 95) if total_gaps_A else float('inf')
    revisit_95th_B = np.percentile(total_gaps_B, 95) if total_gaps_B else float('inf')
    revisit_95th_avg = (revisit_95th_A + revisit_95th_B) / 2
    
    revisit_mean_A = np.mean(total_gaps_A) if total_gaps_A else float('inf')
    revisit_mean_B = np.mean(total_gaps_B) if total_gaps_B else float('inf')
    
    return {
        'revisit_95th_A': revisit_95th_A,
        'revisit_95th_B': revisit_95th_B,
        'revisit_95th_avg': revisit_95th_avg,
        'revisit_mean_A': revisit_mean_A,
        'revisit_mean_B': revisit_mean_B
    }

def compute_additional_performance_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute additional performance metrics
    
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
    
    # Scenario-specific metrics
    ltl_violations = sum(1 for r in results if r.get('ltl_violation', False))
    duty_triggers = sum(1 for r in results if r.get('duty_trigger_time') is not None)
    
    # Compute the average detection rate and false alarm rate
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
    Create performance summary table (for paper) - using new performance metrics
    
    Args:
        comprehensive_metrics: Comprehensive evaluation metrics
        
    Returns:
        str: Formatted table string
    """
    if not comprehensive_metrics or len(comprehensive_metrics) == 0:
        return "No metrics available."
    
    # Table title - in the order requested by the user: Success Rate | TTS (steps) | GOSPA | Plan Time
    table_lines = [
        "Performance Summary Table",
        "=" * 100,
        f"{'Method':<15} | {'Success Rate':<12} | {'TTS (steps↓)':<13} | {'GOSPA':<8} | {'Plan Time':<10}",
        "-" * 100
    ]
    
    # Add data for each planner
    for planner_name, metrics in comprehensive_metrics.items():
        success_rate = metrics['success_metrics']['success_rate']
        
        # New metric 1: TTS (Time-to-Satisfaction)
        tts_mean = metrics['tts_metrics']['mean_tts']
        
        # GOSPA metrics (from accuracy_metrics)
        gospa_error = metrics['accuracy_metrics']['mean_ospa']  
        
        planning_time = metrics['computational_metrics']['mean_planning_time_ms']
        
        # Format numbers
        success_str = f"{success_rate:.3f}"
        tts_str = f"{tts_mean:.1f}" if tts_mean != float('inf') else "∞"
        gospa_str = f"{gospa_error:.1f}" if gospa_error != float('inf') else "∞"
        planning_str = f"{planning_time:.2f}ms"
        
        table_lines.append(
            f"{planner_name:<15} | {success_str:<12} | {tts_str:<13} | {gospa_str:<8} | {planning_str:<10}"
        )
    
    table_lines.append("=" * 100)
    
    return "\n".join(table_lines)

def extract_trajectory_for_visualization(stats: Dict[str, Any], 
                                       target_positions: Dict[int, List[np.ndarray]]) -> Dict[str, Any]:
    """
    Extract trajectory data for visualization
    
    Args:
        stats: Experiment statistics
        target_positions: True target positions history {time_step: [pos1, pos2, ...]}
        
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
