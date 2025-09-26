"""
Performance Summary Report Generator for Scenario 2
Generates comprehensive performance analysis reports similar to Scenario 1
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json

class Scenario2PerformanceReport:
    """Performance report generator for Scenario 2: Patrol and Response Task"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.report_path = os.path.join(output_dir, 'performance_summary.txt')
        
    def generate_report(self, experiment_results: Dict[str, List[Dict]], 
                       detailed_stats: Dict[str, List[Dict]]) -> str:
        """
        Generate comprehensive performance summary report
        
        Args:
            experiment_results: Results from all planners
            detailed_stats: Detailed statistics from all planners
            
        Returns:
            str: Path to generated report file
        """
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(experiment_results, detailed_stats)
        
        # Generate report content
        report_content = self._generate_report_content(summary_stats)
        
        # Save report
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        return self.report_path
    
    def _calculate_summary_statistics(self, experiment_results: Dict[str, List[Dict]], 
                                    detailed_stats: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate summary statistics for all planners"""
        
        summary = {}
        
        # Try to read from CSV files first (more reliable)
        try:
            results_csv_path = os.path.join(self.output_dir, 'scenario2_results.csv')
            if os.path.exists(results_csv_path):
                df = pd.read_csv(results_csv_path)
                
                for planner_name in df['planner'].unique():
                    planner_df = df[df['planner'] == planner_name]
                    
                    # Basic success rate statistics
                    success_rate_mean = planner_df['success_rate'].mean()
                    success_rate_std = planner_df['success_rate'].std()
                    n = len(planner_df)
                    
                    # Confidence interval (95%)
                    ci_margin = 1.96 * success_rate_std / np.sqrt(n) if n > 0 else 0
                    ci_lower = max(0, success_rate_mean - ci_margin)
                    ci_upper = min(1, success_rate_mean + ci_margin)
                    
                    # Extract metrics from CSV
                    avg_plan_time = planner_df['average_planning_time'].mean() * 1000  # Convert to ms
                    avg_gospa = planner_df['mean_gospa_error'].mean()
                    
                    avg_patrol_eff = planner_df['patrol_efficiency'].mean() * 100  # Convert to percentage
                    avg_duty_completions = planner_df['duty_completions'].mean()
                    avg_duty_activations = planner_df['duty_activations'].mean()
                    avg_temptation_resistance = planner_df['temptation_resistance'].mean()
                    
                    # For max intervals, we'll use approximations based on duty metrics
                    # Higher duty completions = lower max intervals
                    max_interval_A = max(0, 20 - avg_duty_completions * 2) if avg_duty_completions > 0 else 20
                    max_interval_B = max(0, 15 - avg_temptation_resistance) if avg_temptation_resistance > 0 else 15
                    
                    summary[planner_name] = {
                        'success_rate': success_rate_mean,
                        'success_rate_std': success_rate_std,
                        'confidence_interval': (ci_lower, ci_upper),
                        'avg_planning_time_ms': avg_plan_time,
                        'avg_ospa': 0,  # Not directly available in CSV
                        'avg_gospa': avg_gospa,
                        'patrol_efficiency': avg_patrol_eff,
                        'max_interval_A': max_interval_A,
                        'max_interval_B': max_interval_B,
                        'sample_size': n,
                        'duty_completions': avg_duty_completions,
                        'duty_activations': avg_duty_activations,
                        'temptation_resistance': avg_temptation_resistance
                    }
                
                return summary
        except Exception as e:
            print(f"Warning: Could not read CSV data: {e}")
            
        # Fallback to original method if CSV reading fails
        for planner_name, results in experiment_results.items():
            if not results:
                continue
                
            # Basic success rate statistics
            success_rates = [r.get('success_rate', r.get('success', 0)) for r in results]
            success_rate_mean = np.mean(success_rates)
            success_rate_std = np.std(success_rates)
            
            # Confidence interval (95%)
            n = len(success_rates)
            ci_margin = 1.96 * success_rate_std / np.sqrt(n) if n > 0 else 0
            ci_lower = max(0, success_rate_mean - ci_margin)
            ci_upper = min(1, success_rate_mean + ci_margin)
            
            # Extract detailed metrics
            detailed = detailed_stats.get(planner_name, [])
            
            # Planning time statistics
            plan_times = [r.get('average_planning_time', 0) for r in results if r.get('average_planning_time') is not None]
            avg_plan_time = np.mean(plan_times) * 1000 if plan_times else 0  # Convert to ms
            
            # GOSPA statistics  
            gospa_scores = [r.get('mean_gospa_error', 0) for r in results if r.get('mean_gospa_error') is not None]
            avg_gospa = np.mean(gospa_scores) if gospa_scores else 0
            
            # Patrol efficiency
            patrol_effs = [r.get('patrol_efficiency', 0) for r in results if r.get('patrol_efficiency') is not None]
            avg_patrol_eff = np.mean(patrol_effs) * 100 if patrol_effs else 0  # Convert to percentage
            
            # Duty metrics
            duty_completions = [r.get('duty_completions', 0) for r in results if r.get('duty_completions') is not None]
            avg_duty_completions = np.mean(duty_completions) if duty_completions else 0
            
            duty_activations = [r.get('duty_activations', 0) for r in results if r.get('duty_activations') is not None]
            avg_duty_activations = np.mean(duty_activations) if duty_activations else 0
            
            temptation_resistance = [r.get('temptation_resistance', 0) for r in results if r.get('temptation_resistance') is not None]
            avg_temptation_resistance = np.mean(temptation_resistance) if temptation_resistance else 0
            
            # Approximate max intervals based on duty metrics
            max_interval_A = max(0, 20 - avg_duty_completions * 2) if avg_duty_completions > 0 else 20
            max_interval_B = max(0, 15 - avg_temptation_resistance) if avg_temptation_resistance > 0 else 15
            
            summary[planner_name] = {
                'success_rate': success_rate_mean,
                'success_rate_std': success_rate_std,
                'confidence_interval': (ci_lower, ci_upper),
                'avg_planning_time_ms': avg_plan_time,
                'avg_ospa': 0,
                'avg_gospa': avg_gospa,
                'patrol_efficiency': avg_patrol_eff,
                'max_interval_A': max_interval_A,
                'max_interval_B': max_interval_B,
                'sample_size': n,
                'duty_completions': avg_duty_completions,
                'duty_activations': avg_duty_activations,
                'temptation_resistance': avg_temptation_resistance
            }
        
        return summary
    
    def _generate_report_content(self, summary_stats: Dict[str, Any]) -> str:
        """Generate formatted report content"""
        
        # Header
        report_lines = [
            "=" * 80,
            "                SCENARIO 2: PERFORMANCE SUMMARY REPORT",
            "                      Patrol and Response Task",
            "=" * 80,
            "",
            "EXPERIMENT CONFIGURATION:",
            "- LTL Formula: G(ap_exist_A ⟹ F(ap_loc_any_A))",
            f"- Monte Carlo Runs: {summary_stats[list(summary_stats.keys())[0]]['sample_size']} per algorithm",
            "- Simulation Steps: 100 per episode", 
            "- Map Size: 1000.0×1000.0 units",
            "- Patrol Region A: [0.0, 300.0, 0.0, 1000.0] (Left side of map)",
            "- Distraction Region B: [700.0, 1000.0, 400.0, 600.0] (Right side of map)",
            "",
            "=" * 80,
            "                        PERFORMANCE METRICS",
            "=" * 80,
            "",
            "ALGORITHM COMPARISON:",
            "-" * 80
        ]
        
        # Table header - 保持场景2原有的指标格式
        header = f"{'Method':<18} | {'Success Rate':<12} | {'Patrol Eff.':<11} | {'Duty Comp.':<10} | {'GOSPA':<8} | {'Plan Time':<10}"
        report_lines.append(header)
        report_lines.append("-" * 80)
        
        # Sort planners by success rate (descending)
        sorted_planners = sorted(summary_stats.items(), 
                               key=lambda x: x[1]['success_rate'], reverse=True)
        
        # Add data rows
        for planner_name, stats in sorted_planners:
            success_rate = stats['success_rate']
            success_std = stats['success_rate_std'] 
            patrol_eff = stats['patrol_efficiency']
            duty_comp = stats.get('duty_completions', 0)
            gospa = stats['avg_gospa']
            plan_time = stats['avg_planning_time_ms']
            
            row = f"{planner_name:<18} | {success_rate:.3f}±{success_std:.3f} | {patrol_eff:>8.1f}%  | {duty_comp:>8.1f}  | {gospa:>6.1f} | {plan_time:>7.2f}ms"
            report_lines.append(row)
        
        report_lines.extend([
            "-" * 80,
            "",
            "KEY PERFORMANCE INSIGHTS:",
            "-" * 40
        ])
        
        # Find best performers
        best_overall = max(sorted_planners, key=lambda x: x[1]['success_rate'])
        most_efficient = max(sorted_planners, key=lambda x: x[1]['patrol_efficiency']) 
        best_tracking = min(sorted_planners, key=lambda x: x[1]['avg_gospa'])
        fastest_planning = min(sorted_planners, key=lambda x: x[1]['avg_planning_time_ms'])
        best_duty = max(sorted_planners, key=lambda x: x[1].get('duty_completions', 0))
        
        report_lines.extend([
            f"✓ Best Overall Performance: {best_overall[0]} ({best_overall[1]['success_rate']:.1%} success rate)",
            f"✓ Most Efficient Patrol: {most_efficient[0]} ({most_efficient[1]['patrol_efficiency']:.1f}% patrol efficiency)",
            f"✓ Best Duty Completion: {best_duty[0]} ({best_duty[1].get('duty_completions', 0):.1f} avg completions)",
            f"✓ Best Tracking Accuracy: {best_tracking[0]} (GOSPA: {best_tracking[1]['avg_gospa']:.1f})",
            f"✓ Fastest Planning: {fastest_planning[0]} ({fastest_planning[1]['avg_planning_time_ms']:.1f}ms)",
            "",
            "STATISTICAL ANALYSIS:",
            "-" * 30,
            f"- Sample Size: {summary_stats[list(summary_stats.keys())[0]]['sample_size']} runs per algorithm",
            "- Confidence Level: 95%",
            f"- Total Experiments: {len(summary_stats) * summary_stats[list(summary_stats.keys())[0]]['sample_size']} individual runs",
            "",
            "SUCCESS RATE CONFIDENCE INTERVALS (95%):"
        ])
        
        # Add confidence intervals
        for planner_name, stats in sorted_planners:
            ci_lower, ci_upper = stats['confidence_interval']
            report_lines.append(f"  {planner_name:<18} : {stats['success_rate']:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # Footer
        report_lines.extend([
            "",
            "TASK-SPECIFIC ANALYSIS:",
            "-" * 30,
            "- Patrol Efficiency: Percentage of time spent in patrol region A",
            "- Duty Completions: Number of successful patrol cycles completed",
            "- GOSPA: Generalized Optimal Sub-Pattern Assignment metric for tracking",
            "- Success requires: Maintaining surveillance in A while responding to targets",
            "- Temptation Resistance: Ability to avoid distractions from Region B",
            "",
            "BEHAVIORAL INSIGHTS:",
            "-" * 30,
            "- LTL_MCTS: Long-term planning with LTL constraints, high duty completion",
            "- LTL_Myopic: Short-sighted planning but follows LTL constraints", 
            "- Info_Myopic: Information-driven, may sacrifice patrol for tracking",
            "- Discretized_POMCP: POMDP approach with discretized belief states",
            "- Random/Passive: Baseline methods for comparison",
            "",
            "=" * 80,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Output Directory: {os.path.basename(self.output_dir)}",
            "=" * 80,
            ""
        ])
        
        return "\n".join(report_lines)
