"""
Scenario 1 Experiment Runner - Persistent Surveillance Task
LTL Formula: G(F(ap_exist_A)) & G(F(ap_exist_B))
Run comparative experiments with different planners and collect results
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import threading
import multiprocessing

# Add project root directory to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)

from configs.config import *
from src.scenarios.persistent_surveillance import PersistentSurveillanceEnvironment
from src.planners.passive_planner import PassivePlanner
from src.planners.info_myopic_planner import InfoMyopicPlanner
from src.planners.ltl_myopic_planner import LTLMyopicPlanner
from src.planners.random_planner import RandomPlanner
from src.utils.evaluation_metrics import compute_ospa_error
from src.utils.gospa_metric import compute_gospa_error

def run_single_experiment_worker(args):
    """Global function for multiprocess execution of single experiment"""
    planner_name, run_id, output_dir = args
    
    # Create experiment instance in worker process
    experiment = Scenario1Experiment(output_dir=output_dir)
    
    # Run single experiment
    return experiment.run_single_experiment(planner_name, run_id)

class Scenario1Experiment:
    """Scenario 1 Experiment Manager"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Use absolute path relative to project root to ensure results go to the correct directory
            project_root = os.path.join(os.path.dirname(__file__), '..')
            project_root = os.path.abspath(project_root)
            self.output_dir = os.path.join(project_root, "results", f"scenario1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize planners
        self.planners = self._initialize_planners()
        
        # Result storage
        self.results = {planner_name: [] for planner_name in self.planners.keys()}
        self.detailed_stats = {planner_name: [] for planner_name in self.planners.keys()}
        
        # Display CPU information to optimize parallel performance
        cpu_count = multiprocessing.cpu_count()
        self.logger.info(f'Detected {cpu_count} CPU cores, will use up to {min(48, cpu_count)} threads in parallel')
    
    def setup_logging(self):
        """Setup logging system"""
        log_file = os.path.join(self.output_dir, 'experiment.log')
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_planners(self) -> Dict[str, Any]:
        """Initialize all planners"""
        from src.planners.discretized_pomcp_planner import DiscretizedPOMCPPlanner
        from src.planners.ltl_mcts_planner import LTLMCTSPlanner
        
        # Only register available planner names, avoid creating shared instances in main thread
        planners = {
            'LTL_MCTS': 'LTL_MCTS',
            'LTL_Myopic': 'LTL_Myopic',
            'Info_Myopic': 'Info_Myopic',
            'Random': 'Random',
            'Passive': 'Passive',
            'Discretized_POMCP': 'Discretized_POMCP',
        }
        
        self.logger.info(f"Initialized {len(planners)} planners: {list(planners.keys())}")
        return planners

    def _create_planner(self, planner_name: str):
        """Create independent planner instance for current task (thread-safe)"""
        if planner_name == 'LTL_MCTS':
            from src.planners.ltl_mcts_planner import LTLMCTSPlanner
            return LTLMCTSPlanner()
        if planner_name == 'LTL_Myopic':
            from src.planners.ltl_myopic_planner import LTLMyopicPlanner
            return LTLMyopicPlanner()
        if planner_name == 'Info_Myopic':
            from src.planners.info_myopic_planner import InfoMyopicPlanner
            return InfoMyopicPlanner()
        if planner_name == 'Random':
            from src.planners.random_planner import RandomPlanner
            return RandomPlanner()
        if planner_name == 'Passive':
            from src.planners.passive_planner import PassivePlanner
            return PassivePlanner()
        if planner_name == 'Discretized_POMCP':
            from src.planners.discretized_pomcp_planner import DiscretizedPOMCPPlanner
            return DiscretizedPOMCPPlanner()
        raise ValueError(f"Unknown planner: {planner_name}")
    
    def run_single_experiment(self, planner_name: str, run_id: int) -> Dict[str, Any]:
        """Run single experiment"""
        self.logger.info(f"Running {planner_name} planner, run {run_id} (thread {threading.current_thread().ident})")
        
        # Thread-safe random seed setting
        thread_id_mod = threading.current_thread().ident % 1000
        random_seed = (
            RANDOM_SEED_BASE
            + run_id * 1000
            + (hash(planner_name) % 1000)
            + thread_id_mod
        )
        import random
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Create independent planner instance for this task
        planner = self._create_planner(planner_name)
        
        # Create environment
        env = PersistentSurveillanceEnvironment(random_seed=random_seed)
        world_state, pmbm_belief, ltl_state = env.initialize()
        
        # Reset planner
        planner.reset()
        
        # Run simulation
        start_time = time.time()
        step_count = 0
        gospa_history: List[float] = []
        planning_times: List[float] = []
        
        while step_count < SIMULATION_TIME_STEPS:
            # Plan action
            t0 = time.time()
            try:
                action = planner.plan_action(world_state, pmbm_belief, ltl_state)
            except Exception as e:
                self.logger.error(f"Planner {planner_name} failed at step {step_count}: {e}")
                # Use default action on error
                action = 0
            planning_times.append(time.time() - t0)
            
            # Execute action (updated to 6-tuple return with info)
            world_state, observation, pmbm_belief, ltl_state, is_done, info = env.step(
                action, planner_name
            )
            
            step_count += 1
            # Calculate GOSPA with improved parameters and filtering
            # 1. Get true alive targets
            alive_targets = world_state.get_alive_targets()
            true_positions = [t.position[:2] for t in alive_targets]
            
            # 2. Get estimated targets with enhanced filtering using config parameters
            map_targets = pmbm_belief.get_map_targets()
            
            # Filter by existence probability using configurable threshold
            high_confidence_targets = [t for t in map_targets if t.existence_prob > OSPA_EXISTENCE_THRESHOLD]
            est_positions = [t.get_position_mean()[:2] for t in high_confidence_targets]
            
            # 3. Fallback with limited target count if no high-confidence targets
            if len(est_positions) == 0 and len(map_targets) > 0:
                sorted_targets = sorted(map_targets, key=lambda x: x.existence_prob, reverse=True)
                # Use configurable ratio to limit estimated targets
                max_targets = min(int(len(true_positions) * OSPA_MAX_TARGETS_RATIO) + 1, len(sorted_targets))
                est_positions = [t.get_position_mean()[:2] for t in sorted_targets[:max_targets] if t.existence_prob > 0.5]
            
            # Use GOSPA instead of OSPA for better evaluation in high-clutter environments
            gospa_value, gospa_decomp = compute_gospa_error(true_positions, est_positions, 
                                                          p=OSPA_ORDER, c=OSPA_CUTOFF, alpha=GOSPA_ALPHA)
            gospa_history.append(gospa_value)
            
            if is_done:
                break
        
        total_time = time.time() - start_time
        
        # Collect results
        env_stats = env.get_statistics()
        planner_stats = planner.get_statistics()
        
        result = {
            'run_id': run_id,
            'planner': planner_name,
            'task_completed': env_stats['task_completed'],
            'completion_time': env_stats['completion_time'],
            'ltl_violation': env_stats.get('ltl_violation', False),
            'violation_time': env_stats.get('violation_time'),
            'patrol_efficiency': env_stats.get('patrol_efficiency', 0),  # LTL refresh count (旧指标，保留以兼容)
            'timeout_steps': env_stats.get('timeout_steps', 0),
            'max_interval_A': env_stats.get('max_interval_A', 0),  # 旧指标，保留以兼容
            'max_interval_B': env_stats.get('max_interval_B', 0),  # 旧指标，保留以兼容
            'total_steps': step_count,
            'total_runtime': total_time,
            'success_rate': 1.0 if env_stats['task_completed'] else 0.0,
            'total_detections': env_stats['total_detections'],
            'total_false_alarms': env_stats['total_false_alarms'],
            'average_planning_time': planner_stats['average_planning_time'],
            'random_seed': random_seed,
            'mean_gospa_error': float(np.mean(gospa_history)) if len(gospa_history) > 0 else float('inf'),
            'final_gospa_error': float(gospa_history[-1]) if len(gospa_history) > 0 else float('inf'),
            'gospa_history': gospa_history,
            'planning_times': planning_times,
            # 新指标所需的数据
            'timer_A_history': env_stats.get('timer_A_history', []),
            'timer_B_history': env_stats.get('timer_B_history', [])
        }
        
        # Detailed statistics
        detailed_result = {
            'run_id': run_id,
            'planner': planner_name,
            'env_stats': env_stats,
            'planner_stats': planner_stats,
            'result_summary': result
        }
        
        self.logger.info(f"{planner_name} run{run_id}: success={result['task_completed']}, "
                        f"time={result['completion_time']}, total_steps={result['total_steps']}")
        
        return result, detailed_result
    
    def run_all_experiments(self):
        """Run all experiments (multiprocess parallel)"""
        self.logger.info(f"Starting scenario 1 experiments (multiprocess parallel), each planner runs {NUM_MONTE_CARLO_RUNS} times")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        total_experiments = len(self.planners) * NUM_MONTE_CARLO_RUNS
        
        # Task distribution (include output directory for worker processes)
        tasks = []
        for run_id in range(NUM_MONTE_CARLO_RUNS):
            for planner_name in self.planners.keys():
                tasks.append((planner_name, run_id, self.output_dir))
        
        # Calculate available process count (reserve 4 for system), not exceeding total task count
        cpu_cores = multiprocessing.cpu_count()
        max_workers_to_use = max(1, cpu_cores - 4)
        max_workers = min(max_workers_to_use, len(tasks))
        self.logger.info(f'Preparing to run {total_experiments} tasks using {max_workers} parallel processes')
        
        start_time = time.time()
        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_single_experiment_worker, task): (task[0], task[1]) for task in tasks}
            for fut in as_completed(futures):
                pn, rid = futures[fut]
                try:
                    result, detailed_result = fut.result()
                    self.results[pn].append(result)
                    self.detailed_stats[pn].append(detailed_result)
                except Exception as e:
                    self.logger.warning(f'Task {pn}-{rid} failed: {str(e)}')
                    failed_result = {
                        'run_id': rid, 'planner': pn, 'task_completed': False,
                        'completion_time': None, 'ltl_violation': True, 'violation_time': 0,
                        'patrol_efficiency': 0, 'total_steps': 0, 'total_runtime': 0,
                        'success_rate': 0.0, 'total_detections': 0, 'total_false_alarms': 0,
                        'average_planning_time': 0.0, 'random_seed': 0,
                        'mean_gospa_error': float('inf'), 'final_gospa_error': float('inf'),
                        'gospa_history': [], 'planning_times': []
                    }
                    self.results[pn].append(failed_result)
                completed += 1
                if completed % max(1, total_experiments // 10) == 0:  # Report progress 10 times only
                    progress = (completed / total_experiments) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({completed}/{total_experiments}) [multiprocess {max_workers}]")
        total_elapsed = time.time() - start_time
        self.logger.info(f"All experiments completed | Total time: {total_elapsed:.2f}s | Average per task: {total_elapsed/total_experiments:.2f}s | Parallel processes: {max_workers}")
    
    def analyze_results(self):
        """Analyze experimental results"""
        self.logger.info("Starting result analysis")
        
        # Create results DataFrame
        all_results = []
        for planner_name, results in self.results.items():
            all_results.extend(results)
        
        df = pd.DataFrame(all_results)
        
        if df.empty:
            self.logger.warning("No experimental results to analyze (results empty; forgot to run run_all_experiments?)")
            empty = pd.DataFrame()
            return empty, empty
        
        # Calculate summary statistics (persistent surveillance task metrics)
        summary_stats = df.groupby('planner').agg({
            'success_rate': ['mean', 'std', 'count'],
            'completion_time': ['mean', 'std'],
            'total_steps': ['mean', 'std'],
            'total_runtime': ['mean', 'std'],
            'average_planning_time': ['mean', 'std'],
            'total_detections': ['mean', 'std'],
            'total_false_alarms': ['mean', 'std'],
            'patrol_efficiency': ['mean', 'std'],  # LTL refresh count (旧指标，保留)
            'timeout_steps': ['mean', 'std'],
            'max_interval_A': ['mean', 'std'],  # 旧指标，保留
            'max_interval_B': ['mean', 'std'],  # 旧指标，保留
            'mean_gospa_error': ['mean', 'std']     # Average GOSPA error
        }).round(4)
        
        # Save results
        results_file = os.path.join(self.output_dir, 'scenario1_results.csv')
        df.to_csv(results_file, index=False)
        
        summary_file = os.path.join(self.output_dir, 'scenario1_summary.csv')
        summary_stats.to_csv(summary_file)
        
        # 计算新的性能指标
        from src.utils.evaluation_metrics import (
            compute_comprehensive_metrics, create_performance_summary_table
        )
        
        # 将结果转换为所需格式
        experiment_results = {}
        for planner in df['planner'].unique():
            planner_results = df[df['planner'] == planner].to_dict('records')
            experiment_results[planner] = planner_results
        
        # 计算综合指标
        comprehensive_metrics = compute_comprehensive_metrics(
            experiment_results, 
            max_steps=SIMULATION_TIME_STEPS, 
            threshold_tau=20  # 使用20步作为违约阈值
        )
        
        # 创建新的性能摘要表
        performance_table = create_performance_summary_table(comprehensive_metrics)
        
        # Print results summary (persistent surveillance task)
        self.logger.info("\n" + "="*100)
        self.logger.info("Scenario 1 Persistent Surveillance Results Summary")
        self.logger.info("="*100)
        self.logger.info(performance_table)
        self.logger.info("="*100)
        
        
        # Generate visualizations
        self.generate_visualizations(df, comprehensive_metrics)
        
        # Generate performance report
        self._generate_performance_report(summary_stats, df, comprehensive_metrics)
        
        return df, summary_stats
    
    def generate_visualizations(self, df: pd.DataFrame, comprehensive_metrics: Dict[str, Dict[str, Any]]):
        """Generate visualization charts"""
        self.logger.info("Generating visualization charts")
        
        # Set matplotlib fonts (English)
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. Success rate comparison
        plt.figure(figsize=(10, 6))
        success_rates = df.groupby('planner')['success_rate'].agg(['mean', 'std', 'count'])
        
        # Calculate 95% confidence interval for binomial proportion
        # Using Wilson score interval for better accuracy
        z = 1.96  # 95% confidence
        n = success_rates['count']
        p = success_rates['mean']
        
        # Wilson score interval
        denominator = 1 + z**2/n
        center = (p + z**2/(2*n)) / denominator
        margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
        
        # Calculate confidence intervals, ensuring they stay within [0,1]
        ci_lower = np.maximum(0, center - margin)
        ci_upper = np.minimum(1, center + margin)
        
        # Use asymmetric error bars
        yerr_lower = success_rates['mean'] - ci_lower
        yerr_upper = ci_upper - success_rates['mean']
        yerr = [yerr_lower, yerr_upper]
        
        bars = plt.bar(success_rates.index, success_rates['mean'], 
                      yerr=yerr, capsize=5, alpha=0.8)
        plt.title('Scenario 1: Success Rate Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Success Rate', fontsize=12)
        plt.xlabel('Planner', fontsize=12)
        
        # Set y-axis limit to accommodate error bars
        plt.ylim(0, 1.05)
        
        # Add value labels
        for bar, mean_val in zip(bars, success_rates['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'success_rate_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Completion time distribution (successful runs only)
        successful_df = df[df['task_completed'] == True]
        if not successful_df.empty:
            plt.figure(figsize=(12, 6))
            
            planners = successful_df['planner'].unique()
            for i, planner in enumerate(planners):
                planner_data = successful_df[successful_df['planner'] == planner]['completion_time']
                plt.subplot(1, len(planners), i+1)
                plt.hist(planner_data, bins=10, alpha=0.7, edgecolor='black')
                plt.title(f'{planner}\nCompletion Time Distribution', fontsize=10)
                plt.xlabel('Completion Time (steps)', fontsize=9)
                plt.ylabel('Count', fontsize=9)
                plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'completion_time_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Planning time comparison
        plt.figure(figsize=(10, 6))
        planning_times = df.groupby('planner')['average_planning_time'].agg(['mean', 'std'])
        
        bars = plt.bar(planning_times.index, planning_times['mean'], 
                      yerr=planning_times['std'], capsize=5, alpha=0.8, color='orange')
        plt.title('Scenario 1: Mean Planning Time', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Planning Time (s)', fontsize=12)
        plt.xlabel('Planner', fontsize=12)
        plt.yscale('log')  # Use log scale due to potentially large differences in planning time
        
        # Add value labels
        for bar, mean_val in zip(bars, planning_times['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{mean_val:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.subplots_adjust(bottom=0.15)  # Increase bottom margin
        plt.savefig(os.path.join(self.output_dir, 'planning_time_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Comprehensive performance radar chart
        if len(df['planner'].unique()) > 1:
            self._create_radar_chart(df)
        
        # 5. New metrics visualization
        from src.utils.visualization import plot_new_metrics, plot_detailed_violation_analysis
        
        plot_new_metrics(comprehensive_metrics, self.output_dir)
        plot_detailed_violation_analysis(comprehensive_metrics, self.output_dir)
        
        self.logger.info(f"Visualization charts (including new metrics) saved to {self.output_dir}")
    
    def _create_radar_chart(self, df: pd.DataFrame):
        """Create radar chart comparing planners' performance across 6 key metrics"""
        # Calculate standardized scores for each metric
        metrics = {}
        
        # Get global min/max for normalization
        all_planners_data = {}
        for metric in ['success_rate', 'patrol_efficiency', 'max_interval_A', 'max_interval_B', 'mean_gospa_error', 'average_planning_time']:
            all_planners_data[metric] = df[metric].values
        
        for planner in df['planner'].unique():
            planner_df = df[df['planner'] == planner]
            
            # 1. Success Rate (higher is better) - already 0-1
            success_rate = planner_df['success_rate'].mean()
            
            # 2. Patrol Efficiency (lower is better, invert and normalize)
            patrol_eff = planner_df['patrol_efficiency'].mean()
            max_patrol_eff = df['patrol_efficiency'].max()
            min_patrol_eff = df['patrol_efficiency'].min()
            if max_patrol_eff > min_patrol_eff:
                patrol_eff_score = 1 - (patrol_eff - min_patrol_eff) / (max_patrol_eff - min_patrol_eff)
            else:
                patrol_eff_score = 1.0
            
            # 3. Region Coverage Balance (lower max interval is better)
            max_int_A = planner_df['max_interval_A'].mean()
            max_int_B = planner_df['max_interval_B'].mean()
            avg_max_interval = (max_int_A + max_int_B) / 2
            max_interval_overall = df[['max_interval_A', 'max_interval_B']].values.max()
            min_interval_overall = df[['max_interval_A', 'max_interval_B']].values.min()
            if max_interval_overall > min_interval_overall:
                coverage_score = 1 - (avg_max_interval - min_interval_overall) / (max_interval_overall - min_interval_overall)
            else:
                coverage_score = 1.0
            
            # 4. Tracking Accuracy (lower GOSPA is better, invert and normalize)
            gospa_error = planner_df['mean_gospa_error'].mean()
            max_gospa = df['mean_gospa_error'].max()
            min_gospa = df['mean_gospa_error'].min()
            if max_gospa > min_gospa:
                tracking_score = 1 - (gospa_error - min_gospa) / (max_gospa - min_gospa)
            else:
                tracking_score = 1.0
            
            # 5. Computational Efficiency (lower planning time is better, invert and normalize)
            planning_time = planner_df['average_planning_time'].mean()
            max_planning_time = df['average_planning_time'].max()
            min_planning_time = df['average_planning_time'].min()
            if max_planning_time > min_planning_time:
                computation_score = 1 - (planning_time - min_planning_time) / (max_planning_time - min_planning_time)
            else:
                computation_score = 1.0
            
            # 6. Task Completion Consistency (based on success rate variance, lower std is better)
            success_std = planner_df['success_rate'].std()
            max_std = df.groupby('planner')['success_rate'].std().max()
            min_std = df.groupby('planner')['success_rate'].std().min()
            if max_std > min_std:
                consistency_score = 1 - (success_std - min_std) / (max_std - min_std)
            else:
                consistency_score = 1.0
            
            metrics[planner] = [success_rate, patrol_eff_score, coverage_score, 
                              tracking_score, computation_score, consistency_score]
        
        # Create radar chart
        labels = ['Success Rate', 'Movement\nEfficiency', 'Coverage\nBalance', 
                 'Tracking\nAccuracy', 'Computational\nEfficiency', 'Task\nConsistency']
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
        for i, (planner, values) in enumerate(metrics.items()):
            values += values[:1]  # Close the polygon
            ax.plot(angles, values, 'o-', linewidth=2.5, label=planner, 
                   color=colors[i % len(colors)], markersize=6)
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=9)
        plt.title('Scenario 1: Multi-Dimensional Performance Comparison', 
                 size=14, fontweight='bold', pad=30)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_radar_chart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_performance_report(self, summary_stats: pd.DataFrame, df: pd.DataFrame, 
                                    comprehensive_metrics: Dict[str, Dict[str, Any]]):
        """Generate a comprehensive performance report in TXT format"""
        report_file = os.path.join(self.output_dir, 'performance_summary.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("                SCENARIO 1: PERFORMANCE SUMMARY REPORT\n")
            f.write("                    Persistent Surveillance Task\n")
            f.write("="*100 + "\n\n")
            
            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write(f"- LTL Formula: G(F(ap_exist_A)) ∧ G(F(ap_exist_B))\n")
            f.write(f"- Monte Carlo Runs: {NUM_MONTE_CARLO_RUNS} per algorithm\n")
            f.write(f"- Simulation Steps: {SIMULATION_TIME_STEPS} per episode\n")
            f.write(f"- Map Size: {MAP_WIDTH}×{MAP_HEIGHT} units\n")
            f.write(f"- Surveillance Regions: A={REGION_A}, B={REGION_B}\n")
            f.write(f"- Violation Threshold (τ): 20 steps\n\n")
            
            f.write("="*100 + "\n")
            f.write("                        PERFORMANCE METRICS\n")
            f.write("="*100 + "\n\n")
            
            # Create performance table - 按照用户要求的顺序：Success Rate | TTS (steps) | GOSPA | Plan Time
            f.write("ALGORITHM COMPARISON:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Method':<15} | {'Success Rate':<12} | {'TTS (steps↓)':<13} | {'GOSPA':<8} | {'Plan Time':<10}\n")
            f.write("-" * 80 + "\n")
            
            # Sort by success rate (descending)
            planners_sorted = df.groupby('planner')['success_rate'].mean().sort_values(ascending=False).index
            
            for planner in planners_sorted:
                if planner in comprehensive_metrics:
                    metrics = comprehensive_metrics[planner]
                    success_rate = metrics['success_metrics']['success_rate']
                    success_std = df[df['planner'] == planner]['success_rate'].std()
                    
                    # 新指标 - 按照用户要求的顺序：Success Rate | TTS (steps) | GOSPA | Plan Time
                    tts_mean = metrics['tts_metrics']['mean_tts']
                    gospa_error = metrics['accuracy_metrics']['mean_ospa']  # 这里实际是GOSPA
                    plan_time = metrics['computational_metrics']['mean_planning_time_ms']
                    
                    # 格式化数值
                    tts_str = f"{tts_mean:.1f}" if tts_mean != float('inf') else "∞"
                    gospa_str = f"{gospa_error:.1f}" if gospa_error != float('inf') else "∞"
                    
                    f.write(f"{planner:<15} | {success_rate:.3f}±{success_std:.3f} | "
                           f"{tts_str:>11}   | {gospa_str:>6}   | "
                           f"{plan_time:>7.2f}ms\n")
            
            f.write("-" * 80 + "\n\n")
            
            # Metric explanations
            f.write("METRIC EXPLANATIONS:\n")
            f.write("-" * 50 + "\n")
            f.write("1) Success Rate: Proportion of runs that successfully completed the LTL task\n")
            f.write("2) TTS (Time-to-Satisfaction): Steps to first LTL acceptance (lower is better)\n")
            f.write("3) GOSPA: Generalized Optimal Sub-Pattern Assignment tracking error (lower is better)\n")
            f.write("4) Plan Time: Average computational time per planning step (lower is better)\n\n")
            
            
            # Key insights
            f.write("KEY PERFORMANCE INSIGHTS:\n")
            f.write("-" * 50 + "\n")
            
            best_planner = planners_sorted[0]
            best_success_rate = df[df['planner'] == best_planner]['success_rate'].mean()
            
            f.write(f"✓ Best Overall Performance: {best_planner} ({best_success_rate:.1%} success rate)\n")
            
            # Find best in each metric category
            best_tts_planner = min(comprehensive_metrics.keys(), 
                                 key=lambda p: comprehensive_metrics[p]['tts_metrics']['mean_tts'])
            best_gospa_planner = df.groupby('planner')['mean_gospa_error'].mean().idxmin()
            fastest_planner = df.groupby('planner')['average_planning_time'].mean().idxmin()
            
            f.write(f"✓ Fastest LTL Satisfaction (TTS): {best_tts_planner}\n")
            f.write(f"✓ Best Tracking Accuracy (GOSPA): {best_gospa_planner}\n")
            f.write(f"✓ Fastest Planning: {fastest_planner}\n\n")
            
            # Statistical significance
            f.write("STATISTICAL ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"- Sample Size: {NUM_MONTE_CARLO_RUNS} runs per algorithm\n")
            f.write(f"- Confidence Level: 95%\n")
            f.write(f"- Total Experiments: {len(df)} individual runs\n\n")
            
            # Success rate confidence intervals
            f.write("SUCCESS RATE CONFIDENCE INTERVALS (95%):\n")
            for planner in planners_sorted:
                planner_df = df[df['planner'] == planner]
                n = len(planner_df)
                p = planner_df['success_rate'].mean()
                
                # Wilson score interval
                z = 1.96
                denominator = 1 + z**2/n
                center = (p + z**2/(2*n)) / denominator
                margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
                ci_lower = max(0, center - margin)
                ci_upper = min(1, center + margin)
                
                f.write(f"  {planner:<18}: {p:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write("="*80 + "\n")
        
        self.logger.info(f"Performance report saved to: {report_file}")

def main():
    """Main function"""
    print("="*80)
    print("Scenario 1: Persistent Surveillance Task Experiment")
    print("LTL Formula: G(F(ap_exist_A)) & G(F(ap_exist_B)) - Persistent monitoring of two regions")
    print("="*80)
    
    # Create experiment manager
    experiment = Scenario1Experiment()
    
    try:
        # Run all experiments
        experiment.run_all_experiments()
        
        # Analyze results
        df, summary = experiment.analyze_results()
        
        print(f"\nExperiment completed! Results saved in: {experiment.output_dir}")
        print("Main files:")
        print(f"  - Detailed results: scenario1_results.csv")
        print(f"  - Summary statistics: scenario1_summary.csv")
        print(f"  - Experiment log: experiment.log")
        print(f"  - Visualization charts: *.png")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nExperiment error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
