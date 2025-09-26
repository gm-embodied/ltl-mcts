"""
Scenario 2 Experiment Runner - Patrol and Respond Task
LTL Formula: G(ap_exist_A => F(ap_loc_any_A))
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
from src.scenarios.patrol_respond import PatrolRespondEnvironment
from src.planners.passive_planner import PassivePlanner
from src.planners.info_myopic_planner import InfoMyopicPlanner
from src.planners.ltl_myopic_planner import LTLMyopicPlanner
from src.planners.ltl_mcts_planner import LTLMCTSPlanner
from src.planners.random_planner import RandomPlanner
from src.utils.evaluation_metrics import compute_ospa_error
from src.utils.gospa_metric import compute_gospa_error

def run_single_experiment_worker(args):
    """Global function for multiprocess execution of single experiment"""
    planner_name, run_id, output_dir = args
    
    # Create experiment instance in worker process
    experiment = Scenario2Experiment(output_dir=output_dir)
    
    # Run single experiment
    return experiment.run_single_experiment(planner_name, run_id)

class Scenario2Experiment:
    """Scenario 2 Experiment Manager"""
    
    def __init__(self, output_dir: str = None):
         
        if output_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.output_dir = os.path.join(project_root, f"results/scenario2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
            return LTLMCTSPlanner(planner_name)
        if planner_name == 'LTL_Myopic':
            from src.planners.ltl_myopic_planner import LTLMyopicPlanner
            return LTLMyopicPlanner(planner_name)
        if planner_name == 'Info_Myopic':
            from src.planners.info_myopic_planner import InfoMyopicPlanner
            return InfoMyopicPlanner(planner_name)
        if planner_name == 'Random':
            from src.planners.random_planner import RandomPlanner
            return RandomPlanner(planner_name)
        if planner_name == 'Passive':
            from src.planners.passive_planner import PassivePlanner
            return PassivePlanner(planner_name)
        if planner_name == 'Discretized_POMCP':
            from src.planners.discretized_pomcp_planner import DiscretizedPOMCPPlanner
            return DiscretizedPOMCPPlanner(planner_name)
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
        env = PatrolRespondEnvironment(random_seed=random_seed)
        world_state, pmbm_belief, ltl_state = env.initialize()
        
        # Reset planner
        planner.reset()
        
        # Run simulation
        start_time = time.time()
        step_count = 0
        gospa_history: List[float] = []
        planning_times: List[float] = []
        
        while step_count < SIMULATION_TIME_STEPS:
            # Plan action with high precision timing
            t0 = time.perf_counter()  # Use high precision timer
            action = planner.plan_action(world_state, pmbm_belief, ltl_state)
            planning_time = time.perf_counter() - t0
            planning_times.append(max(planning_time, 1e-6))  # Ensure minimum 1 microsecond
            
            # Execute action (updated to 6-tuple return with info)
            world_state, observation, pmbm_belief, ltl_state, is_done, info = env.step(
                action, planner_name
            )
            
            step_count += 1
            # Calculate OSPA - only use position coordinates [:2], avoid including velocity
            true_positions = [t.position[:2] for t in world_state.get_alive_targets()]
            est_positions = [t.get_position_mean()[:2] for t in pmbm_belief.get_map_targets() if t.existence_prob > 0.3]
            
            gospa_value, gospa_decomp = compute_gospa_error(true_positions, est_positions, c=OSPA_CUTOFF, alpha=GOSPA_ALPHA, p=OSPA_ORDER)
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
            'patrol_efficiency': env_stats.get('patrol_efficiency', 0),  # Duty completions
            'duty_activations': env_stats.get('duty_activations', 0),
            'duty_completions': env_stats.get('duty_completions', 0),
            'temptation_resistance': env_stats.get('temptation_resistance', 0),
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
            'planning_times': planning_times
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
                        f"duties={result['duty_completions']}/{result['duty_activations']}, "
                        f"total_steps={result['total_steps']}")
        
        return result, detailed_result
    
    def run_all_experiments(self):
        """Run all experiments (multiprocess parallel)"""
        self.logger.info(f"Starting scenario 2 experiments (multiprocess parallel), each planner runs {NUM_MONTE_CARLO_RUNS} times")
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
                        'patrol_efficiency': 0, 'duty_activations': 0, 'duty_completions': 0,
                        'temptation_resistance': 0, 'total_steps': 0, 'total_runtime': 0,
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
        
        # Calculate summary statistics (patrol and respond task metrics)
        summary_stats = df.groupby('planner').agg({
            'success_rate': ['mean', 'std', 'count'],
            'completion_time': ['mean', 'std'],
            'total_steps': ['mean', 'std'],
            'total_runtime': ['mean', 'std'],
            'average_planning_time': ['mean', 'std'],
            'total_detections': ['mean', 'std'],
            'total_false_alarms': ['mean', 'std'],
            'patrol_efficiency': ['mean', 'std'],  # Duty completions
            'duty_activations': ['mean', 'std'],
            'duty_completions': ['mean', 'std'],
            'temptation_resistance': ['mean', 'std'],
            'mean_gospa_error': ['mean', 'std']     # Average GOSPA error
        }).round(4)
        
        # Save results
        results_file = os.path.join(self.output_dir, 'scenario2_results.csv')
        df.to_csv(results_file, index=False)
        
        summary_file = os.path.join(self.output_dir, 'scenario2_summary.csv')
        summary_stats.to_csv(summary_file)
        
        # Print results summary (patrol and respond task)
        self.logger.info("\n" + "="*80)
        self.logger.info("Scenario 2 Patrol and Respond Results Summary")
        self.logger.info("="*80)
        self.logger.info("Performance Summary Table")
        self.logger.info("="*80)
        self.logger.info("Method          | Success Rate | Patrol Eff. | Duty Comp. | GOSPA | Plan Time")
        self.logger.info("-"*80)
        
        for planner in df['planner'].unique():
            planner_df = df[df['planner'] == planner]
            success_rate = planner_df['success_rate'].mean()
            success_std = planner_df['success_rate'].std()
            patrol_efficiency = planner_df['patrol_efficiency'].mean() * 100 if 'patrol_efficiency' in planner_df.columns else 0  # Convert to percentage
            duty_completions = planner_df['duty_completions'].mean() if 'duty_completions' in planner_df.columns else 0
            avg_gospa_error = planner_df['mean_gospa_error'].mean() if 'mean_gospa_error' in planner_df.columns else 0
            avg_planning_time = planner_df['average_planning_time'].mean()
            
            # Format planning time in milliseconds (consistent with scenario 1)
            time_str = f"{avg_planning_time*1000:.2f}ms"
            
            self.logger.info(
                f"{planner:15} | {success_rate:.3f}±{success_std:.3f} | {patrol_efficiency:>8.1f}%  | "
                f"{duty_completions:>8.1f}  | {avg_gospa_error:>6.1f} | {time_str}"
            )
        
        self.logger.info("="*80)
        
        # Generate visualizations
        self.generate_visualizations(df)
        
        # Generate performance summary report
        self.generate_performance_report()
        
        return df, summary_stats
    
    def generate_performance_report(self):
        """Generate comprehensive performance summary report"""
        self.logger.info("Generating performance summary report")
        
        try:
            from src.utils.performance_report import Scenario2PerformanceReport
            
            # Create report generator
            report_generator = Scenario2PerformanceReport(self.output_dir)
            
            # Generate report
            report_path = report_generator.generate_report(self.results, self.detailed_stats)
            
            self.logger.info(f"Performance summary report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {str(e)}")
    
    def generate_visualizations(self, df: pd.DataFrame):
        """Generate visualization charts"""
        self.logger.info("Generating visualization charts")
        
        # Set matplotlib fonts (English)
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. Success rate comparison
        plt.figure(figsize=(10, 6))
        success_rates = df.groupby('planner')['success_rate'].agg(['mean', 'std'])
        
        bars = plt.bar(success_rates.index, success_rates['mean'], 
                      yerr=success_rates['std'], capsize=5, alpha=0.8)
        plt.title('Scenario 2: Success Rate Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Success Rate', fontsize=12)
        plt.xlabel('Planner', fontsize=12)
        plt.ylim(0, 1.1)
        
        # Add value labels
        for bar, mean_val in zip(bars, success_rates['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'success_rate_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Duty completion comparison
        plt.figure(figsize=(10, 6))
        duty_completions = df.groupby('planner')['duty_completions'].agg(['mean', 'std'])
        
        bars = plt.bar(duty_completions.index, duty_completions['mean'], 
                      yerr=duty_completions['std'], capsize=5, alpha=0.8, color='green')
        plt.title('Scenario 2: Duty Completion Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Average Duty Completions', fontsize=12)
        plt.xlabel('Planner', fontsize=12)
        
        # Add value labels
        for bar, mean_val in zip(bars, duty_completions['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'duty_completion_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Planning time comparison
        plt.figure(figsize=(10, 6))
        planning_times = df.groupby('planner')['average_planning_time'].agg(['mean', 'std'])
        
        bars = plt.bar(planning_times.index, planning_times['mean'], 
                      yerr=planning_times['std'], capsize=5, alpha=0.8, color='orange')
        plt.title('Scenario 2: Mean Planning Time', fontsize=14, fontweight='bold')
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
        
        # 4. Trajectory comparison figure
        self._create_trajectory_comparison_figure()
        
        # 5. Comprehensive performance radar chart
        if len(df['planner'].unique()) > 1:
            self._create_radar_chart(df)
        
        self.logger.info(f"Visualization charts saved to {self.output_dir}")
    
    def _create_radar_chart(self, df: pd.DataFrame):
        """Create radar chart comparing planners' performance"""
        # Calculate standardized scores for each metric
        metrics = {}
        
        for planner in df['planner'].unique():
            planner_df = df[df['planner'] == planner]
            
            # Success rate (higher is better)
            success_rate = planner_df['success_rate'].mean()
            
            # Duty completion rate (higher is better)
            duty_activations = planner_df['duty_activations'].mean()
            duty_completions = planner_df['duty_completions'].mean()
            if duty_activations > 0:
                duty_completion_rate = duty_completions / duty_activations
            else:
                duty_completion_rate = 0
            
            # Planning efficiency (lower is better, need to invert)
            avg_planning_time = planner_df['average_planning_time'].mean()
            max_planning_time = df['average_planning_time'].max()
            if max_planning_time > 0:
                efficiency_score = 1 - (avg_planning_time / max_planning_time)
            else:
                efficiency_score = 1
            
            metrics[planner] = [success_rate, duty_completion_rate, efficiency_score]
        
        # Create radar chart
        labels = ['Success Rate', 'Duty Completion Rate', 'Computational Efficiency']
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (planner, values) in enumerate(metrics.items()):
            values += values[:1]  # Close the polygon
            ax.plot(angles, values, 'o-', linewidth=2, label=planner, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Scenario 2: Overall Performance Comparison', size=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_radar_chart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_trajectory_comparison_figure(self):
        """Create trajectory comparison figure"""
        self.logger.info("Generating trajectory comparison figure")
        
        # Find representative success and failure cases
        ltl_mcts_success = None
        ltl_mcts_failure = None
        baseline_success = None
        baseline_failure = None
        
        # Search through detailed stats for good examples
        ltl_myopic_failure = None
        for planner_name, results in self.detailed_stats.items():
            for result in results:
                if result['result_summary']['task_completed']:
                    if planner_name == 'LTL_MCTS' and ltl_mcts_success is None:
                        ltl_mcts_success = result
                    elif planner_name != 'LTL_MCTS' and baseline_success is None:
                        baseline_success = result
                else:
                    if planner_name == 'LTL_MCTS' and ltl_mcts_failure is None:
                        ltl_mcts_failure = result
                    elif planner_name == 'LTL_Myopic' and ltl_myopic_failure is None:
                        ltl_myopic_failure = result
                    elif planner_name != 'LTL_MCTS' and baseline_failure is None:
                        baseline_failure = result
        
        # Create trajectory comparison - prioritize LTL_MCTS success vs LTL_Myopic failure
        if ltl_mcts_success and ltl_myopic_failure:
            from src.utils.trajectory_visualization import TrajectoryComparison
            visualizer = TrajectoryComparison()
            
            save_path = os.path.join(self.output_dir, 'trajectory_comparison_figure.png')
            self.logger.info(f"Creating trajectory comparison: LTL_MCTS (success) vs LTL_Myopic (failure)")
            
            visualizer.create_comparison_figure(
                ltl_mcts_success['env_stats'],
                ltl_myopic_failure['env_stats'],
                save_path,
                success_method='LTL_MCTS',
                failure_method='LTL_Myopic'
            )
        elif ltl_mcts_success and baseline_failure:
            from src.utils.trajectory_visualization import TrajectoryComparison
            visualizer = TrajectoryComparison()
            
            save_path = os.path.join(self.output_dir, 'trajectory_comparison_figure.png')
            self.logger.info(f"Creating trajectory comparison: LTL_MCTS (success) vs {baseline_failure['planner']} (failure)")
            
            visualizer.create_comparison_figure(
                ltl_mcts_success['env_stats'],
                baseline_failure['env_stats'],
                save_path,
                success_method='LTL_MCTS',
                failure_method=baseline_failure['planner']
            )
        elif baseline_success and ltl_mcts_failure:
            from src.utils.trajectory_visualization import TrajectoryComparison
            visualizer = TrajectoryComparison()
            
            save_path = os.path.join(self.output_dir, 'trajectory_comparison_figure.png')
            self.logger.info(f"Creating trajectory comparison: {baseline_success['planner']} (success) vs LTL_MCTS (failure)")
            
            visualizer.create_comparison_figure(
                baseline_success['env_stats'],
                ltl_mcts_failure['env_stats'],
                save_path,
                success_method=baseline_success['planner'],
                failure_method='LTL_MCTS'
            )
        else:
            self.logger.warning("Could not find suitable trajectory comparison cases")
            # Create a placeholder comparison with any available data
            if self.detailed_stats:
                first_planner = list(self.detailed_stats.keys())[0]
                if len(self.detailed_stats[first_planner]) >= 2:
                    from src.utils.trajectory_visualization import TrajectoryComparison
                    visualizer = TrajectoryComparison()
                    
                    save_path = os.path.join(self.output_dir, 'trajectory_comparison_figure.png')
                    self.logger.info(f"Creating basic trajectory comparison with {first_planner} data")
                    
                    visualizer.create_comparison_figure(
                        self.detailed_stats[first_planner][0]['env_stats'],
                        self.detailed_stats[first_planner][1]['env_stats'],
                        save_path,
                        success_method=f'{first_planner}_Run1',
                        failure_method=f'{first_planner}_Run2'
                    )

def main():
    """Main function"""
    print("="*80)
    print("Scenario 2: Patrol and Respond Task Experiment")
    print("LTL Formula: G(ap_exist_A => F(ap_loc_any_A)) - Patrol region A and respond to targets")
    print("="*80)
    
    # Create experiment manager
    experiment = Scenario2Experiment()
    
    try:
        # Run all experiments
        experiment.run_all_experiments()
        
        # Analyze results
        df, summary = experiment.analyze_results()
        
        print(f"\nExperiment completed! Results saved in: {experiment.output_dir}")
        print("Main files:")
        print(f"  - Performance summary: performance_summary.txt")
        print(f"  - Detailed results: scenario2_results.csv")
        print(f"  - Summary statistics: scenario2_summary.csv")
        print(f"  - Experiment log: experiment.log")
        print(f"  - Visualization charts: *.png")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nExperiment error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
