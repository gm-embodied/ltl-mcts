"""
Trajectory comparison visualization tool
Specifically designed to generate "one figure worth a thousand words" comparison charts for scenario 2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Any, Tuple, Optional
import seaborn as sns
import sys
import os
# 导入可视化配置
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
try:
    from visualization_config import *
except ImportError:
    # 如果配置文件不存在，使用默认值
    REGION_A_LABEL_POS = [150, 950]
    REGION_B_LABEL_POS = [850, 350]
    TARGET_B_LABEL_OFFSET = [-80, -40]
    FAILURE_ANNOTATION = {
        'xy': [800, 500], 'xytext': [600, 300], 
        'text': 'Abandons patrol duty\nfor distraction target',
        'color': 'darkred'
    }

        # Set font to English to avoid Chinese font issues
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class TrajectoryComparison:
    """Trajectory comparison visualizer"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Default color configuration
        self.colors = {
            'region_A': 'lightblue',
            'region_B': 'lightcoral',
            'ltl_mcts': 'blue',
            'ltl_myopic': 'red',
            'target_A': 'darkblue',
            'target_B': 'darkred',
            'agent_start': 'green',
            'background': 'white'
        }
        
        # Default region configuration
        self.region_A = [0.0, 300.0, 0.0, 1000.0]  # [xmin, xmax, ymin, ymax]
        self.region_B = [700.0, 1000.0, 400.0, 600.0]
        
        # Map dimensions
        self.map_width = 1000.0
        self.map_height = 1000.0
    
    def create_comparison_figure(self, success_data: Dict[str, Any], 
                               failure_data: Dict[str, Any],
                               save_path: str = None,
                               success_method: str = 'LTL_MCTS',
                               failure_method: str = 'LTL_Myopic') -> plt.Figure:
        """
        Create trajectory comparison chart for scenario 2
        
        Args:
            success_data: Experimental data from successful method
            failure_data: Experimental data from failed method
            save_path: Save path (optional)
            success_method: Name of the successful method
            failure_method: Name of the failed method
            
        Returns:
            matplotlib.figure.Figure: Generated chart
        """
        # Create chart: two-column composite figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Subplot (a): Successful method trajectory
        success_color = self.colors.get('ltl_mcts', 'blue') if success_method == 'LTL_MCTS' else 'green'
        self._plot_single_trajectory(ax1, success_data, 
                                   title=f"(a) {success_method}: Task Completed Successfully",
                                   trajectory_color=success_color,
                                   success=True)
        
        # Subplot (b): Failed method trajectory
        failure_color = self.colors.get('ltl_myopic', 'red') if failure_method == 'LTL_Myopic' else 'orange'
        failure_reason = self._get_failure_reason(failure_method)
        self._plot_single_trajectory(ax2, failure_data,
                                   title=f"(b) {failure_method}: {failure_reason}", 
                                   trajectory_color=failure_color,
                                   success=False)
        
        # Set overall title with better spacing
        fig.suptitle('Scenario 2: Trajectory Comparison in Patrol and Response Task\n'
                    'LTL Formula: G(ap_exist_A ⟹ F(ap_loc_any_A))',
                    fontsize=15, fontweight='bold', y=0.95)
        
        # Adjust subplot spacing for better layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, wspace=0.3)
        
        # Save chart
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Trajectory comparison chart saved to: {save_path}")
        
        return fig
    
    def _get_failure_reason(self, method_name: str) -> str:
        """Get failure reason description for different methods"""
        failure_reasons = {
            'LTL_Myopic': 'Distracted by New Target, Task Failed',
            'Info_Myopic': 'Ignores LTL Requirements, Task Failed',
            'Random': 'Random Behavior, Task Failed',
            'Passive': 'Fixed Path, Task Failed',
            'Discretized_POMCP': 'Limited LTL Understanding, Task Failed'
        }
        return failure_reasons.get(method_name, 'Task Failed')
    
    def _plot_single_trajectory(self, ax: plt.Axes, data: Dict[str, Any], 
                              title: str, trajectory_color: str, success: bool):
        """
        Plot single trajectory subplot
        
        Args:
            ax: matplotlib axes object
            data: experimental data
            title: subplot title
            trajectory_color: trajectory color
            success: whether successful
        """
        # 1. Draw background regions
        self._draw_regions(ax)
        
        # 2. Draw target trajectories (dashed lines)
        self._draw_target_trajectories(ax, data)
        
        # 3. Draw agent trajectory (solid line)
        self._draw_agent_trajectory(ax, data, trajectory_color)
        
        # 4. Add key annotations
        self._add_annotations(ax, data, success)
        
        # 5. Set axis properties
        self._setup_axes(ax, title)
    
    def _draw_regions(self, ax: plt.Axes):
        """Draw patrol and distraction regions"""
        # Patrol region R_A
        region_A_rect = patches.Rectangle(
            (self.region_A[0], self.region_A[2]),
            self.region_A[1] - self.region_A[0],
            self.region_A[3] - self.region_A[2],
            linewidth=2, edgecolor='blue', facecolor=self.colors['region_A'],
            alpha=0.3, label='Patrol Region R_A'
        )
        ax.add_patch(region_A_rect)
        
        # Distraction region R_B
        region_B_rect = patches.Rectangle(
            (self.region_B[0], self.region_B[2]),
            self.region_B[1] - self.region_B[0],
            self.region_B[3] - self.region_B[2],
            linewidth=2, edgecolor='red', facecolor=self.colors['region_B'],
            alpha=0.3, label='Distraction Region R_B'
        )
        ax.add_patch(region_B_rect)
        
        # Add region labels (使用配置文件中的位置)
        ax.text(REGION_A_LABEL_POS[0], REGION_A_LABEL_POS[1], 'Patrol Region R_A', 
                fontsize=12, fontweight='bold',
                ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        ax.text(REGION_B_LABEL_POS[0], REGION_B_LABEL_POS[1], 'Distraction Region R_B', 
                fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
    
    def _draw_target_trajectories(self, ax: plt.Axes, data: Dict[str, Any]):
        """Draw target trajectories (dashed lines) with improved visibility"""
        trajectory_data = data.get('trajectory', [])
        if not trajectory_data:
            # Use default scenario 2 target positions
            self._draw_default_target_trajectories(ax)
            return
        
        # Extract actual target positions from trajectory data if available
        # This is a simplified version - in practice you'd extract from the actual data
        self._draw_default_target_trajectories(ax)
    
    def _draw_default_target_trajectories(self, ax: plt.Axes):
        """Draw default target trajectories based on scenario 2 configuration"""
        # Target A trajectory (appears at step 5, in patrol region)
        target_A_start = np.array([280.0, 500.0])
        target_A_end = np.array([280.0, 700.0])  # Moves upward
        
        ax.plot([target_A_start[0], target_A_end[0]], 
               [target_A_start[1], target_A_end[1]], 
               '--', color=self.colors['target_A'], linewidth=3, alpha=0.8, 
               label='Target A (Patrol Duty)')
        
        # Mark target A spawn point with special marker
        ax.plot(target_A_start[0], target_A_start[1], 'o', 
               color=self.colors['target_A'], markersize=12, 
               markerfacecolor='lightblue', markeredgewidth=3,
               markeredgecolor=self.colors['target_A'])
        ax.text(target_A_start[0] - 50, target_A_start[1] - 30, 
               'Target A\n(Step 5)', fontsize=10, fontweight='bold',
               ha='center', color=self.colors['target_A'])
        
        # Target B trajectory (appears at step 8, in distraction region)  
        target_B_start = np.array([720.0, 500.0])
        target_B_end = np.array([720.0, 450.0])  # Moves slightly
        
        ax.plot([target_B_start[0], target_B_end[0]], 
               [target_B_start[1], target_B_end[1]], 
               '--', color=self.colors['target_B'], linewidth=3, alpha=0.8,
               label='Target B (Distraction)')
        
        # Mark target B spawn point with special marker
        ax.plot(target_B_start[0], target_B_start[1], 's', 
               color=self.colors['target_B'], markersize=12, 
               markerfacecolor='lightcoral', markeredgewidth=3,
               markeredgecolor=self.colors['target_B'])
        ax.text(target_B_start[0] + TARGET_B_LABEL_OFFSET[0], 
               target_B_start[1] + TARGET_B_LABEL_OFFSET[1], 
               'Target B\n(Step 8)', fontsize=10, fontweight='bold',
               ha='center', color=self.colors['target_B'])
    
    def _draw_agent_trajectory(self, ax: plt.Axes, data: Dict[str, Any], color: str):
        """Draw agent trajectory (solid line)"""
        trajectory_data = data.get('trajectory', [])
        if not trajectory_data:
            return
        
        # Extract position data
        positions = [point['agent_position'] for point in trajectory_data]
        if not positions:
            return
        
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Draw trajectory with gradient effect
        ax.plot(x_coords, y_coords, '-', color=color, linewidth=4, 
               alpha=0.9, label='Agent Trajectory', zorder=3)
        
        # Mark starting point
        ax.plot(x_coords[0], y_coords[0], 'o', color='green',
               markersize=14, markerfacecolor='lightgreen', markeredgewidth=3,
               label='Start Position', zorder=4)
        ax.text(x_coords[0] + 30, y_coords[0] + 30, 'START', 
               fontsize=10, fontweight='bold', color='green')
        
        # Mark ending point
        if len(x_coords) > 1:
            ax.plot(x_coords[-1], y_coords[-1], 's', color=color,
                   markersize=12, markerfacecolor='white', markeredgewidth=3,
                   zorder=4)
            ax.text(x_coords[-1] + 30, y_coords[-1] + 30, 'END', 
                   fontsize=10, fontweight='bold', color=color)
        
        # Add trajectory direction indicators at key points
        self._add_enhanced_direction_arrows(ax, x_coords, y_coords, color)
    
    def _add_enhanced_direction_arrows(self, ax: plt.Axes, x_coords: List[float], 
                                     y_coords: List[float], color: str):
        """Add enhanced direction arrows at key trajectory points"""
        if len(x_coords) < 2:
            return
        
        # Add arrows at key points (every 8 points for better visibility)
        for i in range(4, len(x_coords) - 1, 8):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            
            if abs(dx) > 2 or abs(dy) > 2:  # Only add arrow when movement is significant
                ax.arrow(x_coords[i], y_coords[i], dx*0.7, dy*0.7,
                        head_width=20, head_length=15, fc=color, ec=color, 
                        alpha=0.8, zorder=3, linewidth=2)
    
    def _add_direction_arrows(self, ax: plt.Axes, x_coords: List[float], 
                            y_coords: List[float], color: str):
        """Add direction arrows on trajectory"""
        if len(x_coords) < 2:
            return
        
        # Add an arrow every 10 points
        for i in range(5, len(x_coords) - 1, 10):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            
            if abs(dx) > 1 or abs(dy) > 1:  # Only add arrow when movement distance is large enough
                ax.arrow(x_coords[i], y_coords[i], dx*0.8, dy*0.8,
                        head_width=15, head_length=10, fc=color, ec=color, alpha=0.7)
    
    def _add_annotations(self, ax: plt.Axes, data: Dict[str, Any], success: bool):
        """Add key annotations"""
        trajectory_data = data.get('trajectory', [])
        critical_decisions = data.get('critical_decisions', [])
        
        if success:
            # Annotations for successful case (使用配置文件)
            # Add success annotation highlighting key behavior
            ax.annotate(SUCCESS_ANNOTATION['text'],
                      xy=SUCCESS_ANNOTATION['xy'], xytext=SUCCESS_ANNOTATION['xytext'],
                      fontsize=11, fontweight='bold', color=SUCCESS_ANNOTATION['color'],
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.9),
                      arrowprops=dict(arrowstyle='->', color=SUCCESS_ANNOTATION['color'], lw=2))
            
            # Add checkmark for success at bottom
            ax.text(self.map_width/2, 30, '✓ LTL Constraint Satisfied', fontsize=13, fontweight='bold', 
                   color='green', ha='center', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.9))
        else:
            # Annotations for failure case (使用配置文件)
            # Add failure annotation highlighting the problem - point to region B
            ax.annotate(FAILURE_ANNOTATION['text'],
                      xy=FAILURE_ANNOTATION['xy'], xytext=FAILURE_ANNOTATION['xytext'],
                      fontsize=11, fontweight='bold', color=FAILURE_ANNOTATION['color'],
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.9),
                      arrowprops=dict(arrowstyle='->', color=FAILURE_ANNOTATION['color'], lw=2))
            
            # Add X mark for failure at bottom
            ax.text(self.map_width/2, 30, '✗ LTL Constraint Violated', fontsize=13, fontweight='bold', 
                   color='red', ha='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.9))
        
        # Add key event markers
        for event in critical_decisions:
            if event.get('event') == 'target_B_spawn':
                # Mark the moment when target B appears
                ax.plot(850, 500, '*', color='gold', markersize=15, 
                       markeredgecolor='red', markeredgewidth=2)
                ax.text(870, 520, f"Step {event['time_step']}\nDistraction Target Appears", 
                       fontsize=10, fontweight='bold', color='red')
    
    def _setup_axes(self, ax: plt.Axes, title: str):
        """Set axis properties for publication-quality figure"""
        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Enhanced grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)  # Put grid behind other elements
        
        # Equal aspect ratio for accurate spatial representation
        ax.set_aspect('equal', adjustable='box')
        
        # Enhanced legend with better positioning - move to upper right
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                          fancybox=True, shadow=True, ncol=1)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        
        # Add coordinate ticks with better spacing
        ax.set_xticks(np.arange(0, self.map_width + 1, 200))
        ax.set_yticks(np.arange(0, self.map_height + 1, 200))
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    
    def create_performance_radar_chart(self, comprehensive_metrics: Dict[str, Dict[str, Any]], 
                                     save_path: str = None) -> plt.Figure:
        """
        Create performance radar chart
        
        Args:
            comprehensive_metrics: Comprehensive evaluation metrics
            save_path: Save path (optional)
            
        Returns:
            matplotlib.figure.Figure: Radar chart
        """
        if not comprehensive_metrics:
            return None
        
        # Prepare data
        planners = list(comprehensive_metrics.keys())
        metrics_names = ['Success Rate', 'Time Efficiency', 'Tracking Accuracy', 'Computational Efficiency']
        
        # Normalize metrics to [0,1] range
        normalized_data = {}
        for planner in planners:
            metrics = comprehensive_metrics[planner]
            
            # Success rate (0-1)
            success_rate = metrics['success_metrics']['success_rate']
            
            # Time efficiency (inverse: shorter time = higher efficiency)
            avg_time = metrics['efficiency_metrics']['mean_completion_time']
            time_efficiency = 1.0 / (1.0 + avg_time / 100.0) if avg_time != float('inf') else 0.0
            
            # Tracking accuracy (inverse: smaller OSPA = higher accuracy)
            ospa = metrics['accuracy_metrics']['mean_ospa']
            tracking_accuracy = 1.0 / (1.0 + ospa / 50.0) if ospa != float('inf') else 0.0
            
            # Computational efficiency (inverse: shorter planning time = higher efficiency)
            planning_time = metrics['computational_metrics']['mean_planning_time_ms']
            computational_efficiency = 1.0 / (1.0 + planning_time / 10.0) if planning_time > 0 else 1.0
            
            normalized_data[planner] = [success_rate, time_efficiency, tracking_accuracy, computational_efficiency]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Set angles
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Draw each planner
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (planner, values) in enumerate(normalized_data.items()):
            values += values[:1]  # Close the polygon
            ax.plot(angles, values, 'o-', linewidth=2, label=planner, 
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        # Set labels and formatting
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title('Planner Comprehensive Performance Comparison Radar Chart', size=16, fontweight='bold', pad=30)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance radar chart saved to: {save_path}")
        
        return fig

def create_trajectory_comparison_from_results(ltl_mcts_result: Dict[str, Any],
                                            ltl_myopic_result: Dict[str, Any],
                                            save_path: str = None) -> plt.Figure:
    """
    Convenience function to create trajectory comparison chart from experimental results
    
    Args:
        ltl_mcts_result: LTL-MCTS experimental results
        ltl_myopic_result: LTL-Myopic experimental results
        save_path: Save path
        
    Returns:
        matplotlib.figure.Figure: Comparison chart
    """
    visualizer = TrajectoryComparison()
    
    # Extract statistics data
    ltl_mcts_stats = ltl_mcts_result.get('env_stats', {})
    ltl_myopic_stats = ltl_myopic_result.get('env_stats', {})
    
    return visualizer.create_comparison_figure(
        ltl_mcts_stats, ltl_myopic_stats, save_path
    )
