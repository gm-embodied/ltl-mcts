"""
Visualization utilities for Scenario 2: Patrol and Respond
Provides functions to create plots and visualizations for experimental results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional
import os

def setup_matplotlib():
    """Setup matplotlib with proper fonts and settings"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

def plot_environment_layout(save_path: str = None):
    """Plot the patrol and respond environment layout"""
    from configs.config import (MAP_WIDTH, MAP_HEIGHT, SCENARIO_2_REGION_A, SCENARIO_2_REGION_B, 
                               AGENT_INITIAL_POSITION, SCENARIO_2_TARGET_A_SPAWN_POS, 
                               SCENARIO_2_TARGET_B_SPAWN_POS)
    
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw map boundary
    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    
    # Draw patrol region A
    region_a_rect = patches.Rectangle(
        (SCENARIO_2_REGION_A[0], SCENARIO_2_REGION_A[2]), 
        SCENARIO_2_REGION_A[1] - SCENARIO_2_REGION_A[0], 
        SCENARIO_2_REGION_A[3] - SCENARIO_2_REGION_A[2],
        linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.3,
        label='Patrol Region A'
    )
    
    # Draw distraction region B
    region_b_rect = patches.Rectangle(
        (SCENARIO_2_REGION_B[0], SCENARIO_2_REGION_B[2]), 
        SCENARIO_2_REGION_B[1] - SCENARIO_2_REGION_B[0], 
        SCENARIO_2_REGION_B[3] - SCENARIO_2_REGION_B[2],
        linewidth=3, edgecolor='red', facecolor='lightcoral', alpha=0.3,
        label='Distraction Region B'
    )
    
    ax.add_patch(region_a_rect)
    ax.add_patch(region_b_rect)
    
    # Draw agent initial position
    ax.plot(AGENT_INITIAL_POSITION[0], AGENT_INITIAL_POSITION[1], 
            'ko', markersize=12, label='Agent Start')
    
    # Draw scripted target positions
    ax.plot(SCENARIO_2_TARGET_A_SPAWN_POS[0], SCENARIO_2_TARGET_A_SPAWN_POS[1],
            'bs', markersize=10, label='Target A (Patrol Duty)')
    ax.plot(SCENARIO_2_TARGET_B_SPAWN_POS[0], SCENARIO_2_TARGET_B_SPAWN_POS[1],
            'rs', markersize=10, label='Target B (Distraction)')
    
    # Add labels and formatting
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Scenario 2: Patrol and Respond Environment', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # Add region labels
    ax.text(SCENARIO_2_REGION_A[0] + 50, SCENARIO_2_REGION_A[3] - 100, 'Patrol\nRegion A', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(SCENARIO_2_REGION_B[0] + 50, SCENARIO_2_REGION_B[3] - 50, 'Distraction\nRegion B', 
            fontsize=12, fontweight='bold', ha='center')
    
    # Add LTL formula
    ax.text(MAP_WIDTH/2, MAP_HEIGHT - 50, 
            'LTL Formula: G(ap_exist_A ⟹ F(ap_loc_any_A))', 
            fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Environment layout saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_trajectory_with_events(trajectory_data: List[Dict], planner_name: str, 
                                target_spawn_times: Dict[str, int], save_path: str = None):
    """Plot agent trajectory with scripted events"""
    from configs.config import (MAP_WIDTH, MAP_HEIGHT, SCENARIO_2_REGION_A, SCENARIO_2_REGION_B,
                               SCENARIO_2_TARGET_A_SPAWN_POS, SCENARIO_2_TARGET_B_SPAWN_POS)
    
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract trajectory points
    positions = [point['agent_position'] for point in trajectory_data]
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    # Draw map and regions
    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    
    # Draw regions
    region_a_rect = patches.Rectangle(
        (SCENARIO_2_REGION_A[0], SCENARIO_2_REGION_A[2]), 
        SCENARIO_2_REGION_A[1] - SCENARIO_2_REGION_A[0], 
        SCENARIO_2_REGION_A[3] - SCENARIO_2_REGION_A[2],
        linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.2
    )
    
    region_b_rect = patches.Rectangle(
        (SCENARIO_2_REGION_B[0], SCENARIO_2_REGION_B[2]), 
        SCENARIO_2_REGION_B[1] - SCENARIO_2_REGION_B[0], 
        SCENARIO_2_REGION_B[3] - SCENARIO_2_REGION_B[2],
        linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.2
    )
    
    ax.add_patch(region_a_rect)
    ax.add_patch(region_b_rect)
    
    # Plot trajectory with color gradient
    for i in range(len(x_coords)-1):
        alpha = 0.3 + 0.7 * i / len(x_coords)
        ax.plot(x_coords[i:i+2], y_coords[i:i+2], 'b-', linewidth=2, alpha=alpha)
    
    # Mark important points
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    
    # Mark target spawn positions
    ax.plot(SCENARIO_2_TARGET_A_SPAWN_POS[0], SCENARIO_2_TARGET_A_SPAWN_POS[1],
            'bs', markersize=12, label='Target A Spawn')
    ax.plot(SCENARIO_2_TARGET_B_SPAWN_POS[0], SCENARIO_2_TARGET_B_SPAWN_POS[1],
            'rs', markersize=12, label='Target B Spawn')
    
    # Add formatting
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Agent Trajectory with Events - {planner_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_duty_performance(results_df, save_path: str = None):
    """Plot patrol duty performance metrics"""
    setup_matplotlib()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Success rate
    success_rates = results_df.groupby('planner')['success_rate'].agg(['mean', 'std'])
    
    bars1 = ax1.bar(success_rates.index, success_rates['mean'], 
                   yerr=success_rates['std'], capsize=5, alpha=0.8, color='green')
    ax1.set_title('Success Rate', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars1, success_rates['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Duty completion rate
    duty_stats = results_df.groupby('planner').apply(
        lambda x: x['duty_completions'].sum() / x['duty_activations'].sum() if x['duty_activations'].sum() > 0 else 0
    )
    
    bars2 = ax2.bar(duty_stats.index, duty_stats.values, alpha=0.8, color='blue')
    ax2.set_title('Duty Completion Rate', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Completion Rate', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, duty_stats.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Average duty activations
    duty_activations = results_df.groupby('planner')['duty_activations'].agg(['mean', 'std'])
    
    bars3 = ax3.bar(duty_activations.index, duty_activations['mean'], 
                   yerr=duty_activations['std'], capsize=5, alpha=0.8, color='orange')
    ax3.set_title('Average Duty Activations', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Duty Activations', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars3, duty_activations['mean']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Average duty completions
    duty_completions = results_df.groupby('planner')['duty_completions'].agg(['mean', 'std'])
    
    bars4 = ax4.bar(duty_completions.index, duty_completions['mean'], 
                   yerr=duty_completions['std'], capsize=5, alpha=0.8, color='purple')
    ax4.set_title('Average Duty Completions', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Duty Completions', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars4, duty_completions['mean']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Duty performance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_ap_probabilities_timeline(ap_history: List[Dict], save_path: str = None):
    """Plot atomic proposition probabilities over time"""
    setup_matplotlib()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Extract data
    time_steps = [entry['time_step'] for entry in ap_history]
    ap_exist_A = [entry['ap_exist_A'] for entry in ap_history]
    ap_loc_any_A = [entry['ap_loc_any_A'] for entry in ap_history]
    patrol_duty_active = [entry.get('patrol_duty_active', False) for entry in ap_history]
    
    # Plot ap_exist_A
    ax1.plot(time_steps, ap_exist_A, 'b-', linewidth=2, label='ap_exist_A')
    ax1.axhline(y=0.4, color='r', linestyle='--', alpha=0.7, label='Threshold (0.4)')
    ax1.set_ylabel('P(exist_A)', fontsize=12)
    ax1.set_title('Atomic Proposition: ap_exist_A', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Plot ap_loc_any_A
    ax2.plot(time_steps, ap_loc_any_A, 'g-', linewidth=2, label='ap_loc_any_A')
    ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Threshold (0.1)')
    ax2.set_ylabel('P(loc_any_A)', fontsize=12)
    ax2.set_title('Atomic Proposition: ap_loc_any_A', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Plot patrol duty status
    duty_status = [1 if active else 0 for active in patrol_duty_active]
    ax3.fill_between(time_steps, duty_status, alpha=0.5, color='orange', label='Patrol Duty Active')
    ax3.set_ylabel('Duty Status', fontsize=12)
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_title('Patrol Duty Status', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"AP probabilities timeline saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_all_plots(results_df, output_dir: str):
    """Create all standard plots for the experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating visualization plots...")
    
    # Environment layout
    plot_environment_layout(os.path.join(output_dir, 'environment_layout.png'))
    
    # Duty performance
    plot_duty_performance(results_df, os.path.join(output_dir, 'duty_performance.png'))
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    plot_environment_layout()
