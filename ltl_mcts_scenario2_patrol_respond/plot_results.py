#!/usr/bin/env python3
"""
Scenario 2 Results Plotting Script
Generate visualizations from experimental results
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.utils.visualization import (
    plot_environment_layout, plot_duty_performance, 
    create_all_plots, setup_matplotlib
)

def find_latest_results_dir():
    """Find the most recent results directory"""
    results_base = Path("results")
    if not results_base.exists():
        print("No results directory found. Please run experiments first.")
        return None
    
    # Find all scenario2 result directories
    scenario2_dirs = [d for d in results_base.iterdir() 
                     if d.is_dir() and d.name.startswith("scenario2_")]
    
    if not scenario2_dirs:
        print("No scenario2 results found. Please run experiments first.")
        return None
    
    # Return the most recent one
    latest_dir = max(scenario2_dirs, key=lambda d: d.stat().st_mtime)
    print(f"Using results from: {latest_dir}")
    return latest_dir

def load_results(results_dir: Path):
    """Load experimental results"""
    results_file = results_dir / "scenario2_results.csv"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} experimental results")
    print(f"Planners: {df['planner'].unique().tolist()}")
    return df

def main():
    """Main plotting function"""
    print("="*60)
    print("Scenario 2: Patrol and Respond - Results Plotting")
    print("="*60)
    
    # Find and load results
    results_dir = find_latest_results_dir()
    if results_dir is None:
        return
    
    df = load_results(results_dir)
    if df is None:
        return
    
    # Create output directory for plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots in: {plots_dir}")
    
    # Setup matplotlib
    setup_matplotlib()
    
    # Generate all standard plots
    create_all_plots(df, str(plots_dir))
    
    # Additional custom plots
    print("\nGenerating additional plots...")
    
    # Planning time comparison
    plt.figure(figsize=(10, 6))
    planning_times = df.groupby('planner')['average_planning_time'].agg(['mean', 'std'])
    
    bars = plt.bar(planning_times.index, planning_times['mean'], 
                  yerr=planning_times['std'], capsize=5, alpha=0.8, color='orange')
    plt.title('Planning Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Planning Time (s)', fontsize=12)
    plt.xlabel('Planner', fontsize=12)
    plt.yscale('log')
    
    # Add value labels
    for bar, mean_val in zip(bars, planning_times['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{mean_val:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'planning_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Duty completion efficiency plot
    plt.figure(figsize=(12, 8))
    
    # Calculate duty completion rates
    duty_efficiency = {}
    for planner in df['planner'].unique():
        planner_data = df[df['planner'] == planner]
        total_activations = planner_data['duty_activations'].sum()
        total_completions = planner_data['duty_completions'].sum()
        
        if total_activations > 0:
            completion_rate = total_completions / total_activations
        else:
            completion_rate = 0
            
        duty_efficiency[planner] = {
            'completion_rate': completion_rate,
            'avg_activations': planner_data['duty_activations'].mean(),
            'avg_completions': planner_data['duty_completions'].mean()
        }
    
    # Create subplot for duty efficiency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Completion rate
    planners = list(duty_efficiency.keys())
    completion_rates = [duty_efficiency[p]['completion_rate'] for p in planners]
    
    bars1 = ax1.bar(planners, completion_rates, alpha=0.8, color='green')
    ax1.set_title('Duty Completion Rate', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Completion Rate', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars1, completion_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Activations vs Completions
    activations = [duty_efficiency[p]['avg_activations'] for p in planners]
    completions = [duty_efficiency[p]['avg_completions'] for p in planners]
    
    x = np.arange(len(planners))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, activations, width, label='Activations', alpha=0.8, color='blue')
    bars3 = ax2.bar(x + width/2, completions, width, label='Completions', alpha=0.8, color='red')
    
    ax2.set_title('Duty Activations vs Completions', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(planners)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, activations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    for bar, val in zip(bars3, completions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'duty_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Success rate vs Duty completion scatter plot
    plt.figure(figsize=(10, 8))
    
    colors = {'LTL_MCTS': 'blue', 'LTL_Myopic': 'red', 'Info_Myopic': 'green', 
              'Random': 'orange', 'Passive': 'gray'}
    
    for planner in df['planner'].unique():
        planner_data = df[df['planner'] == planner]
        color = colors.get(planner, 'black')
        
        # Calculate completion rates for each run
        completion_rates = []
        for _, row in planner_data.iterrows():
            if row['duty_activations'] > 0:
                rate = row['duty_completions'] / row['duty_activations']
            else:
                rate = 0
            completion_rates.append(rate)
        
        plt.scatter(planner_data['success_rate'], completion_rates, 
                   c=color, label=planner, alpha=0.6, s=50)
    
    plt.xlabel('Success Rate', fontsize=12)
    plt.ylabel('Duty Completion Rate', fontsize=12)
    plt.title('Success Rate vs Duty Completion Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'success_vs_duty_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare summary data
    summary_data = []
    for planner in df['planner'].unique():
        planner_df = df[df['planner'] == planner]
        total_activations = planner_df['duty_activations'].sum()
        total_completions = planner_df['duty_completions'].sum()
        completion_rate = total_completions / total_activations if total_activations > 0 else 0
        
        summary_data.append([
            planner,
            f"{planner_df['success_rate'].mean():.3f} ± {planner_df['success_rate'].std():.3f}",
            f"{completion_rate:.3f}",
            f"{planner_df['duty_activations'].mean():.1f} ± {planner_df['duty_activations'].std():.1f}",
            f"{planner_df['duty_completions'].mean():.1f} ± {planner_df['duty_completions'].std():.1f}",
            f"{planner_df['average_planning_time'].mean()*1000:.2f} ± {planner_df['average_planning_time'].std()*1000:.2f}"
        ])
    
    columns = ['Planner', 'Success Rate', 'Duty Completion Rate', 'Avg Activations', 'Avg Completions', 'Planning Time (ms)']
    
    table = ax.table(cellText=summary_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Scenario 2: Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(plots_dir / 'performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ All plots generated successfully!")
    print(f"📁 Plots saved in: {plots_dir}")
    print("\nGenerated files:")
    for plot_file in sorted(plots_dir.glob("*.png")):
        print(f"  - {plot_file.name}")

if __name__ == "__main__":
    main()
