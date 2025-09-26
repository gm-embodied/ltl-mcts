#!/usr/bin/env python3
"""
Scenario 1 Results Plotting Script
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
    plot_environment_layout, plot_ospa_comparison, 
    plot_patrol_efficiency, create_all_plots, setup_matplotlib
)

def find_latest_results_dir():
    """Find the most recent results directory"""
    results_base = Path("results")
    if not results_base.exists():
        print("No results directory found. Please run experiments first.")
        return None
    
    # Find all scenario1 result directories
    scenario1_dirs = [d for d in results_base.iterdir() 
                     if d.is_dir() and d.name.startswith("scenario1_")]
    
    if not scenario1_dirs:
        print("No scenario1 results found. Please run experiments first.")
        return None
    
    # Return the most recent one
    latest_dir = max(scenario1_dirs, key=lambda d: d.stat().st_mtime)
    print(f"Using results from: {latest_dir}")
    return latest_dir

def load_results(results_dir: Path):
    """Load experimental results"""
    results_file = results_dir / "scenario1_results.csv"
    
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
    print("Scenario 1: Persistent Surveillance - Results Plotting")
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
    
    # Success rate vs OSPA scatter plot
    plt.figure(figsize=(10, 8))
    
    colors = {'LTL_MCTS': 'blue', 'LTL_Myopic': 'red', 'Info_Myopic': 'green', 
              'Random': 'orange', 'Passive': 'gray', 'Discretized_POMCP': 'purple'}
    
    for planner in df['planner'].unique():
        planner_data = df[df['planner'] == planner]
        color = colors.get(planner, 'black')
        
        plt.scatter(planner_data['success_rate'], planner_data['mean_ospa_error'], 
                   c=color, label=planner, alpha=0.6, s=50)
    
    plt.xlabel('Success Rate', fontsize=12)
    plt.ylabel('Mean OSPA Error (m)', fontsize=12)
    plt.title('Success Rate vs OSPA Error', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'success_vs_ospa_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare summary data
    summary_data = []
    for planner in df['planner'].unique():
        planner_df = df[df['planner'] == planner]
        summary_data.append([
            planner,
            f"{planner_df['success_rate'].mean():.3f} ± {planner_df['success_rate'].std():.3f}",
            f"{planner_df['mean_ospa_error'].mean():.1f} ± {planner_df['mean_ospa_error'].std():.1f}",
            f"{planner_df['patrol_efficiency'].mean():.1f} ± {planner_df['patrol_efficiency'].std():.1f}",
            f"{planner_df['average_planning_time'].mean()*1000:.2f} ± {planner_df['average_planning_time'].std()*1000:.2f}"
        ])
    
    columns = ['Planner', 'Success Rate', 'OSPA Error (m)', 'Patrol Efficiency', 'Planning Time (ms)']
    
    table = ax.table(cellText=summary_data, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Scenario 1: Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(plots_dir / 'performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ All plots generated successfully!")
    print(f"📁 Plots saved in: {plots_dir}")
    print("\nGenerated files:")
    for plot_file in sorted(plots_dir.glob("*.png")):
        print(f"  - {plot_file.name}")

if __name__ == "__main__":
    main()
