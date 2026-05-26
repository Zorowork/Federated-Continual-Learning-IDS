import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json
import os
from matplotlib.gridspec import GridSpec

# Set professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Color palette
COLORS = {
    'naive': '#E74C3C',      # Red - shows problem
    'ewc': '#3498DB',        # Blue - good
    'replay': '#2ECC71',     # Green - good
    'ewc_replay': '#9B59B6'  # Purple - best
}

STRATEGY_NAMES = {
    'naive': 'Naive Fine-tuning',
    'ewc': 'EWC',
    'replay': 'Experience Replay',
    'ewc_replay': 'EWC + Replay'
}

def load_results(path='results/multidataset_fcl_final_results.json'):
    # Load experiment results
    with open(path, 'r') as f:
        results = json.load(f)
    return results


def plot_forgetting_comparison(results, save_path='results/fig_forgetting_comparison.png'):
    # Bar chart comparing forgetting across strategies
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['naive', 'ewc', 'replay', 'ewc_replay']
    forgetting = [results[s]['forgetting'] for s in strategies]
    labels = [STRATEGY_NAMES[s] for s in strategies]
    colors = [COLORS[s] for s in strategies]
    
    bars = ax.bar(labels, forgetting, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, forgetting):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement annotations
    naive_forgetting = forgetting[0]
    for i, (bar, val) in enumerate(zip(bars[1:], forgetting[1:]), 1):
        improvement = naive_forgetting - val
        ax.annotate(f'↓ {improvement:.1f}%\nreduction',
                    xy=(bar.get_x() + bar.get_width()/2., val/2),
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))
    
    ax.set_ylabel('Catastrophic Forgetting (%)', fontweight='bold')
    ax.set_title('Catastrophic Forgetting Across Continual Learning Strategies\n6G Federated IDS with Multi-Dataset Scenario', 
                 fontweight='bold')
    ax.set_ylim(0, max(forgetting) * 1.25)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line for naive baseline
    ax.axhline(y=naive_forgetting, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(3.4, naive_forgetting + 0.5, 'Naive Baseline', fontsize=9, color='red', style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_retention_comparison(results, save_path='results/fig_retention_comparison.png'):
    # Bar chart of Task 1 retention after Task 2 training
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['naive', 'ewc', 'replay', 'ewc_replay']
    task1_before = [results[s]['task1_accuracy'] for s in strategies]
    task1_after = [results[s]['task1_retention'] for s in strategies]
    labels = [STRATEGY_NAMES[s] for s in strategies]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, task1_before, width, label='Task 1 Accuracy (Before Task 2)',
                   color='#34495E', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, task1_after, width, label='Task 1 Retention (After Task 2)',
                   color=[COLORS[s] for s in strategies], edgecolor='black', linewidth=1.2)
    
    # Value labels
    for bars, values in [(bars1, task1_before), (bars2, task1_after)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Continual Learning Strategy', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Task 1 Knowledge Retention: Before vs. After Task 2 Training',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower center', ncol=2, framealpha=0.9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_strategy_performance_matrix(results, save_path='results/fig_performance_matrix.png'):
    """Heatmap showing all metrics for all strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['naive', 'ewc', 'replay', 'ewc_replay']
    metrics = ['task1_accuracy', 'task1_retention', 'full_accuracy', 'forgetting']
    metric_labels = ['Task 1\nAccuracy', 'Task 1\nRetention', 'Full Test\nAccuracy', 'Forgetting\n(lower=better)']
    
    data = np.array([[results[s][m] for m in metrics] for s in strategies])
    
    # Create custom colormap (green=good, red=bad)
    # Forgetting is inverted (lower is better)
    data_normalized = data.copy()
    data_normalized[:, -1] = 100 - data_normalized[:, -1]  # Invert forgetting
    
    im = ax.imshow(data_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_xticklabels(metric_labels, fontweight='bold')
    ax.set_yticklabels([STRATEGY_NAMES[s] for s in strategies], fontweight='bold')
    
    # Add value annotations
    for i in range(len(strategies)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha='center', va='center', color='black', fontsize=11, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.set_title('Performance Matrix: All Metrics Across Strategies',
                 fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance Score (higher is better)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_scatter_accuracy_vs_forgetting(results, save_path='results/fig_accuracy_vs_forgetting.png'):
    """Scatter plot: Full accuracy vs forgetting (lower-right is best)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['naive', 'ewc', 'replay', 'ewc_replay']
    
    for strategy in strategies:
        acc = results[strategy]['full_accuracy']
        forg = results[strategy]['forgetting']
        color = COLORS[strategy]
        
        ax.scatter(forg, acc, s=500, color=color, edgecolor='black', linewidth=2,
                   label=STRATEGY_NAMES[strategy], zorder=3)
        
        # Label each point
        ax.annotate(STRATEGY_NAMES[strategy],
                    xy=(forg, acc), xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2, edgecolor='none'))
    
    ax.set_xlabel('Forgetting (%) — lower is better', fontweight='bold')
    ax.set_ylabel('Full Test Accuracy (%) — higher is better', fontweight='bold')
    ax.set_title('Trade-off: Full Test Accuracy vs. Forgetting\n(Ideal: top-left corner)',
                 fontweight='bold')
    
    # Add ideal region indicator
    ax.axvline(x=10, color='green', linestyle=':', alpha=0.3)
    ax.axhline(y=70, color='green', linestyle=':', alpha=0.3)
    ax.text(1, 75, '← Ideal Region', fontsize=10, color='green', style='italic', fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, max([results[s]['forgetting'] for s in strategies]) + 5)
    ax.set_ylim(40, max([results[s]['full_accuracy'] for s in strategies]) + 10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_improvement_over_naive(results, save_path='results/fig_improvement.png'):
    """Visualize percentage improvement over naive baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['ewc', 'replay', 'ewc_replay']
    naive_forgetting = results['naive']['forgetting']
    
    improvements = []
    for s in strategies:
        reduction = naive_forgetting - results[s]['forgetting']
        pct_improvement = (reduction / naive_forgetting) * 100
        improvements.append(pct_improvement)
    
    labels = [STRATEGY_NAMES[s] for s in strategies]
    colors = [COLORS[s] for s in strategies]
    
    bars = ax.barh(labels, improvements, color=colors, edgecolor='black', linewidth=1.5)
    
    # Value labels
    for bar, val in zip(bars, improvements):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}% reduction',
                ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Forgetting Reduction vs. Naive Baseline (%)', fontweight='bold')
    ax.set_title(f'Continual Learning Effectiveness\nReduction in Forgetting from Naive Baseline ({naive_forgetting:.1f}%)',
                 fontweight='bold')
    ax.set_xlim(0, max(improvements) * 1.2)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Best at top
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_summary_figure(results, save_path='results/fig_comprehensive_summary.png'):
    """Comprehensive 4-panel summary figure for dissertation."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    strategies = ['naive', 'ewc', 'replay', 'ewc_replay']
    labels = [STRATEGY_NAMES[s] for s in strategies]
    colors = [COLORS[s] for s in strategies]
    

    # Panel 1: Forgetting Comparison

    ax1 = fig.add_subplot(gs[0, 0])
    forgetting = [results[s]['forgetting'] for s in strategies]
    bars = ax1.bar(labels, forgetting, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, forgetting):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Forgetting (%)', fontweight='bold')
    ax1.set_title('(a) Catastrophic Forgetting by Strategy', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)
    

    # Panel 2: Task 1 Retention

    ax2 = fig.add_subplot(gs[0, 1])
    task1_before = [results[s]['task1_accuracy'] for s in strategies]
    task1_after = [results[s]['task1_retention'] for s in strategies]
    
    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width/2, task1_before, width, label='Before Task 2',
            color='#7F8C8D', edgecolor='black', linewidth=1.2)
    ax2.bar(x + width/2, task1_after, width, label='After Task 2',
            color=colors, edgecolor='black', linewidth=1.2)
    
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('(b) Task 1 Knowledge: Before vs. After Task 2', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)


    # Panel 3: Trade-off Scatter

    ax3 = fig.add_subplot(gs[1, 0])
    for strategy in strategies:
        acc = results[strategy]['full_accuracy']
        forg = results[strategy]['forgetting']
        ax3.scatter(forg, acc, s=400, color=COLORS[strategy], edgecolor='black', linewidth=2,
                    zorder=3)
        ax3.annotate(STRATEGY_NAMES[strategy],
                     xy=(forg, acc), xytext=(8, 8),
                     textcoords='offset points',
                     fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Forgetting (%) — lower is better', fontweight='bold')
    ax3.set_ylabel('Full Test Accuracy (%)', fontweight='bold')
    ax3.set_title('(c) Accuracy vs. Forgetting Trade-off', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    

    # Panel 4: Improvement over Naive

    ax4 = fig.add_subplot(gs[1, 1])
    naive_forgetting = results['naive']['forgetting']
    cl_strategies = ['ewc', 'replay', 'ewc_replay']
    cl_labels = [STRATEGY_NAMES[s] for s in cl_strategies]
    cl_colors = [COLORS[s] for s in cl_strategies]
    
    improvements = []
    for s in cl_strategies:
        reduction = naive_forgetting - results[s]['forgetting']
        pct = (reduction / naive_forgetting) * 100
        improvements.append(pct)
    
    bars = ax4.barh(cl_labels, improvements, color=cl_colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, improvements):
        ax4.text(val + 1, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax4.set_xlabel('Forgetting Reduction (%)', fontweight='bold')
    ax4.set_title('(d) Improvement Over Naive Baseline', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    

    # Main title

    fig.suptitle('Federated Continual Learning for 6G Intrusion Detection: Multi-Dataset Analysis',
                 fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_client_breakdown(save_path='results/fig_client_breakdown.png'):
    """Show per-client performance for EWC+Replay (best strategy)."""
    # Hard-coded from logs (best strategy per-client results)
    client_data = {
        'eMBB-1\n(CICIDS2017)': {'task1': 88.76, 'retention': 74.75, 'dataset': 'cicids2017'},
        'eMBB-2\n(CICIDS2017)': {'task1': 90.69, 'retention': 80.18, 'dataset': 'cicids2017'},
        'URLLC-1\n(NSL-KDD)':   {'task1': 84.43, 'retention': 46.61, 'dataset': 'nslkdd'},
        'URLLC-2\n(NSL-KDD)':   {'task1': 84.49, 'retention': 60.40, 'dataset': 'nslkdd'},
        'mMTC\n(CICIoT2023)':   {'task1': 73.40, 'retention': 72.05, 'dataset': 'ciciot2023'}
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    clients = list(client_data.keys())
    task1_accs = [client_data[c]['task1'] for c in clients]
    retention_accs = [client_data[c]['retention'] for c in clients]
    
    # Color by dataset
    dataset_colors = {
        'cicids2017': '#3498DB',
        'nslkdd': '#E67E22',
        'ciciot2023': '#27AE60'
    }
    
    x = np.arange(len(clients))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, task1_accs, width, 
                   label='Task 1 Accuracy (Before Task 2)',
                   color='#95A5A6', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, retention_accs, width,
                   label='Task 1 Retention (After Task 2)',
                   color=[dataset_colors[client_data[c]['dataset']] for c in clients],
                   edgecolor='black', linewidth=1.2)
    
    # Value labels
    for bars, values in [(bars1, task1_accs), (bars2, retention_accs)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Add forgetting annotations
    for i, (t1, ret) in enumerate(zip(task1_accs, retention_accs)):
        forgetting = t1 - ret
        ax.annotate(f'Forgetting:\n{forgetting:.1f}%',
                    xy=(i, max(t1, ret) + 5),
                    ha='center', fontsize=8, style='italic', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.8, edgecolor='red'))
    
    ax.set_xlabel('6G Network Slice (Dataset)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Per-Client Performance: EWC + Replay (Best Strategy)\nAcross 6G Network Slices with Heterogeneous Datasets',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clients)
    ax.legend(loc='lower center', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)
    
    # Add dataset legend
    dataset_patches = [
        mpatches.Patch(color='#3498DB', label='CICIDS2017'),
        mpatches.Patch(color='#E67E22', label='NSL-KDD'),
        mpatches.Patch(color='#27AE60', label='CICIoT2023')
    ]
    legend2 = ax.legend(handles=dataset_patches, loc='upper right',
                        title='Dataset', framealpha=0.9)
    ax.add_artist(legend2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_6g_architecture(save_path='results/fig_6g_architecture.png'):
    # Conceptual diagram of the 6G federated continual learning system
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Federated Continual Learning Architecture for 6G Network Slices',
            fontsize=15, fontweight='bold', ha='center')
    
    # Central server
    server_color = '#2C3E50'
    server = mpatches.FancyBboxPatch((5.5, 5), 3, 1.2, boxstyle='round,pad=0.1',
                                      facecolor=server_color, edgecolor='black', linewidth=2)
    ax.add_patch(server)
    ax.text(7, 5.6, 'Central Server', color='white', ha='center', fontweight='bold', fontsize=12)
    ax.text(7, 5.2, 'FedBN Aggregation', color='white', ha='center', fontsize=10, style='italic')
    
    # Client boxes
    clients = [
        {'name': 'eMBB-1', 'dataset': 'CICIDS2017', 'x': 0.5, 'color': '#3498DB'},
        {'name': 'eMBB-2', 'dataset': 'CICIDS2017', 'x': 3.5, 'color': '#3498DB'},
        {'name': 'URLLC-1', 'dataset': 'NSL-KDD', 'x': 6.5, 'color': '#E67E22'},
        {'name': 'URLLC-2', 'dataset': 'NSL-KDD', 'x': 9.5, 'color': '#E67E22'},
        {'name': 'mMTC', 'dataset': 'CICIoT2023', 'x': 12.5, 'color': '#27AE60'}
    ]
    
    for client in clients:
        # Client box
        cbox = mpatches.FancyBboxPatch((client['x'], 2), 1.2, 1.5, boxstyle='round,pad=0.1',
                                        facecolor=client['color'], edgecolor='black', linewidth=2)
        ax.add_patch(cbox)
        ax.text(client['x'] + 0.6, 3.1, client['name'], color='white', 
                ha='center', fontweight='bold', fontsize=10)
        ax.text(client['x'] + 0.6, 2.7, client['dataset'], color='white',
                ha='center', fontsize=8, style='italic')
        ax.text(client['x'] + 0.6, 2.3, '+ EWC', color='white',
                ha='center', fontsize=8)
        ax.text(client['x'] + 0.6, 2.1, '+ Replay', color='white',
                ha='center', fontsize=8)
        
        # Arrow from client to server (upload weights)
        ax.annotate('', xy=(7, 5), xytext=(client['x'] + 0.6, 3.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # Data below client
        ax.text(client['x'] + 0.6, 1.6, '▼ Local Data', ha='center', fontsize=8, style='italic')
        ax.text(client['x'] + 0.6, 1.2, '• Task 1: Initial attacks', ha='center', fontsize=7)
        ax.text(client['x'] + 0.6, 0.95, '• Task 2: New attacks', ha='center', fontsize=7)
    
    # Legend for arrows
    ax.text(2, 4.2, 'Upload: Model Weights\n(no raw data shared)', ha='center', fontsize=9,
            style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Architecture components
    ax.text(7, 6.7, 'Shared: Trunk weights (federated)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    ax.text(7, 0.5, 'Local: BatchNorm stats + Classification heads', ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")


def create_results_csv(results, save_path='results/final_results_table.csv'):
    """Create a clean CSV of final results for dissertation tables."""
    data = []
    for strategy_key, strategy_name in STRATEGY_NAMES.items():
        res = results[strategy_key]
        data.append({
            'Strategy': strategy_name,
            'Task 1 Accuracy (%)': f"{res['task1_accuracy']:.2f}",
            'Task 2 Accuracy (%)': f"{res['task2_accuracy']:.2f}",
            'Full Test Accuracy (%)': f"{res['full_accuracy']:.2f}",
            'Task 1 Retention (%)': f"{res['task1_retention']:.2f}",
            'Forgetting (%)': f"{res['forgetting']:.2f}"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"✓ Saved: {save_path}")
    return df


def main():
    """Generate all visualizations."""
    print("="*70)
    print("GENERATING VISUALIZATIONS FOR FINAL RESULTS")
    print("="*70)
    
    # Load results
    try:
        results = load_results('results/multidataset_fcl_final_results.json')
        print(f"\n✓ Loaded results from: results/multidataset_fcl_final_results.json")
    except FileNotFoundError:
        print("❌ Results file not found! Please check the path.")
        return
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    print("\nGenerating figures...")
    print("-"*70)
    
    # Individual figures
    plot_forgetting_comparison(results)
    plot_retention_comparison(results)
    plot_strategy_performance_matrix(results)
    plot_scatter_accuracy_vs_forgetting(results)
    plot_improvement_over_naive(results)
    plot_client_breakdown()
    plot_6g_architecture()
    
    # Summary figure
    plot_summary_figure(results)
    
    # Results table
    print("\nGenerating results table...")
    print("-"*70)
    df = create_results_csv(results)
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\nFinal Results Table:")
    print("-"*70)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("Files saved to results/:")
    print("="*70)
    output_files = [
        'fig_forgetting_comparison.png',
        'fig_retention_comparison.png', 
        'fig_performance_matrix.png',
        'fig_accuracy_vs_forgetting.png',
        'fig_improvement.png',
        'fig_client_breakdown.png',
        'fig_6g_architecture.png',
        'fig_comprehensive_summary.png',
        'final_results_table.csv'
    ]
    for f in output_files:
        print(f"  ✓ {f}")


if __name__ == '__main__':
    main()