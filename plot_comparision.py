import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# List your files and the label you want on the graph
FILES = [
    ("edge_experiment_log.csv", "Hoeffding Tree (Yours)"),
    ("rf_log.csv", "Adaptive Random Forest"),
    ("mlp_log.csv", "Online MLP (Deep Learning)"),
    ("logreg_log.csv", "Logistic Regression")
]

# Colors for consistency
COLORS = {
    "Hoeffding Tree (Yours)": "green",
    "Adaptive Random Forest": "blue",
    "Online MLP (Deep Learning)": "red",
    "Logistic Regression": "orange"
}

def load_data():
    dfs = []
    for filename, label in FILES:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['Model'] = label
                dfs.append(df)
                print(f"✅ Loaded {filename}")
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
        else:
            print(f"❌ File not found: {filename} (Skipping)")
    return dfs

def plot_comparison(dfs, metric_col, title, ylabel, filename, y_limit=None):
    plt.figure(figsize=(10, 6))
    
    has_data = False
    for df in dfs:
        if metric_col in df.columns:
            label = df['Model'].iloc[0]
            # Rolling average to smooth out the jitter in Latency/RAM
            if metric_col in ['Latency_ms', 'RAM_MB']:
                data = df[metric_col].rolling(window=5).mean()
            else:
                data = df[metric_col]
            
            plt.plot(df['Sample_Count'], data, label=label, color=COLORS.get(label, 'black'), alpha=0.8)
            has_data = True
    
    if has_data:
        plt.title(title)
        plt.xlabel('Packets Processed')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        if y_limit:
            plt.ylim(y_limit)
        
        plt.tight_layout()
        plt.savefig(filename)
        print(f"📊 Saved Comparison: {filename}")
        plt.close()
    else:
        print(f"⚠️ No data found for {metric_col}, skipping plot.")

def plot_individual_dashboard(df):
    model_name = df['Model'].iloc[0]
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    
    # Determine available metrics
    cols = ['Accuracy', 'Latency_ms', 'RAM_MB', 'CPU_Percent', 'Battery_Percent']
    available_cols = [c for c in cols if c in df.columns]
    
    num_plots = len(available_cols)
    if num_plots == 0: return

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes] # Handle single plot case
    
    for i, col in enumerate(available_cols):
        ax = axes[i]
        color = COLORS.get(model_name, 'black')
        
        # Plot raw data
        ax.plot(df['Sample_Count'], df[col], color=color, alpha=0.6, label='Raw')
        
        # Add trend line (Moving Average)
        if len(df) > 10:
            ax.plot(df['Sample_Count'], df[col].rolling(window=10).mean(), color='black', linestyle='--', linewidth=1, label='Trend')
        
        ax.set_ylabel(col.replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Specific formatting
        if col == 'Accuracy':
            ax.set_ylim(0.5, 1.0)
    
    axes[-1].set_xlabel('Packets Processed')
    fig.suptitle(f"Dashboard: {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"dashboard_{safe_name}.png")
    print(f"📈 Saved Dashboard: dashboard_{safe_name}.png")
    plt.close()

# --- MAIN EXECUTION ---
print("🚀 Generatng Full Research Report Plots...")
all_dfs = load_data()

if all_dfs:
    # 1. Generate Comparisons
    plot_comparison(all_dfs, 'Accuracy', 'Accuracy Benchmark (Higher is Better)', 'Accuracy', 'compare_accuracy.png', y_limit=(0.8, 1.0))
    plot_comparison(all_dfs, 'RAM_MB', 'Memory Footprint (Lower is Better)', 'RAM Usage (MB)', 'compare_ram.png')
    plot_comparison(all_dfs, 'Latency_ms', 'Inference Speed (Lower is Better)', 'Latency (ms)', 'compare_latency.png')
    
    # CPU Comparison (Likely only Hoeffding Tree has this, but if others do, it handles it)
    plot_comparison(all_dfs, 'CPU_Percent', 'CPU Efficiency', 'CPU Usage (%)', 'compare_cpu.png')

    # 2. Generate Individual Dashboards
    for df in all_dfs:
        plot_individual_dashboard(df)

    print("\n✅ All graphics generated successfully!")
else:
    print("❌ No log files found. Did you run the experiments?")

