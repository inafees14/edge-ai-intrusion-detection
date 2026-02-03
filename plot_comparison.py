import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("📊 Loading Log Files...")

# 1. Load Each File and Assign a Label
try:
    # Your Main Model (The one you already ran)
    df_hat = pd.read_csv("edge_experiment_log.csv")
    df_hat['Model'] = 'Hoeffding Tree (Yours)'
    
    # The New Benchmarks
    df_rf = pd.read_csv("rf_log.csv")
    df_rf['Model'] = 'Adaptive Random Forest'
    
    df_mlp = pd.read_csv("mlp_log.csv")
    df_mlp['Model'] = 'Online MLP (Deep Learning)'
    
    df_lr = pd.read_csv("logreg_log.csv")
    df_lr['Model'] = 'Logistic Regression'

    # 2. Combine them
    df_final = pd.concat([df_hat, df_rf, df_mlp, df_lr])

    # 3. Plot Accuracy Comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_final, x='Sample_Count', y='Accuracy', hue='Model')
    plt.title('Performance Benchmark: Edge Intrusion Detection')
    plt.ylabel('Accuracy')
    plt.xlabel('Packets Processed')
    plt.grid(True)
    plt.ylim(0.80, 1.0) # Zoom in to see the difference
    plt.savefig("final_comparison.png")
    
    print("✅ Graph saved to: final_comparison.png")

except FileNotFoundError as e:
    print(f"❌ Missing File: {e.filename}")
    print("Make sure you have run ALL 4 separate scripts first!")

