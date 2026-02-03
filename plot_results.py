import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Data
print("📊 Reading experiment logs...")
try:
    df = pd.read_csv("edge_experiment_log.csv")
except FileNotFoundError:
    print("❌ Error: Log file not found. Did you run the experiment?")
    exit()

# 2. Setup the Graph
plt.figure(figsize=(10, 6))

# Plot Accuracy (Left Axis)
ax1 = plt.gca()
line1, = ax1.plot(df['Sample_Count'], df['Accuracy'], color='blue', label='Accuracy')
ax1.set_xlabel('Packets Processed')
ax1.set_ylabel('Accuracy', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0.5, 1.0) # Scale accuracy from 50% to 100%

# Plot CPU Usage (Right Axis)
ax2 = ax1.twinx()
line2, = ax2.plot(df['Sample_Count'], df['CPU_Percent'], color='red', alpha=0.3, label='CPU Usage')
ax2.set_ylabel('CPU Usage (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 100)

# Add Titles
plt.title('Edge AI Performance: Online Learning on Android')
plt.grid(True)

# 3. Save to Android Storage
# We save it to the /mnt/sdcard/Download folder so you can see it in your Gallery!
output_path = "edge_results.png"
plt.savefig(output_path)

print(f"✅ Graph saved to: {output_path}")
print("👉 Go check your Phone Gallery (or Downloads folder)!")

