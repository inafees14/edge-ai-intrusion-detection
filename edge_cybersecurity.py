import time
import csv
import psutil
import subprocess
import json
from river import stream, tree, metrics

# --- CONFIGURATION ---
DATASET_FILE = "KDDTrain+.txt"
LOG_FILE = "edge_experiment_log.csv"
CHECK_INTERVAL = 500  # Log system stats every 500 samples

# Define column names for NSL-KDD (It has 43 columns, we map the basics)
# We treat it as a binary classification: Normal vs Attack
CONVERTERS = {"41": lambda x: "normal" if x == "normal" else "attack"}
# GENERATE COLUMN NAMES (0 to 42)
COL_NAMES = [str(i) for i in range(43)] 


def get_battery_status():
    """Fetches battery info from Android via Termux-API."""
    try:
        # Calls the 'termux-battery-status' command from the OS
        result = subprocess.run(['termux-battery-status'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = json.loads(result.stdout.decode('utf-8'))
        return data.get('percentage', 0), data.get('current', 0) # Current is roughly proportional to energy draw
    except:
        return 0, 0 # Fail silently if API not installed

def run_experiment():
    # 1. Initialize Model (Hoeffding Tree - The "Green AI" Model)
    model = tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=100,
        delta=1e-5
    )
    metric = metrics.Accuracy()
    
    # 2. Prepare Logging
    print(f"🚀 Starting Edge Cyber-Defense Experiment...")
    print(f"📂 Reading data from: {DATASET_FILE}")
    print(f"📊 Logging stats to: {LOG_FILE}")
    
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_Count', 'Accuracy', 'CPU_Percent', 'RAM_MB', 'Battery_Percent', 'Latency_ms'])

        # 3. Start Streaming Data
        start_time = time.time()
        process = psutil.Process()
        
        # Stream the CSV file (Simulating network packets arriving 1 by 1)
        # We skip column 41 (label) in X, and use it as Y
        for i, (x, y) in enumerate(stream.iter_csv(DATASET_FILE, target="41", converters=CONVERTERS, fieldnames=COL_NAMES)):

            
            loop_start = time.time()
            
            # --- THE ONLINE LEARNING STEP ---
            # 1. Predict (Is this packet an attack?)
            y_pred = model.predict_one(x)
            
            # 2. Learn (Update model immediately)
            model.learn_one(x, y)
            
            # 3. Update Accuracy
            metric.update(y, y_pred)
            
            loop_end = time.time()
            
            # --- MONITORING (The "Edge" Novelty) ---
            if i % CHECK_INTERVAL == 0:
                # Calculate Latency (Time to process one sample)
                latency = (loop_end - loop_start) * 1000 # to ms
                
                # Get System Stats
                cpu = psutil.cpu_percent()
                ram = process.memory_info().rss / (1024 * 1024) # to MB
                batt_pct, batt_curr = get_battery_status()
                
                # Print to screen
                print(f"Packet {i:6d} | Acc: {metric.get():.2%} | CPU: {cpu}% | RAM: {ram:.1f}MB | Latency: {latency:.2f}ms")
                
                # Save to CSV
                writer.writerow([i, metric.get(), cpu, ram, batt_pct, latency])

    print("-" * 50)
    print(f"✅ Experiment Complete.")
    print(f"Final Accuracy: {metric.get():.2%}")

if __name__ == "__main__":
    run_experiment()

