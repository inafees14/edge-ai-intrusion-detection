import time
import csv
import psutil
from river import stream, forest, metrics

# --- CONFIGURATION ---
DATASET_FILE = "KDDTrain+.txt"
LOG_FILE = "rf_log.csv"  # Separate log file
CHECK_INTERVAL = 1000

COL_NAMES = [str(i) for i in range(43)]
CONVERTERS = {"41": lambda x: "normal" if x == "normal" else "attack"}

def run():
    # 1. Define Model (10 Trees)
    model = forest.ARFClassifier(n_models=10, seed=42)
    metric = metrics.Accuracy()
    
    print(f"🚀 Starting Benchmark: Adaptive Random Forest...")
    
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_Count', 'Accuracy', 'Latency_ms', 'RAM_MB'])
        
        process = psutil.Process()
        start_time = time.time()
        
        for i, (x, y) in enumerate(stream.iter_csv(DATASET_FILE, target="41", converters=CONVERTERS, fieldnames=COL_NAMES)):
            
            loop_start = time.time()
            
            # Predict & Learn
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            metric.update(y, y_pred)
            
            # Log Stats
            if i % CHECK_INTERVAL == 0:
                latency = (time.time() - loop_start) * 1000
                ram = process.memory_info().rss / (1024 * 1024)
                print(f"RF Packet {i} | Acc: {metric.get():.2%} | RAM: {ram:.1f}MB")
                writer.writerow([i, metric.get(), latency, ram])

    print(f"✅ RF Benchmark Complete. Saved to {LOG_FILE}")

if __name__ == "__main__":
    run()

