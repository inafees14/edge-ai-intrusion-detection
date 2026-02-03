import time
import csv
import psutil
from river import stream, linear_model, metrics, compose, preprocessing

# --- CONFIGURATION ---
DATASET_FILE = "KDDTrain+.txt"
LOG_FILE = "logreg_log.csv"
CHECK_INTERVAL = 1000

COL_NAMES = [str(i) for i in range(43)]
CONVERTERS = {"41": lambda x: "normal" if x == "normal" else "attack"}

def run():
    # 1. Define Model (Pipeline)
    # OneHotEncoder handles text features ("tcp", "http")
    # StandardScaler scales numbers (crucial for LogReg)
    model = compose.Pipeline(
        preprocessing.OneHotEncoder(),
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression()
    )
    
    metric = metrics.Accuracy()
    
    print(f"🚀 Starting Benchmark: Logistic Regression...")
    
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_Count', 'Accuracy', 'Latency_ms', 'RAM_MB'])
        
        process = psutil.Process()
        
        for i, (x, y) in enumerate(stream.iter_csv(DATASET_FILE, target="41", converters=CONVERTERS, fieldnames=COL_NAMES)):
            
            loop_start = time.time()
            
            # --- FIX: Convert Label to Boolean (True/False) ---
            # Logistic Regression needs True(1) or False(0), not "attack"/"normal"
            y_bool = (y == "attack")
            
            # Predict
            y_pred = model.predict_one(x)
            
            # Learn
            model.learn_one(x, y_bool)
            
            # Update Metrics
            metric.update(y_bool, y_pred)
            
            # Log Stats
            if i % CHECK_INTERVAL == 0:
                latency = (time.time() - loop_start) * 1000
                ram = process.memory_info().rss / (1024 * 1024)
                print(f"LogReg Packet {i} | Acc: {metric.get():.2%} | Latency: {latency:.3f}ms")
                writer.writerow([i, metric.get(), latency, ram])

    print(f"✅ LogReg Benchmark Complete. Saved to {LOG_FILE}")

if __name__ == "__main__":
    run()

