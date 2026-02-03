import time
import csv
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from river import stream, metrics, preprocessing

# --- CONFIGURATION ---
DATASET_FILE = "KDDTrain+.txt"
LOG_FILE = "mlp_log.csv"
CHECK_INTERVAL = 1000

# Columns 1, 2, 3 are strings (protocol, service, flag). We must handle them.
COL_NAMES = [str(i) for i in range(43)]
CONVERTERS = {"41": lambda x: "normal" if x == "normal" else "attack"}

# --- DEFINE PYTORCH MODEL ---
class OnlineMLP(nn.Module):
    def __init__(self, input_dim):
        super(OnlineMLP, self).__init__()
        # Simple, fast architecture for Edge AI
        self.network = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def vectorize(x, scaler, input_dim=40):
    """
    Robust vectorizer that handles mixed String/Number data safely.
    """
    x_num = {} # Store numbers here
    x_cat = {} # Store strings here
    
    # 1. Separate Data Types
    for k, v in x.items():
        try:
            # Try converting to float
            val = float(v)
            x_num[k] = val
        except ValueError:
            # If it fails, it's a string (e.g., 'tcp')
            x_cat[k] = v

    # 2. Scale ONLY the numbers
    if x_num:
        scaler.learn_one(x_num)
        x_num_scaled = scaler.transform_one(x_num)
    else:
        x_num_scaled = {}

    # 3. Create Tensor (Hashing Trick)
    # This maps any number of features into a fixed-size vector (input_dim)
    vec = torch.zeros(input_dim)
    
    # Add scaled numbers
    for k, v in x_num_scaled.items():
        idx = hash(k) % input_dim
        vec[idx] += v
        
    # Add strings (Simple Hashing)
    for k, v in x_cat.items():
        # Hash the string value itself (e.g., hash('tcp'))
        idx = hash(v) % input_dim
        vec[idx] += 1.0
            
    return vec.unsqueeze(0)

def run():
    input_dim = 40 
    model = OnlineMLP(input_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    scaler = preprocessing.StandardScaler()
    metric = metrics.Accuracy()
    
    print(f"🚀 Starting Benchmark: Online PyTorch MLP...")
    
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_Count', 'Accuracy', 'Latency_ms', 'RAM_MB'])
        
        process = psutil.Process()
        
        for i, (x, y) in enumerate(stream.iter_csv(DATASET_FILE, target="41", converters=CONVERTERS, fieldnames=COL_NAMES)):
            
            loop_start = time.time()
            
            # 1. Prepare Data (Safe Vectorization)
            y_tensor = torch.tensor([[1.0]]) if y == "attack" else torch.tensor([[0.0]])
            x_tensor = vectorize(x, scaler, input_dim)
            
            # 2. Forward (Predict)
            optimizer.zero_grad()
            prediction = model(x_tensor)
            
            y_pred = "attack" if prediction.item() > 0.5 else "normal"
            metric.update(y, y_pred)
            
            # 3. Backward (Learn)
            loss = criterion(prediction, y_tensor)
            loss.backward()
            optimizer.step()
            
            # 4. Log Stats
            if i % CHECK_INTERVAL == 0:
                latency = (time.time() - loop_start) * 1000
                ram = process.memory_info().rss / (1024 * 1024)
                print(f"MLP Packet {i} | Acc: {metric.get():.2%} | Latency: {latency:.2f}ms")
                writer.writerow([i, metric.get(), latency, ram])

    print(f"✅ MLP Benchmark Complete. Saved to {LOG_FILE}")

if __name__ == "__main__":
    run()

