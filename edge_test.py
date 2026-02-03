import torch
import torch.nn as nn

# 1. Define a tiny model (simulating an Edge AI model)
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# 2. Create dummy data (Batch size 1, 10 features)
data = torch.randn(1, 10)

# 3. Run inference
output = model(data)

print("-" * 30)
print("✅ PyTorch is working on Termux!")
print(f"Input shape: {data.shape}")
print(f"Output: {output}")
print("-" * 30)

