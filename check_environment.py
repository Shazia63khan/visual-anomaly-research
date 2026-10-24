
import sys
import platform

import numpy as np
import torch

print("=== Visual Anomaly Research: Environment Check ===")
print(f"Python version: {sys.version.split()[0]}")
print(f"Operating system: {platform.system()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy version: {np.__version__}")

# Verify basic tensor computation
sample = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor test: {sample * 2}")
print("\nEnvironment check completed successfully.")
