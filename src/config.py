import os
from pathlib import Path

# --- PATHS ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_YAML_PATH = PROJECT_ROOT / "data.yaml"

# --- MODEL SETTINGS ---
MODEL_NAME = "yolov8n.pt"
IMAGE_SIZE = 640
PROJECT_NAME = "PPE_Detection_Pro"  # Changed name to indicate Pro settings

# --- TRAINING HYPERPARAMETERS (The Qwen/Perplexity Hybrid) ---
EPOCHS = 50           # 50 is plenty for a first "Pro" run (approx 1-2 hours)
BATCH_SIZE = 4    
WORKERS = 1   # Lower to 8 if you get "CUDA Out of Memory" (or RAM crash)
PATIENCE = 10         # Stop early if we don't improve (saves time)
LEARNING_RATE = 0.01  # Standard start point, let AdamW optimize it

# --- PRO AUGMENTATION (The "Secret Sauce") ---
# Based on the "Industry Consensus" values you found 
AUGMENT = True
MOSAIC = 1.0          # 100% Mosaic. Forces model to find small objects (Helmets)
MIXUP = 0.1           # Blends images slightly. Prevents overfitting on small data
COPY_PASTE = 0.3      # Copies objects to new spots. "Fake" data generation
DEGREES = 10.0        # Rotations (Simulates shaky cameras)
HSV_H = 0.015         # Color jitter (Simulates different lighting conditions)
HSV_S = 0.7           # Saturation variance
HSV_V = 0.4           # Value (Brightness) variance