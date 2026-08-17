#!/usr/bin/env python
"""
Simple script to run model training
"""
import subprocess
import sys
from pathlib import Path

def run_training():
    """Run the training script"""
    print("🚀 Starting model training...")
    print("="*50)
    
    dataset_path = Path("data/raw/fraud_detection.csv")
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please place your fraud_detection.csv in data/raw/ directory")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    
    try:
        subprocess.run([sys.executable, "scripts/train_for_streamlit.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_training()