import os
from pathlib import Path

print("="*50)
print("📊 Checking Model File Sizes")
print("="*50)

models_dir = Path("models")

if not models_dir.exists():
    print("❌ models directory not found!")
    exit()

total_size = 0
for file in models_dir.glob("*"):
    size = file.stat().st_size / (1024 * 1024)  # MB
    total_size += size
    print(f"{file.name}: {size:.2f} MB")

print("-"*50)
print(f"Total Size: {total_size:.2f} MB")

if total_size > 100:
    print("⚠️ WARNING: Total size > 100MB!")
    print("   Streamlit Cloud may have issues loading this.")
    print("   Consider using Git LFS or Cloud Storage.")
else:
    print("✅ Total size is within limits.")