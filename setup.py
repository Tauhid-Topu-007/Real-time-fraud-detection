from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="real-time-fraud-detection",
    version="1.0.0",
    author="Fraud Detection Team",
    description="Real-Time Fraud Detection System for Bank Transactions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/real-time-fraud-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "pyyaml>=6.0.0",
        "joblib>=1.3.0",
        "loguru>=0.7.0",
    ],
)