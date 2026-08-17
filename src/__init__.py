"""
Real-Time Fraud Detection System
"""

__version__ = "1.0.0"
__author__ = "Fraud Detection Team"

from src.data.ingestion import BigDataIngestion
from src.data.validation import DataValidator
from src.features.feature_engineering import BigDataFeatureEngineer
from src.training.train import BigDataFraudModelTrainer
from src.inference.predictor import FraudPredictor

__all__ = [
    'BigDataIngestion',
    'DataValidator',
    'BigDataFeatureEngineer',
    'BigDataFraudModelTrainer',
    'FraudPredictor'
]