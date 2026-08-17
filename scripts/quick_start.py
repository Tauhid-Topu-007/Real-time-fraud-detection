#!/usr/bin/env python
"""
Quick start script for fraud detection system
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
import time
import gc

sys.path.append(str(Path(__file__).parent.parent))

from src.data.ingestion import BigDataIngestion
from src.data.validation import DataValidator
from src.features.feature_engineering import BigDataFeatureEngineer
from src.training.train import BigDataFraudModelTrainer

def main():
    """Main execution"""
    start_time = time.time()
    
    logger.info("🚀 Starting Fraud Detection System")
    logger.info("="*60)
    
    ingestion = BigDataIngestion()
    df = ingestion.load_data(use_processed=True)
    
    logger.info(f"Loaded {len(df):,} transactions")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    logger.info("\n✅ Validating Data...")
    validator = DataValidator()
    validator.validate_schema(df)
    is_valid, issues = validator.validate_values(df)
    
    if not is_valid:
        logger.warning(f"Validation issues: {issues}")
    
    stats = validator.get_data_stats(df)
    logger.info(f"Fraud rate: {stats['fraud_rate']:.4%}")
    logger.info(f"Transaction types: {stats['transaction_types']}")
    
    logger.info("\n🔧 Creating Features...")
    fe = BigDataFeatureEngineer()
    df_featured = fe.create_all_features(df)
    
    logger.info("\n💾 Saving Feature Engineer...")
    fe.save('models/feature_engineer.pkl')
    
    logger.info("\n🤖 Training Model...")
    trainer = BigDataFraudModelTrainer()
    model, metrics, data = trainer.train_pipeline(df_featured)
    
    del df, df_featured
    gc.collect()
    
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("✅ Training Complete!")
    logger.info("="*60)
    logger.info(f"⏱️ Total time: {elapsed_time/60:.2f} minutes")
    logger.info(f"📊 Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"   {metric}: {value:.4f}")
    logger.info("\n📁 Files saved:")
    logger.info("   - models/feature_engineer.pkl")
    logger.info("   - models/xgboost_model.pkl")
    logger.info("   - configs/config.yaml (updated)")
    logger.info("="*60)

if __name__ == "__main__":
    main()