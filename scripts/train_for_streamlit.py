#!/usr/bin/env python
"""
Train model specifically for Streamlit deployment
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
import time
import gc
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Fix for Categorical error - Convert before importing
def fix_categorical_columns(df):
    """Convert categorical columns to string to avoid errors"""
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype(str)
    return df

from src.data.ingestion import BigDataIngestion
from src.data.validation import DataValidator
from src.features.feature_engineering import BigDataFeatureEngineer
from src.training.train import BigDataFraudModelTrainer

def find_dataset():
    """Find dataset in common locations"""
    possible_paths = [
        Path("data/raw/fraud_detection.csv"),
        Path("data/raw/transactions.csv"),
        Path("data/raw/fraud.csv"),
        Path("data/raw/data.csv"),
        Path("fraud_detection.csv"),
        Path("transactions.csv"),
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"✅ Found dataset at: {path}")
            return path
    
    return None

def main():
    """Train model for Streamlit deployment"""
    start_time = time.time()
    
    logger.info("🚀 Training Model for Streamlit Deployment")
    logger.info("="*60)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Find dataset
    dataset_path = find_dataset()
    
    if dataset_path is None:
        logger.error("❌ Dataset not found!")
        logger.info("\n📂 Please place your dataset in one of these locations:")
        logger.info("   1. data/raw/fraud_detection.csv")
        logger.info("   2. data/raw/transactions.csv")
        logger.info("   3. fraud_detection.csv (project root)")
        return
    
    # Update config
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['data']['raw_path'] = str(dataset_path)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        logger.info(f"✅ Updated config with dataset path: {dataset_path}")
    
    # Load data
    ingestion = BigDataIngestion()
    
    try:
        df = ingestion.load_data(use_processed=False)
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return
    
    if df is None or len(df) == 0:
        logger.error("❌ Failed to load data or dataset is empty")
        return
    
    # Fix categorical columns before processing
    df = fix_categorical_columns(df)
    
    logger.info(f"✅ Loaded {len(df):,} transactions")
    logger.info(f"📊 Columns: {df.columns.tolist()}")
    
    # Data statistics
    logger.info("\n📊 Data Statistics:")
    logger.info(f"  - Total transactions: {len(df):,}")
    logger.info(f"  - Fraud rate: {df['isFraud'].mean():.4%}")
    logger.info(f"  - Transaction types: {df['type'].value_counts().to_dict()}")
    logger.info(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Feature engineering
    logger.info("\n🔧 Creating Features...")
    fe = BigDataFeatureEngineer()
    
    try:
        df_featured = fe.create_all_features(df)
        logger.info(f"✅ Created {len(fe.feature_columns)} features")
    except Exception as e:
        logger.error(f"❌ Feature engineering error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save feature engineer
    logger.info("\n💾 Saving Feature Engineer...")
    try:
        fe.save('models/feature_engineer.pkl')
        logger.info("✅ Feature engineer saved successfully")
    except Exception as e:
        logger.error(f"❌ Error saving feature engineer: {e}")
        return
    
    # Train model
    logger.info("\n🤖 Training Model...")
    trainer = BigDataFraudModelTrainer()
    
    try:
        model, metrics, data = trainer.train_pipeline(df_featured)
        logger.info("✅ Model training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save model info
    logger.info("\n💾 Saving Model Information...")
    try:
        model_info = {
            'model_type': trainer.model_type,
            'features': fe.feature_columns,
            'metrics': metrics,
            'training_time': time.time() - start_time,
            'dataset_size': len(df),
            'fraud_rate': float(df['isFraud'].mean())
        }
        
        import json
        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info("✅ Model info saved")
    except Exception as e:
        logger.warning(f"⚠️ Could not save model info: {e}")
    
    # Clean up
    del df, df_featured
    gc.collect()
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("✅ Training Complete!")
    logger.info("="*60)
    logger.info(f"⏱️ Total time: {elapsed_time/60:.2f} minutes")
    logger.info(f"📊 Model Performance:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"   {metric}: {value:.4f}")
        else:
            logger.info(f"   {metric}: {value}")
    logger.info("\n📁 Files saved in 'models/' directory:")
    logger.info("   ✅ feature_engineer.pkl")
    logger.info("   ✅ xgboost_model.pkl")
    logger.info("   ✅ model_info.json")
    logger.info("\n🚀 Now you can run the Streamlit app:")
    logger.info("   streamlit run streamlit_app/app.py")
    logger.info("="*60)

if __name__ == "__main__":
    main()