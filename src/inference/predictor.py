import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from loguru import logger
import yaml
from typing import Dict, Any, Optional, List

class FraudPredictor:
    """Fraud prediction service"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = Path("models")
        self.model_version = "v1.0.0"
        
        self.thresholds = self.config.get('thresholds', {})
        self.optimal_threshold = self.thresholds.get('optimal', 0.5)
        
        self.feature_engineer = None
        self.model = None
        
        self.load_models()
        
        self.predictions_count = 0
        self.fraud_alerts = 0
        
        logger.info(f"FraudPredictor initialized with threshold: {self.optimal_threshold}")
    
    def load_models(self):
        """Load model and feature engineer"""
        try:
            fe_path = self.model_path / "feature_engineer.pkl"
            if fe_path.exists():
                self.feature_engineer = joblib.load(fe_path)
                logger.info("Feature engineer loaded")
            else:
                logger.warning("No feature engineer found, creating new one")
                from src.features.feature_engineering import BigDataFeatureEngineer
                self.feature_engineer = BigDataFeatureEngineer()
            
            model_path = self.model_path / "xgboost_model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            else:
                for model_type in ['xgboost', 'lightgbm', 'random_forest']:
                    model_path = self.model_path / f"{model_type}_model.pkl"
                    if model_path.exists():
                        self.model = joblib.load(model_path)
                        logger.info(f"Loaded {model_type} model")
                        break
                else:
                    raise FileNotFoundError("No model found in models directory")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def prepare_features(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for prediction"""
        df = pd.DataFrame([transaction])
        
        required_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        df_featured = self.feature_engineer.create_all_features(df)
        
        if self.feature_engineer.feature_columns:
            available_cols = [col for col in self.feature_engineer.feature_columns 
                            if col in df_featured.columns]
            df_featured = df_featured[available_cols]
            
            for col in self.feature_engineer.feature_columns:
                if col not in df_featured.columns:
                    df_featured[col] = 0
        
        df_scaled = self.feature_engineer.scale_features(df_featured, fit=False)
        
        return df_scaled
    
    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud for a single transaction"""
        try:
            X = self.prepare_features(transaction)
            fraud_prob = float(self.model.predict_proba(X)[0][1])
            
            risk_level, decision = self.get_risk_decision(fraud_prob)
            
            self.predictions_count += 1
            if risk_level == 'HIGH':
                self.fraud_alerts += 1
            
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_names = X.columns
                top_indices = np.argsort(importance)[-5:][::-1]
                feature_importance = [
                    {'feature': feature_names[i], 'importance': float(importance[i])} 
                    for i in top_indices
                ]
            
            return {
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'fraud_probability': round(fraud_prob, 4),
                'risk_level': risk_level,
                'decision': decision,
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_version,
                'features_used': len(self.feature_engineer.feature_columns) 
                                if self.feature_engineer.feature_columns else 0,
                'top_features': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_risk_decision(self, fraud_prob: float) -> tuple:
        """Determine risk level and decision based on probability"""
        approve_threshold = self.thresholds.get('approve', 0.3)
        
        if fraud_prob < approve_threshold:
            return 'LOW', 'APPROVE'
        elif fraud_prob < self.optimal_threshold:
            return 'MEDIUM', 'REVIEW'
        else:
            return 'HIGH', 'BLOCK'
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict for multiple transactions"""
        results = []
        for txn in transactions:
            result = self.predict_single(txn)
            results.append(result)
        return results
    
    def log_prediction(self, transaction: Dict[str, Any], result: Dict[str, Any]):
        """Log prediction for monitoring"""
        logger.info(f"Transaction {transaction.get('transaction_id')}: "
                   f"Probability={result['fraud_probability']:.3f}, "
                   f"Decision={result['decision']}, "
                   f"Risk={result['risk_level']}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get prediction metrics"""
        return {
            'total_predictions': self.predictions_count,
            'fraud_alerts': self.fraud_alerts,
            'alert_rate': self.fraud_alerts / self.predictions_count 
                         if self.predictions_count > 0 else 0,
            'model_version': self.model_version,
            'threshold': self.optimal_threshold,
            'timestamp': datetime.now().isoformat()
        }