import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml
import sys
import json

class ModelLoader:
    """Load and manage ML models for Streamlit app"""
    
    def __init__(self, model_path="models", config_path="configs/config.yaml"):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        self.model = None
        self.feature_engineer = None
        self.scaler = None
        self.config = None
        self.feature_columns = None
        self.model_info = None
        
        self.load_config()
        self.load_models()
    
    def load_config(self):
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Config loaded successfully")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}
    
    def load_models(self):
        """Load all models"""
        try:
            info_path = self.model_path / "model_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
                logger.info("Model info loaded")
            
            fe_path = self.model_path / "feature_engineer.pkl"
            if fe_path.exists():
                self.feature_engineer = joblib.load(fe_path)
                self.feature_columns = self.feature_engineer.feature_columns
                logger.info("Feature engineer loaded")
            else:
                logger.warning("Feature engineer not found")
            
            model_files = list(self.model_path.glob("*_model.pkl"))
            if model_files:
                model_file = model_files[0]
                self.model = joblib.load(model_file)
                logger.info(f"Model loaded from {model_file.name}")
            else:
                logger.error("No model found")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def prepare_features(self, transaction_dict):
        """Prepare features for prediction"""
        try:
            df = pd.DataFrame([transaction_dict])
            
            required_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                            'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
            
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0
            
            if self.feature_engineer:
                df_featured = self.feature_engineer.create_all_features(df)
                
                if self.feature_columns:
                    available_cols = [col for col in self.feature_columns 
                                    if col in df_featured.columns]
                    df_featured = df_featured[available_cols]
                    
                    for col in self.feature_columns:
                        if col not in df_featured.columns:
                            df_featured[col] = 0
                
                df_scaled = self.feature_engineer.scale_features(df_featured, fit=False)
                
                return df_scaled
            else:
                return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def predict(self, transaction_dict):
        """Make prediction"""
        try:
            X = self.prepare_features(transaction_dict)
            
            if X is None or self.model is None:
                return None
            
            fraud_prob = float(self.model.predict_proba(X)[0][1])
            
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                if len(importance) == len(X.columns):
                    top_indices = np.argsort(importance)[-5:][::-1]
                    feature_importance = [
                        {'feature': X.columns[i], 'importance': float(importance[i])} 
                        for i in top_indices
                    ]
            
            thresholds = self.config.get('thresholds', {})
            approve_threshold = thresholds.get('approve', 0.30)
            block_threshold = thresholds.get('optimal', 0.75)
            
            if fraud_prob < approve_threshold:
                risk_level = "LOW"
                decision = "APPROVE"
                color = "green"
            elif fraud_prob < block_threshold:
                risk_level = "MEDIUM"
                decision = "REVIEW"
                color = "orange"
            else:
                risk_level = "HIGH"
                decision = "BLOCK"
                color = "red"
            
            return {
                'fraud_probability': round(fraud_prob, 4),
                'risk_level': risk_level,
                'decision': decision,
                'color': color,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def get_model_info(self):
        """Get model information"""
        info = {
            'model_type': self.model.__class__.__name__ if self.model else 'Not loaded',
            'features_used': len(self.feature_columns) if self.feature_columns else 0,
            'config': self.config
        }
        
        if self.model_info:
            info.update(self.model_info)
        
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
        
        return info