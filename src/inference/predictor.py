import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class FraudPredictor:
    """Fraud prediction service - MLflow free version"""
    
    def __init__(self, model_path="models"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_columns = []
        self.optimal_threshold = 0.5
        self.model_info = {}
        
        # Load everything
        self.load_models()
    
    def load_models(self):
        """Load model and artifacts"""
        try:
            # Load model
            model_file = self.model_path / "xgboost_model.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
                print("✅ Model loaded successfully")
            else:
                print(f"❌ Model not found at {model_file}")
                return False
            
            # Load feature columns
            feature_file = self.model_path / "feature_engineer.pkl"
            if feature_file.exists():
                self.feature_columns = joblib.load(feature_file)
                print(f"✅ Features loaded: {len(self.feature_columns)}")
            
            # Load model info
            info_file = self.model_path / "model_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    self.model_info = json.load(f)
                self.optimal_threshold = self.model_info.get('optimal_threshold', 0.5)
                print(f"✅ Model info loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def prepare_features(self, transaction):
        """Prepare features for prediction"""
        try:
            df = pd.DataFrame([transaction])
            
            # Transaction type features
            type_dummies = pd.get_dummies(df['type'], prefix='type')
            df = pd.concat([df, type_dummies], axis=1)
            
            type_risk = {'TRANSFER': 0.8, 'CASH_OUT': 0.7, 'PAYMENT': 0.2, 
                        'CASH_IN': 0.1, 'DEBIT': 0.3}
            df['type_risk_score'] = df['type'].map(type_risk).fillna(0.5)
            
            # Balance features
            df['balance_change_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
            df['balance_error_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
            df['amount_vs_balance_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
            
            # Customer features (default values for new customers)
            df['orig_txn_count'] = 0
            df['orig_avg_amount'] = 0
            df['orig_fraud_ratio'] = 0
            df['dest_txn_count'] = 0
            df['dest_avg_amount'] = 0
            
            # Amount features
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_z_score'] = (df['amount'] - df['amount'].mean()) / (df['amount'].std() + 1e-6)
            
            # Time features
            df['hour'] = df['step'] % 24
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
            
            # Fill missing columns
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in df.columns:
                        df[col] = 0
                df = df[self.feature_columns]
            
            return df
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None
    
    def predict(self, transaction):
        """Make prediction"""
        try:
            if self.model is None:
                return None
            
            X = self.prepare_features(transaction)
            if X is None:
                return None
            
            # Get probability
            fraud_prob = float(self.model.predict_proba(X)[0][1])
            
            # Determine risk and decision
            if fraud_prob < 0.30:
                risk_level = "LOW"
                decision = "APPROVE"
                color = "green"
            elif fraud_prob < self.optimal_threshold:
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
                'color': color
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def predict_batch(self, transactions):
        """Make multiple predictions"""
        results = []
        for transaction in transactions:
            result = self.predict(transaction)
            if result:
                results.append(result)
        return results
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': self.model_info.get('model_type', 'xgboost'),
            'features': len(self.feature_columns),
            'threshold': self.optimal_threshold,
            'metrics': self.model_info.get('metrics', {})
        }