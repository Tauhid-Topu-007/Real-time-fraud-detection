import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
import json
import warnings
warnings.filterwarnings('ignore')

class FraudPredictor:
    """Fraud prediction service for Colab-trained model"""
    
    def __init__(self, model_path="models"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_columns = None
        self.model_info = None
        self.optimal_threshold = 0.5
        
        # Load everything
        self.load_models()
    
    def load_models(self):
        """Load model and artifacts"""
        try:
            # Load model
            model_file = self.model_path / "xgboost_model.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
                logger.info("✅ Model loaded successfully")
            else:
                logger.error(f"❌ Model not found at {model_file}")
                return False
            
            # Load feature columns
            feature_file = self.model_path / "feature_engineer.pkl"
            if feature_file.exists():
                self.feature_columns = joblib.load(feature_file)
                logger.info(f"✅ Feature columns loaded: {len(self.feature_columns)} features")
            else:
                logger.warning("⚠️ Feature columns not found")
                self.feature_columns = []
            
            # Load model info
            info_file = self.model_path / "model_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    self.model_info = json.load(f)
                self.optimal_threshold = self.model_info.get('optimal_threshold', 0.5)
                logger.info(f"✅ Model info loaded, threshold: {self.optimal_threshold}")
            else:
                logger.warning("⚠️ Model info not found")
                self.model_info = {}
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def prepare_features(self, transaction):
        """Prepare features for prediction"""
        try:
            df = pd.DataFrame([transaction])
            
            # Create all features (same as Colab)
            
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
            
            # Customer features (for new customers, use 0)
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
            
            # Fill any missing columns with 0
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in df.columns:
                        df[col] = 0
                
                # Keep only the features we need
                df = df[self.feature_columns]
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}")
            return None
    
    def predict(self, transaction):
        """Make prediction for a single transaction"""
        try:
            if self.model is None:
                logger.error("❌ Model not loaded")
                return None
            
            # Prepare features
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
            logger.error(f"❌ Prediction error: {e}")
            return None
    
    def predict_batch(self, transactions):
        """Make predictions for multiple transactions"""
        results = []
        for transaction in transactions:
            result = self.predict(transaction)
            if result:
                results.append(result)
        return results
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': self.model_info.get('model_type', 'xgboost') if self.model_info else 'xgboost',
            'features': len(self.feature_columns) if self.feature_columns else 0,
            'threshold': self.optimal_threshold,
            'metrics': self.model_info.get('metrics', {}) if self.model_info else {}
        }


# Test the predictor
if __name__ == "__main__":
    print("Testing FraudPredictor...")
    predictor = FraudPredictor()
    
    if predictor.model is None:
        print("❌ Model not loaded. Please train the model first.")
        print("Run: python scripts/train_for_streamlit.py")
    else:
        # Test transaction
        test_txn = {
            'step': 5,
            'type': 'TRANSFER',
            'amount': 2450.0,
            'nameOrig': 'C1234567890',
            'oldbalanceOrg': 5000.0,
            'newbalanceOrig': 2550.0,
            'nameDest': 'M9876543210',
            'oldbalanceDest': 1000.0,
            'newbalanceDest': 3450.0
        }
        
        result = predictor.predict(test_txn)
        
        print("\n" + "="*50)
        print("🔮 Prediction Result")
        print("="*50)
        if result:
            print(f"Fraud Probability: {result['fraud_probability']*100:.1f}%")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Decision: {result['decision']}")
        else:
            print("❌ Prediction failed")
        print("="*50)