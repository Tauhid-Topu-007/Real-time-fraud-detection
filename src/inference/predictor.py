import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class FraudPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.threshold = 0.5
        self._load_models()
    
    def _load_models(self):
        """Load model files"""
        try:
            models_path = Path("models")
            
            # Load model
            model_file = models_path / "xgboost_model.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
                print(f"✅ Model loaded")
            
            # Load feature columns from training
            feat_file = models_path / "feature_columns.pkl"
            if feat_file.exists():
                self.feature_columns = joblib.load(feat_file)
                print(f"✅ Features loaded: {len(self.feature_columns)}")
            else:
                # If feature_columns.pkl doesn't exist, use the model's feature names
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_columns = list(self.model.feature_names_in_)
                    print(f"✅ Features from model: {len(self.feature_columns)}")
            
            # Load model info
            info_file = models_path / "model_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                self.threshold = info.get('optimal_threshold', 0.5)
                print(f"✅ Threshold: {self.threshold}")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
    
    def prepare_features(self, transaction):
        """
        Prepare features EXACTLY as in training
        """
        try:
            # Create features dictionary - ONLY features used in training
            features = {}
            
            # ============================================
            # 1. Step (original)
            # ============================================
            features['step'] = int(transaction.get('step', 0))
            
            # ============================================
            # 2. Amount (original)
            # ============================================
            amount = float(transaction.get('amount', 0))
            features['amount'] = amount
            
            # ============================================
            # 3. Balance (original)
            # ============================================
            features['oldbalanceOrg'] = float(transaction.get('oldbalanceOrg', 0))
            features['newbalanceOrig'] = float(transaction.get('newbalanceOrig', 0))
            features['oldbalanceDest'] = float(transaction.get('oldbalanceDest', 0))
            features['newbalanceDest'] = float(transaction.get('newbalanceDest', 0))
            
            # ============================================
            # 4. Transaction Type Features (as in training)
            # ============================================
            txn_type = transaction.get('type', '')
            
            # One-hot encoding (match training)
            features['type_PAYMENT'] = 1 if txn_type == 'PAYMENT' else 0
            features['type_TRANSFER'] = 1 if txn_type == 'TRANSFER' else 0
            features['type_CASH_OUT'] = 1 if txn_type == 'CASH_OUT' else 0
            features['type_DEBIT'] = 1 if txn_type == 'DEBIT' else 0
            features['type_CASH_IN'] = 1 if txn_type == 'CASH_IN' else 0
            
            # Type risk score
            type_risk = {'TRANSFER': 0.8, 'CASH_OUT': 0.7, 'PAYMENT': 0.2, 
                        'CASH_IN': 0.1, 'DEBIT': 0.3}
            features['type_risk_score'] = type_risk.get(txn_type, 0.5)
            
            # ============================================
            # 5. Balance Features (as in training)
            # ============================================
            features['balance_change_orig'] = features['oldbalanceOrg'] - features['newbalanceOrig']
            features['balance_error_orig'] = features['oldbalanceOrg'] - features['newbalanceOrig'] - amount
            features['amount_vs_balance_orig'] = amount / (features['oldbalanceOrg'] + 1)
            
            # ============================================
            # 6. Customer Features (default as in training)
            # ============================================
            features['orig_txn_count'] = 0
            features['orig_avg_amount'] = 0
            features['orig_fraud_ratio'] = 0
            features['dest_txn_count'] = 0
            features['dest_avg_amount'] = 0
            
            # ============================================
            # 7. Amount Features (as in training)
            # ============================================
            features['amount_log'] = np.log1p(amount)
            features['amount_z_score'] = 0  # Default for single prediction
            
            # ============================================
            # 8. Time Features (as in training)
            # ============================================
            step = features['step']
            features['hour'] = step % 24
            features['is_night'] = 1 if (step % 24 >= 22 or step % 24 <= 5) else 0
            
            # ============================================
            # 9. Create DataFrame with EXACT training columns
            # ============================================
            X = pd.DataFrame([features])
            
            # Add missing columns with 0 (match training)
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in X.columns:
                        X[col] = 0
                
                # Keep only training columns and maintain order
                X = X[self.feature_columns]
            
            # Convert to float
            X = X.astype(float)
            
            return X
            
        except Exception as e:
            print(f"❌ Feature preparation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, transaction):
        """Make prediction"""
        try:
            if self.model is None:
                print("❌ Model not loaded")
                return None
            
            # Remove transaction_id if present
            if isinstance(transaction, dict) and 'transaction_id' in transaction:
                transaction = transaction.copy()
                del transaction['transaction_id']
            
            X = self.prepare_features(transaction)
            if X is None or X.empty:
                print("❌ Feature preparation failed")
                return None
            
            # Predict
            prob = float(self.model.predict_proba(X)[0][1])
            
            # Decision
            if prob < 0.3:
                return {
                    'fraud_probability': round(prob, 4),
                    'risk_level': 'LOW',
                    'decision': 'APPROVE'
                }
            elif prob < self.threshold:
                return {
                    'fraud_probability': round(prob, 4),
                    'risk_level': 'MEDIUM',
                    'decision': 'REVIEW'
                }
            else:
                return {
                    'fraud_probability': round(prob, 4),
                    'risk_level': 'HIGH',
                    'decision': 'BLOCK'
                }
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_info(self):
        """Get model information"""
        return {
            'features': len(self.feature_columns),
            'threshold': self.threshold,
            'model_type': 'xgboost'
        }


# Test
if __name__ == "__main__":
    print("="*50)
    print("Testing FraudPredictor...")
    print("="*50)
    
    predictor = FraudPredictor()
    
    if predictor.model is None:
        print("❌ Model not loaded")
    else:
        print(f"✅ Model loaded with {len(predictor.feature_columns)} features")
        print(f"📊 Feature columns: {predictor.feature_columns}")
        
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
        print(f"\n🔮 Result: {result}")