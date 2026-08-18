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
            
            model_file = models_path / "xgboost_model.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
                print(f"✅ Model loaded")
            
            fe_file = models_path / "feature_engineer.pkl"
            if fe_file.exists():
                fe = joblib.load(fe_file)
                if hasattr(fe, 'feature_columns'):
                    self.feature_columns = fe.feature_columns
                    print(f"✅ Features: {len(self.feature_columns)}")
            
            info_file = models_path / "model_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                self.threshold = info.get('optimal_threshold', 0.5)
                print(f"✅ Threshold: {self.threshold}")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def prepare_features(self, transaction):
        """
        Prepare features for XGBoost prediction.
        XGBoost only accepts: int, float, bool, or category dtypes.
        """
        try:
            # Create a clean dictionary with only numeric features
            features = {}
            
            # ============================================
            # 1. Amount Features (numeric)
            # ============================================
            amount = float(transaction.get('amount', 0))
            features['amount'] = amount
            features['amount_log'] = np.log1p(amount)
            
            # ============================================
            # 2. Balance Features (numeric)
            # ============================================
            oldbalanceOrg = float(transaction.get('oldbalanceOrg', 0))
            newbalanceOrig = float(transaction.get('newbalanceOrig', 0))
            oldbalanceDest = float(transaction.get('oldbalanceDest', 0))
            newbalanceDest = float(transaction.get('newbalanceDest', 0))
            
            features['balance_change_orig'] = oldbalanceOrg - newbalanceOrig
            features['balance_change_dest'] = newbalanceDest - oldbalanceDest
            features['balance_error_orig'] = oldbalanceOrg - newbalanceOrig - amount
            features['amount_vs_balance_orig'] = amount / (oldbalanceOrg + 1)
            features['amount_vs_balance_dest'] = amount / (oldbalanceDest + 1)
            
            # ============================================
            # 3. Transaction Type Features (numeric)
            # ============================================
            txn_type = transaction.get('type', '')
            
            # One-hot encoding for transaction type
            type_dummies = {
                'type_PAYMENT': 1 if txn_type == 'PAYMENT' else 0,
                'type_TRANSFER': 1 if txn_type == 'TRANSFER' else 0,
                'type_CASH_OUT': 1 if txn_type == 'CASH_OUT' else 0,
                'type_DEBIT': 1 if txn_type == 'DEBIT' else 0,
                'type_CASH_IN': 1 if txn_type == 'CASH_IN' else 0,
            }
            features.update(type_dummies)
            
            # Type risk score
            type_risk = {'TRANSFER': 0.8, 'CASH_OUT': 0.7, 'PAYMENT': 0.2, 
                        'CASH_IN': 0.1, 'DEBIT': 0.3}
            features['type_risk_score'] = type_risk.get(txn_type, 0.5)
            
            # ============================================
            # 4. Time Features (numeric)
            # ============================================
            step = int(transaction.get('step', 0))
            features['step'] = step
            features['hour'] = step % 24
            features['day'] = (step // 24) % 7
            features['is_weekend'] = 1 if (step // 24) % 7 >= 5 else 0
            features['is_night'] = 1 if (step % 24 >= 22 or step % 24 <= 5) else 0
            
            # ============================================
            # 5. Customer Features (numeric - default values)
            # ============================================
            features['orig_txn_count'] = 0
            features['orig_avg_amount'] = 0
            features['orig_fraud_ratio'] = 0
            features['dest_txn_count'] = 0
            features['dest_avg_amount'] = 0
            features['is_new_origin'] = 1
            
            # ============================================
            # 6. Interaction Features (numeric)
            # ============================================
            features['amount_type_risk'] = amount * features['type_risk_score']
            features['balance_change_amount_interaction'] = features['balance_change_orig'] * amount
            
            # ============================================
            # 7. Create DataFrame
            # ============================================
            X = pd.DataFrame([features])
            
            # Add any missing feature columns with 0
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in X.columns:
                        X[col] = 0
                X = X[self.feature_columns]
            
            # ============================================
            # 8. Ensure all columns are numeric (float)
            # ============================================
            X = X.astype(float)
            
            return X
            
        except Exception as e:
            print(f"❌ Feature preparation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, transaction):
        """Make prediction for a single transaction"""
        try:
            if self.model is None:
                print("❌ Model not loaded")
                return None
            
            # Remove transaction_id if present (it's string)
            if 'transaction_id' in transaction:
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
            'features': len(self.feature_columns),
            'threshold': self.threshold,
            'model_type': 'xgboost',
            'metrics': {}
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