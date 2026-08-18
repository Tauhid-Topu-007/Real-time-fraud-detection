import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import warnings
import sys
warnings.filterwarnings('ignore')

class FraudPredictor:
    """Fraud prediction service - MLflow free version"""
    
    def __init__(self, model_path=None):
        # Try multiple paths to find models
        if model_path is None:
            possible_paths = [
                Path("models"),                                    # Project root
                Path("../models"),                                 # One level up
                Path("../../models"),                              # Two levels up
                Path("/mount/src/real-time-fraud-detection/models"),  # Streamlit Cloud
                Path(__file__).parent.parent.parent / "models",    # From src/inference
                Path(__file__).parent.parent / "models",           # From src/inference
            ]
            
            self.model_path = None
            for path in possible_paths:
                if path.exists() and (path / "xgboost_model.pkl").exists():
                    self.model_path = path
                    print(f"✅ Found models at: {self.model_path}")
                    break
            
            if self.model_path is None:
                self.model_path = Path("models")
                print(f"⚠️ Using default models path: {self.model_path}")
        else:
            self.model_path = Path(model_path)
        
        self.model = None
        self.feature_columns = []
        self.optimal_threshold = 0.5
        self.model_info = {}
        self.feature_engineer = None
        
        # Load everything
        self.load_models()
    
    def load_models(self):
        """Load model and artifacts"""
        try:
            # 1. Check if directory exists
            if not self.model_path.exists():
                print(f"❌ Models directory not found: {self.model_path}")
                print(f"   Current directory: {Path.cwd()}")
                print(f"   Looking for models in: {list(Path.cwd().glob('**/models'))}")
                return False
            
            # 2. Load Model
            model_file = self.model_path / "xgboost_model.pkl"
            if model_file.exists():
                try:
                    self.model = joblib.load(model_file)
                    print(f"✅ Model loaded successfully from {model_file}")
                except Exception as e:
                    print(f"❌ Error loading model: {e}")
                    self.model = None
                    return False
            else:
                print(f"❌ Model not found at {model_file}")
                print(f"   Files in {self.model_path}: {list(self.model_path.glob('*'))}")
                return False
            
            # 3. Load Feature Engineer
            fe_file = self.model_path / "feature_engineer.pkl"
            if fe_file.exists():
                try:
                    self.feature_engineer = joblib.load(fe_file)
                    if hasattr(self.feature_engineer, 'feature_columns'):
                        self.feature_columns = self.feature_engineer.feature_columns
                        print(f"✅ Features loaded: {len(self.feature_columns)}")
                except Exception as e:
                    print(f"⚠️ Error loading feature_engineer: {e}")
            else:
                print(f"⚠️ feature_engineer.pkl not found at {fe_file}")
            
            # 4. Load Model Info
            info_file = self.model_path / "model_info.json"
            if info_file.exists():
                try:
                    with open(info_file, 'r') as f:
                        self.model_info = json.load(f)
                    self.optimal_threshold = self.model_info.get('optimal_threshold', 0.5)
                    print(f"✅ Model info loaded, threshold: {self.optimal_threshold}")
                except Exception as e:
                    print(f"⚠️ Error loading model_info: {e}")
                    self.optimal_threshold = 0.5
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_features(self, transaction):
        """Prepare features for prediction"""
        try:
            df = pd.DataFrame([transaction])
            
            # If we have feature_engineer, use it
            if self.feature_engineer is not None:
                try:
                    df_featured = self.feature_engineer.create_all_features(df)
                    
                    if self.feature_columns:
                        for col in self.feature_columns:
                            if col not in df_featured.columns:
                                df_featured[col] = 0
                        df_featured = df_featured[self.feature_columns]
                    
                    return df_featured
                except Exception as e:
                    print(f"⚠️ Feature engineer failed: {e}")
            
            # Fallback: Manual feature creation
            return self._manual_features(df)
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None
    
    def _manual_features(self, df):
        """Manual feature creation as fallback"""
        try:
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
            print(f"❌ Error in manual features: {e}")
            return None
    
    def predict(self, transaction):
        """Make prediction"""
        try:
            if self.model is None:
                print("❌ Model not loaded")
                return None
            
            X = self.prepare_features(transaction)
            if X is None or X.empty:
                print("❌ Feature preparation failed")
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
            'model_type': self.model_info.get('model_type', 'xgboost'),
            'features': len(self.feature_columns),
            'threshold': self.optimal_threshold,
            'metrics': self.model_info.get('metrics', {})
        }


# Test the predictor
if __name__ == "__main__":
    print("="*50)
    print("Testing FraudPredictor...")
    print("="*50)
    
    predictor = FraudPredictor()
    
    if predictor.model is None:
        print("❌ Model not loaded. Please check the models directory.")
        print(f"   Looking in: {predictor.model_path}")
        print(f"   Files there: {list(predictor.model_path.glob('*')) if predictor.model_path.exists() else 'Directory not found'}")
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