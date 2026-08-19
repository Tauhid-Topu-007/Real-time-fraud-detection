# ============================================
# src/inference/predictor.py
# HYBRID SYSTEM - ML + RULE-BASED
# ============================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering class - creates features for ML model"""
    
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.amount_mean = 181349.35
        self.amount_std = 627939.33
    
    def create_all_features(self, df):
        """Create all features matching training data"""
        try:
            df_featured = df.copy()
            
            # ============================================
            # 1. Ensure all required columns exist
            # ============================================
            required_cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                           'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
            
            for col in required_cols:
                if col not in df_featured.columns:
                    if col == 'type':
                        df_featured[col] = 'TRANSFER'
                    else:
                        df_featured[col] = 0
            
            # ============================================
            # 2. Convert to numeric
            # ============================================
            numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                           'oldbalanceDest', 'newbalanceDest']
            for col in numeric_cols:
                df_featured[col] = pd.to_numeric(df_featured[col], errors='coerce').fillna(0)
            
            amount = float(df_featured['amount'].iloc[0])
            txn_type = str(df_featured['type'].iloc[0])
            
            # ============================================
            # 3. Transaction Type Features (One-Hot Encoding)
            # ============================================
            df_featured['type_PAYMENT'] = 1 if txn_type == 'PAYMENT' else 0
            df_featured['type_TRANSFER'] = 1 if txn_type == 'TRANSFER' else 0
            df_featured['type_CASH_OUT'] = 1 if txn_type == 'CASH_OUT' else 0
            df_featured['type_DEBIT'] = 1 if txn_type == 'DEBIT' else 0
            df_featured['type_CASH_IN'] = 1 if txn_type == 'CASH_IN' else 0
            
            # Type risk score
            type_risk = {'TRANSFER': 0.8, 'CASH_OUT': 0.7, 'PAYMENT': 0.2, 
                        'CASH_IN': 0.1, 'DEBIT': 0.3}
            df_featured['type_risk_score'] = type_risk.get(txn_type, 0.5)
            
            # ============================================
            # 4. Balance Features
            # ============================================
            oldbalanceOrg = float(df_featured['oldbalanceOrg'].iloc[0])
            newbalanceOrig = float(df_featured['newbalanceOrig'].iloc[0])
            oldbalanceDest = float(df_featured['oldbalanceDest'].iloc[0])
            newbalanceDest = float(df_featured['newbalanceDest'].iloc[0])
            
            df_featured['balance_change_orig'] = oldbalanceOrg - newbalanceOrig
            df_featured['balance_error_orig'] = oldbalanceOrg - newbalanceOrig - amount
            df_featured['amount_vs_balance_orig'] = amount / (oldbalanceOrg + 1)
            
            # ============================================
            # 5. Amount Features
            # ============================================
            df_featured['amount_log'] = np.log1p(amount)
            df_featured['amount_z_score'] = (amount - self.amount_mean) / (self.amount_std + 1e-6)
            
            # ============================================
            # 6. Time Features
            # ============================================
            step = int(df_featured['step'].iloc[0])
            df_featured['hour'] = step % 24
            df_featured['is_night'] = 1 if (step % 24 >= 22 or step % 24 <= 5) else 0
            
            # ============================================
            # 7. Customer Features (default values)
            # ============================================
            df_featured['orig_txn_count'] = 10
            df_featured['orig_avg_amount'] = amount * 0.8
            df_featured['orig_fraud_ratio'] = 0.0
            df_featured['dest_txn_count'] = 10
            df_featured['dest_avg_amount'] = amount * 0.7
            
            # ============================================
            # 8. Interaction Features
            # ============================================
            df_featured['amount_type_risk'] = amount * df_featured['type_risk_score'].iloc[0]
            
            # ============================================
            # 9. Add missing columns with 0
            # ============================================
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in df_featured.columns:
                        df_featured[col] = 0
                df_featured = df_featured[self.feature_columns]
            
            # ============================================
            # 10. Convert all to float
            # ============================================
            for col in df_featured.columns:
                df_featured[col] = pd.to_numeric(df_featured[col], errors='coerce').fillna(0)
            
            return df_featured.astype(float)
            
        except Exception as e:
            print(f"❌ Feature creation error: {e}")
            import traceback
            traceback.print_exc()
            return None


class FraudPredictor:
    """Hybrid Fraud Predictor - ML + Rule-Based"""
    
    def __init__(self, model_path="models"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_columns = []
        self.threshold = 0.50
        self.feature_engineer = None
        self.amount_mean = 181349.35
        self.amount_std = 627939.33
        self.model_info = {}
        self._load_models()
    
    def _load_models(self):
        """Load all model files"""
        try:
            # ============================================
            # 1. Load Model
            # ============================================
            model_file = self.model_path / "xgboost_model.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
                print(f"✅ Model loaded: {model_file}")
            else:
                print(f"⚠️ Model not found at {model_file}")
                print("   Continuing with rule-based only")
            
            # ============================================
            # 2. Load Feature Columns
            # ============================================
            fc_file = self.model_path / "feature_columns.pkl"
            if fc_file.exists():
                self.feature_columns = joblib.load(fc_file)
                print(f"✅ Features loaded: {len(self.feature_columns)}")
            else:
                print(f"⚠️ feature_columns.pkl not found")
                print("   Using default features")
                self.feature_columns = [
                    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest', 'type_PAYMENT', 
                    'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 
                    'type_CASH_IN', 'type_risk_score', 'balance_change_orig',
                    'balance_error_orig', 'amount_vs_balance_orig', 
                    'amount_log', 'amount_z_score', 'hour', 'is_night',
                    'orig_txn_count', 'orig_avg_amount', 'orig_fraud_ratio',
                    'dest_txn_count', 'dest_avg_amount', 'amount_type_risk'
                ]
            
            # ============================================
            # 3. Load Model Info
            # ============================================
            info_file = self.model_path / "model_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    self.model_info = json.load(f)
                self.threshold = self.model_info.get('optimal_threshold', 0.50)
                print(f"✅ Threshold: {self.threshold}")
                
                if 'amount_mean' in self.model_info:
                    self.amount_mean = self.model_info['amount_mean']
                if 'amount_std' in self.model_info:
                    self.amount_std = self.model_info['amount_std']
            else:
                print(f"⚠️ model_info.json not found")
                print(f"   Using default threshold: {self.threshold}")
            
            # ============================================
            # 4. Create Feature Engineer
            # ============================================
            self.feature_engineer = FeatureEngineer(self.feature_columns)
            self.feature_engineer.amount_mean = self.amount_mean
            self.feature_engineer.amount_std = self.amount_std
            
            print(f"✅ Feature engineer created")
            print(f"   Amount Mean: {self.amount_mean:.2f}")
            print(f"   Amount Std: {self.amount_std:.2f}")
            print("="*50)
            print("✅ All models loaded successfully!")
            print("="*50)
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
    
    def create_features(self, df):
        """Create features using feature engineer"""
        if self.feature_engineer:
            return self.feature_engineer.create_all_features(df)
        return None
    
    def rule_based_detect(self, transaction):
        """
        Rule-based fraud detection (fallback when ML fails)
        Returns: dict with prediction result
        """
        score = 0
        reasons = []
        
        amount = float(transaction.get('amount', 0))
        oldbalanceOrg = float(transaction.get('oldbalanceOrg', 0))
        newbalanceOrig = float(transaction.get('newbalanceOrig', 0))
        oldbalanceDest = float(transaction.get('oldbalanceDest', 0))
        newbalanceDest = float(transaction.get('newbalanceDest', 0))
        txn_type = str(transaction.get('type', ''))
        
        # ============================================
        # RULE 1: Very high amount
        # ============================================
        if amount > 100000:
            score += 30
            reasons.append(f"Very high amount (${amount:,.2f})")
        
        # ============================================
        # RULE 2: Amount > 80% of balance
        # ============================================
        if oldbalanceOrg > 0:
            ratio = amount / oldbalanceOrg
            if ratio > 0.8:
                score += 25
                reasons.append(f"Amount is {ratio:.1%} of balance")
        
        # ============================================
        # RULE 3: Balance nearly emptied
        # ============================================
        if oldbalanceOrg > 0:
            emptied_ratio = (oldbalanceOrg - newbalanceOrig) / oldbalanceOrg
            if emptied_ratio > 0.9:
                score += 20
                reasons.append(f"Balance {emptied_ratio:.1%} emptied")
        
        # ============================================
        # RULE 4: Large transfer
        # ============================================
        if txn_type == 'TRANSFER' and amount > 50000:
            score += 15
            reasons.append(f"Large transfer (${amount:,.2f})")
        
        # ============================================
        # RULE 5: Destination balance increased significantly
        # ============================================
        if oldbalanceDest >= 0 and newbalanceDest > oldbalanceDest:
            dest_increase = newbalanceDest - oldbalanceDest
            if dest_increase > amount * 0.9 and amount > 10000:
                score += 10
                reasons.append(f"Destination balance increased by ${dest_increase:,.2f}")
        
        # ============================================
        # RULE 6: Large cash out
        # ============================================
        if txn_type == 'CASH_OUT' and amount > 50000:
            score += 15
            reasons.append(f"Large cash out (${amount:,.2f})")
        
        # ============================================
        # RULE 7: Amount equals full balance
        # ============================================
        if oldbalanceOrg > 0 and abs(amount - oldbalanceOrg) < 10:
            score += 20
            reasons.append("Amount equals full balance")
        
        # ============================================
        # RULE 8: High amount + TRANSFER type
        # ============================================
        if txn_type == 'TRANSFER' and amount > 100000:
            score += 10
            reasons.append("High amount transfer")
        
        # ============================================
        # Determine risk level
        # ============================================
        probability = min(score / 100, 0.99)
        
        if score >= 60:
            risk_level = "HIGH"
            decision = "BLOCK"
        elif score >= 30:
            risk_level = "MEDIUM"
            decision = "REVIEW"
        else:
            risk_level = "LOW"
            decision = "APPROVE"
        
        return {
            'fraud_probability': round(probability, 4),
            'risk_level': risk_level,
            'decision': decision,
            'score': score,
            'reasons': reasons,
            'method': 'rule_based'
        }
    
    def _ml_predict(self, transaction):
        """ML model prediction"""
        try:
            if self.model is None:
                return None
            
            df = pd.DataFrame([transaction])
            X = self.create_features(df)
            if X is None:
                return None
            
            prob = float(self.model.predict_proba(X)[0][1])
            
            if prob < 0.30:
                return {
                    'fraud_probability': round(prob, 4),
                    'risk_level': 'LOW',
                    'decision': 'APPROVE',
                    'method': 'ml'
                }
            elif prob < self.threshold:
                return {
                    'fraud_probability': round(prob, 4),
                    'risk_level': 'MEDIUM',
                    'decision': 'REVIEW',
                    'method': 'ml'
                }
            else:
                return {
                    'fraud_probability': round(prob, 4),
                    'risk_level': 'HIGH',
                    'decision': 'BLOCK',
                    'method': 'ml'
                }
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            return None
    
    def predict(self, transaction):
        """
        Make prediction - Hybrid approach:
        1. Try ML model first
        2. If ML gives LOW risk, check rules
        3. If rules detect HIGH risk, override ML
        4. If ML fails, use rule-based only
        """
        try:
            # Remove transaction_id if present
            if 'transaction_id' in transaction:
                transaction = transaction.copy()
                del transaction['transaction_id']
            
            # ============================================
            # Step 1: Try ML model
            # ============================================
            ml_result = self._ml_predict(transaction)
            
            # ============================================
            # Step 2: If ML worked, check if rules should override
            # ============================================
            if ml_result:
                # If ML says LOW risk, verify with rules
                if ml_result['risk_level'] == 'LOW':
                    rule_result = self.rule_based_detect(transaction)
                    
                    # If rules say HIGH or MEDIUM, override ML
                    if rule_result['risk_level'] in ['HIGH', 'MEDIUM']:
                        print(f"⚡ Rule-based override: ML said LOW but rules say {rule_result['risk_level']}")
                        return rule_result
                
                # ML result is acceptable
                return ml_result
            
            # ============================================
            # Step 3: If ML failed, use rule-based only
            # ============================================
            print("⚠️ ML failed, using rule-based only")
            return self.rule_based_detect(transaction)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.rule_based_detect(transaction)
    
    def predict_batch(self, transactions):
        """Make multiple predictions - each transaction processed individually"""
        results = []
        
        for i, transaction in enumerate(transactions):
            try:
                if 'transaction_id' not in transaction:
                    transaction['transaction_id'] = f"BATCH_{i}"
                
                result = self.predict(transaction)
                
                if result:
                    result['transaction_id'] = transaction['transaction_id']
                    results.append(result)
                else:
                    rule_result = self.rule_based_detect(transaction)
                    rule_result['transaction_id'] = transaction['transaction_id']
                    results.append(rule_result)
                    
            except Exception as e:
                print(f"❌ Error processing transaction {i}: {e}")
                results.append({
                    'fraud_probability': 0.0,
                    'risk_level': 'LOW',
                    'decision': 'APPROVE',
                    'transaction_id': transaction.get('transaction_id', f'BATCH_{i}'),
                    'method': 'error',
                    'error': str(e)
                })
        
        return results
    
    def get_model_info(self):
        """Get model information"""
        return {
            'features': len(self.feature_columns),
            'threshold': self.threshold,
            'model_type': 'xgboost + rule-based (hybrid)',
            'amount_mean': self.amount_mean,
            'amount_std': self.amount_std,
            'metrics': self.model_info.get('metrics', {}),
            'model_loaded': self.model is not None
        }


# ============================================
# Test the predictor
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("🔮 Testing Hybrid Predictor")
    print("="*60)
    
    predictor = FraudPredictor()
    
    # Test 1: Fraudulent Transaction
    fraud_txn = {
        'step': 1,
        'type': 'TRANSFER',
        'amount': 999999.0,
        'nameOrig': 'C1111111111',
        'oldbalanceOrg': 1000000.0,
        'newbalanceOrig': 1.0,
        'nameDest': 'M9999999999',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 999999.0
    }
    
    print(f"\n🚨 Testing Fraudulent Transaction:")
    print(f"   Amount: ${fraud_txn['amount']:,.2f}")
    
    result = predictor.predict(fraud_txn)
    print(f"\n📊 Result:")
    print(f"   Probability: {result['fraud_probability']*100:.2f}%")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Decision: {result['decision']}")
    if 'reasons' in result:
        print(f"   Reasons: {result['reasons']}")
    
    # Test 2: Normal Transaction
    normal_txn = {
        'step': 10,
        'type': 'PAYMENT',
        'amount': 50.0,
        'nameOrig': 'C4444444444',
        'oldbalanceOrg': 1000.0,
        'newbalanceOrig': 950.0,
        'nameDest': 'M6666666666',
        'oldbalanceDest': 500.0,
        'newbalanceDest': 550.0
    }
    
    print(f"\n✅ Testing Normal Transaction:")
    print(f"   Amount: ${normal_txn['amount']:,.2f}")
    
    result = predictor.predict(normal_txn)
    print(f"\n📊 Result:")
    print(f"   Probability: {result['fraud_probability']*100:.2f}%")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Decision: {result['decision']}")
    
    print("\n" + "="*60)
    print("✅ Testing Complete!")
    print("="*60)