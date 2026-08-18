import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

class BigDataFeatureEngineer:
    """Advanced feature engineering for large datasets"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.column_types = {}
        logger.info("BigDataFeatureEngineer initialized")
    
    def create_transaction_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on transaction type"""
        logger.info("Creating transaction type features")
        
        # One-hot encoding for transaction type
        type_dummies = pd.get_dummies(df['type'], prefix='type')
        df = pd.concat([df, type_dummies], axis=1)
        
        # Transaction type risk score - FIXED: Convert to float first
        type_risk = {
            'TRANSFER': 0.8,
            'CASH_OUT': 0.7,
            'PAYMENT': 0.2,
            'CASH_IN': 0.1,
            'DEBIT': 0.3
        }
        
        # Map and fill with 0.5, ensuring float type
        df['type_risk_score'] = df['type'].map(type_risk).astype(float).fillna(0.5)
        df['is_transfer_or_cashout'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
        df['is_cash_in'] = (df['type'] == 'CASH_IN').astype(int)
        
        return df
    
    def create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create balance-related features"""
        logger.info("Creating balance features")
        
        # Convert to float to avoid categorical issues
        df['balance_change_orig'] = (df['oldbalanceOrg'] - df['newbalanceOrig']).astype(float)
        df['balance_change_ratio_orig'] = (df['balance_change_orig'] / (df['oldbalanceOrg'] + 1)).astype(float)
        df['balance_change_dest'] = (df['newbalanceDest'] - df['oldbalanceDest']).astype(float)
        df['balance_change_ratio_dest'] = (df['balance_change_dest'] / (df['oldbalanceDest'] + 1)).astype(float)
        df['balance_error_orig'] = (df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']).astype(float)
        df['balance_error_dest'] = (df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']).astype(float)
        df['has_balance_error'] = (abs(df['balance_error_orig']) > 0.01).astype(int)
        df['amount_vs_balance_orig'] = (df['amount'] / (df['oldbalanceOrg'] + 1)).astype(float)
        df['amount_vs_balance_dest'] = (df['amount'] / (df['oldbalanceDest'] + 1)).astype(float)
        df['is_full_balance_orig'] = (abs(df['balance_change_orig']) / (df['oldbalanceOrg'] + 1) > 0.9).astype(int)
        df['balance_diff'] = (df['oldbalanceOrg'] - df['oldbalanceDest']).astype(float)
        df['has_enough_balance'] = (df['oldbalanceOrg'] >= df['amount']).astype(int)
        
        return df
    
    def create_customer_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level features with optimization"""
        logger.info("Creating customer features (optimized)")
        
        # Convert name columns to string if they are categorical
        if df['nameOrig'].dtype.name == 'category':
            df['nameOrig'] = df['nameOrig'].astype(str)
        if df['nameDest'].dtype.name == 'category':
            df['nameDest'] = df['nameDest'].astype(str)
        
        orig_groups = df.groupby('nameOrig', observed=True)
        
        df['orig_txn_count'] = orig_groups['amount'].transform('count').astype(int)
        df['orig_avg_amount'] = orig_groups['amount'].transform('mean').astype(float)
        df['orig_std_amount'] = orig_groups['amount'].transform('std').fillna(0).astype(float)
        df['orig_max_amount'] = orig_groups['amount'].transform('max').astype(float)
        df['orig_min_amount'] = orig_groups['amount'].transform('min').astype(float)
        df['orig_sum_amount'] = orig_groups['amount'].transform('sum').astype(float)
        df['orig_amount_range'] = (df['orig_max_amount'] - df['orig_min_amount']).astype(float)
        
        if 'isFraud' in df.columns:
            df['orig_fraud_ratio'] = orig_groups['isFraud'].transform('mean').astype(float)
            df['orig_fraud_count'] = orig_groups['isFraud'].transform('sum').astype(int)
        
        dest_groups = df.groupby('nameDest', observed=True)
        df['dest_txn_count'] = dest_groups['amount'].transform('count').astype(int)
        df['dest_avg_amount'] = dest_groups['amount'].transform('mean').astype(float)
        df['dest_std_amount'] = dest_groups['amount'].transform('std').fillna(0).astype(float)
        df['dest_max_amount'] = dest_groups['amount'].transform('max').astype(float)
        df['dest_sum_amount'] = dest_groups['amount'].transform('sum').astype(float)
        
        df['amount_vs_orig_avg'] = (df['amount'] / (df['orig_avg_amount'] + 1)).astype(float)
        df['amount_vs_dest_avg'] = (df['amount'] / (df['dest_avg_amount'] + 1)).astype(float)
        df['is_new_origin'] = (df['orig_txn_count'] == 1).astype(int)
        df['is_new_dest'] = (df['dest_txn_count'] == 1).astype(int)
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating time features")
        
        if 'step' in df.columns:
            df['hour'] = (df['step'] % 24).astype(int)
            df['day'] = ((df['step'] // 24) % 7).astype(int)
            df['is_weekend'] = (df['day'] >= 5).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
            df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 11)).astype(int)
            df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] <= 17)).astype(int)
            df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
            
            hour_freq = df.groupby('hour').size() / len(df)
            df['hour_frequency'] = df['hour'].map(hour_freq).astype(float)
            day_freq = df.groupby('day').size() / len(df)
            df['day_frequency'] = df['day'].map(day_freq).astype(float)
        
        return df
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features"""
        logger.info("Creating amount features")
        
        df['amount_log'] = np.log1p(df['amount']).astype(float)
        df['amount_z_score'] = ((df['amount'] - df['amount'].mean()) / (df['amount'].std() + 1e-6)).astype(float)
        df['amount_percentile'] = df['amount'].rank(pct=True).astype(float)
        
        # Convert to category with proper handling
        df['amount_category'] = pd.cut(
            df['amount'],
            bins=[0, 100, 1000, 10000, 100000, float('inf')],
            labels=['micro', 'small', 'medium', 'large', 'huge']
        ).astype(str)  # Convert to string to avoid categorical issues
        
        df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['is_rounded_amount'] = (df['amount'] % 10 == 0).astype(int)
        df['is_very_round'] = (df['amount'] % 1000 == 0).astype(int)
        df['has_decimals'] = (df['amount'] % 1 != 0).astype(int)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        logger.info("Creating interaction features")
        
        df['amount_type_risk'] = (df['amount'] * df['type_risk_score']).astype(float)
        df['balance_change_amount_interaction'] = (df['balance_change_orig'] * df['amount']).astype(float)
        
        if 'orig_fraud_ratio' in df.columns:
            df['amount_orig_fraud_ratio'] = (df['amount'] * df['orig_fraud_ratio']).astype(float)
        
        df['type_risk_balance_change'] = (df['type_risk_score'] * abs(df['balance_change_orig'])).astype(float)
        
        return df
    
    def create_velocity_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create velocity features with optimization"""
        logger.info("Creating velocity features (optimized)")
        
        # Ensure proper sorting
        df = df.sort_values(['nameOrig', 'step']).reset_index(drop=True)
        
        # Convert to float
        df['time_since_last_txn'] = df.groupby('nameOrig')['step'].diff().fillna(0).astype(float)
        
        windows = [1, 5, 24, 168]
        
        for window in windows:
            col_name = f'transactions_last_{window}h'
            df[col_name] = df.groupby('nameOrig')['step'].transform(
                lambda x: x.rolling(window, min_periods=1).count()
            ) - 1
            df[col_name] = df[col_name].clip(upper=50).astype(int)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables")
        
        categorical_cols = ['type', 'amount_category']
        
        for col in categorical_cols:
            if col in df.columns:
                # Ensure string type
                df[col] = df[col].astype(str)
                
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    # Handle unseen categories
                    for val in df[col].unique():
                        if val not in self.label_encoders[col].classes_:
                            df.loc[df[col] == val, col] = self.label_encoders[col].classes_[0]
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                
                # Ensure integer type
                df[f'{col}_encoded'] = df[f'{col}_encoded'].astype(int)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features"""
        logger.info("Creating all features")
        logger.info(f"Starting with {len(df):,} rows")
        
        # Create a copy and ensure no categorical issues
        df_featured = df.copy()
        
        # Convert categorical columns to string to avoid issues
        for col in df_featured.select_dtypes(include=['category']).columns:
            df_featured[col] = df_featured[col].astype(str)
        
        # Create all feature groups
        df_featured = self.create_transaction_type_features(df_featured)
        df_featured = self.create_balance_features(df_featured)
        df_featured = self.create_amount_features(df_featured)
        df_featured = self.create_customer_features_optimized(df_featured)
        df_featured = self.create_time_features(df_featured)
        df_featured = self.create_interaction_features(df_featured)
        df_featured = self.create_velocity_features_optimized(df_featured)
        df_featured = self.encode_categorical(df_featured)
        
        # Drop non-feature columns
        exclude_cols = ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud', 
                       'type', 'amount_category']
        
        # Keep only numeric features
        feature_cols = [col for col in df_featured.columns 
                       if col not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(df_featured[col])]
        
        self.feature_columns = feature_cols
        
        logger.info(f"Created {len(feature_cols)} features")
        logger.info(f"Feature columns: {feature_cols[:10]}...")
        logger.info(f"Memory usage: {df_featured.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        gc.collect()
        
        return df_featured
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features"""
        logger.info("Scaling features")
        
        if self.feature_columns is None:
            self.feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure all features are float before scaling
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        if fit:
            df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        else:
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df
    
    def save(self, path: str = "models/feature_engineer.pkl"):
        """Save feature engineer"""
        joblib.dump(self, path, compress=3)
        logger.info(f"FeatureEngineer saved to {path}")
    
    @classmethod
    def load(cls, path: str = "models/feature_engineer.pkl"):
        """Load feature engineer"""
        return joblib.load(path)