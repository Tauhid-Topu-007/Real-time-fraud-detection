import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple, List

class DataValidator:
    """Validate data quality and integrity"""
    
    def __init__(self):
        self.required_cols = [
            'step', 'type', 'amount', 'nameOrig', 
            'oldbalanceOrg', 'newbalanceOrig',
            'nameDest', 'oldbalanceDest', 'newbalanceDest',
            'isFraud', 'isFlaggedFraud'
        ]
        
        self.valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        
        logger.info("DataValidator initialized")
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate dataframe schema"""
        missing_cols = [col for col in self.required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        logger.info("Schema validation passed")
        return True
    
    def validate_values(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data values"""
        issues = []
        
        invalid_types = df[~df['type'].isin(self.valid_types)]['type'].unique()
        if len(invalid_types) > 0:
            issues.append(f"Invalid transaction types: {invalid_types}")
        
        if (df['amount'] < 0).any():
            negative_count = (df['amount'] < 0).sum()
            issues.append(f"Found {negative_count:,} negative amounts")
        
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            issues.append(f"Null values found in: {null_cols}")
        
        fraud_rate = df['isFraud'].mean()
        logger.info(f"Fraud rate: {fraud_rate:.4%}")
        
        if fraud_rate == 0:
            issues.append("No fraud cases found in dataset")
        
        balance_errors = (
            abs(df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']) > 0.01
        ).sum()
        if balance_errors > len(df) * 0.1:
            issues.append(f"Found {balance_errors:,} balance inconsistencies")
        
        if issues:
            logger.warning(f"Validation issues found: {issues}")
            return False, issues
        
        logger.info("All validations passed")
        return True, []
    
    def get_data_stats(self, df: pd.DataFrame) -> dict:
        """Get data statistics"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'fraud_count': df['isFraud'].sum(),
            'fraud_rate': df['isFraud'].mean(),
            'total_amount': df['amount'].sum(),
            'avg_amount': df['amount'].mean(),
            'max_amount': df['amount'].max(),
            'transaction_types': df['type'].value_counts().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
        }