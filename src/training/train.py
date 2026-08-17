import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report)
import mlflow
import mlflow.sklearn
from loguru import logger
import yaml
import joblib
from pathlib import Path
from imblearn.over_sampling import SMOTE
import gc
import warnings
warnings.filterwarnings('ignore')

class BigDataFraudModelTrainer:
    """Train and evaluate fraud detection models for big data"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.model_type = self.model_config['type']
        self.params = self.model_config['parameters']
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        if self.model_type == 'xgboost':
            self.params['tree_method'] = 'hist'
            self.params['n_jobs'] = -1
            self.params['verbosity'] = 0
        elif self.model_type == 'lightgbm':
            self.params['n_jobs'] = -1
            self.params['verbose'] = -1
        
        logger.info(f"BigDataModelTrainer initialized with {self.model_type}")
    
    def get_model(self):
        """Initialize model based on configuration"""
        if self.model_type == 'xgboost':
            return XGBClassifier(**self.params)
        elif self.model_type == 'lightgbm':
            return LGBMClassifier(**self.params)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**self.params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance"""
        logger.info(f"Handling imbalance using {method}")
        
        if method == 'smote':
            if len(X_train) > 100000:
                logger.info("Large dataset detected, using sampled SMOTE")
                combined = pd.concat([X_train, y_train], axis=1)
                fraud = combined[combined[y_train.name] == 1]
                non_fraud = combined[combined[y_train.name] == 0].sample(
                    n=min(len(fraud) * 10, 100000),
                    random_state=42
                )
                combined_sample = pd.concat([fraud, non_fraud])
                X_sampled = combined_sample.drop(columns=[y_train.name])
                y_sampled = combined_sample[y_train.name]
                
                smote = SMOTE(random_state=42, n_jobs=-1)
                X_resampled, y_resampled = smote.fit_resample(X_sampled, y_sampled)
            else:
                smote = SMOTE(random_state=42, n_jobs=-1)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            logger.info(f"Resampled to {len(X_resampled):,} samples")
            return X_resampled, y_resampled
        else:
            return X_train, y_train
    
    def train(self, X_train, y_train, use_smote=True):
        """Train the model with MLflow tracking"""
        
        with mlflow.start_run(run_name=f"{self.model_type}_training") as run:
            
            mlflow.log_params(self.params)
            mlflow.log_param("use_smote", use_smote)
            
            if use_smote:
                X_train_processed, y_train_processed = self.handle_imbalance(X_train, y_train)
            else:
                X_train_processed, y_train_processed = X_train, y_train
            
            if 'scale_pos_weight' not in self.params:
                fraud_ratio = y_train.sum() / len(y_train)
                scale_pos_weight = (1 - fraud_ratio) / (fraud_ratio + 1e-6)
                self.params['scale_pos_weight'] = min(scale_pos_weight, 100)
                logger.info(f"Set scale_pos_weight to {self.params['scale_pos_weight']:.2f}")
            
            model = self.get_model()
            
            logger.info(f"Training {self.model_type} model...")
            model.fit(
                X_train_processed, 
                y_train_processed,
                eval_metric='aucpr' if self.model_type == 'xgboost' else None
            )
            
            mlflow.sklearn.log_model(model, "fraud_model")
            
            model_path = self.models_dir / f"{self.model_type}_model.pkl"
            joblib.dump(model, model_path, compress=3)
            logger.info(f"Model saved to {model_path}")
            
            del X_train_processed, y_train_processed
            gc.collect()
            
            return model
    
    def evaluate(self, model, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            logger.info(f"{metric_name}: {value:.4f}")
        
        logger.info("\n" + "="*50)
        logger.info("Classification Report")
        logger.info("="*50)
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        return metrics, cm
    
    def find_optimal_threshold(self, model, X_val, y_val):
        """Find optimal threshold for fraud detection"""
        logger.info("Finding optimal threshold...")
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        results_df = pd.DataFrame(results)
        logger.info(f"Best threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
        
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1_at_threshold", best_f1)
        
        return best_threshold, results_df
    
    def train_pipeline(self, df: pd.DataFrame, target_col: str = 'isFraud'):
        """Complete training pipeline"""
        logger.info("Starting training pipeline...")
        logger.info("="*50)
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Validation set: {len(X_val):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        logger.info(f"Fraud rate in training: {y_train.mean():.4f}")
        
        model = self.train(X_train, y_train, use_smote=True)
        
        best_threshold, threshold_results = self.find_optimal_threshold(model, X_val, y_val)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
        
        metrics_optimal = {
            'precision': precision_score(y_test, y_pred_optimal),
            'recall': recall_score(y_test, y_pred_optimal),
            'f1_score': f1_score(y_test, y_pred_optimal),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'optimal_threshold': best_threshold
        }
        
        logger.info("\n" + "="*50)
        logger.info("Final Model Performance (Optimal Threshold)")
        logger.info("="*50)
        for metric, value in metrics_optimal.items():
            logger.info(f"{metric}: {value:.4f}")
        
        self.config['thresholds']['optimal'] = float(best_threshold)
        with open("configs/config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        del X_train, X_val, X_test, y_train, y_val, y_test
        gc.collect()
        
        return model, metrics_optimal, {
            'threshold_results': threshold_results
        }