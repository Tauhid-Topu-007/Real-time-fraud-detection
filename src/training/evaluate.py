import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import joblib
from pathlib import Path

class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self):
        self.results = {}
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(self, model, X_test, y_test, threshold=0.5):
        """Evaluate model with detailed metrics"""
        logger.info("Evaluating model...")
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'threshold': threshold
        }
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        
        self.results = {
            'metrics': metrics,
            'precision_curve': precision,
            'recall_curve': recall,
            'pr_thresholds': thresholds,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds
        }
        
        return self.results
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pr_curve(self, precision, recall, pr_auc, save_path=None):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_path=None):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, marker='.')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_best_threshold(self, precision, recall, thresholds, metric='f1'):
        """Find best threshold based on metric"""
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        if metric == 'f1':
            best_idx = np.argmax(f1_scores[:-1])
        elif metric == 'precision':
            best_idx = np.argmax(precision[:-1])
        elif metric == 'recall':
            best_idx = np.argmax(recall[:-1])
        
        best_threshold = thresholds[best_idx]
        best_score = [precision[best_idx], recall[best_idx], f1_scores[best_idx]]
        
        return best_threshold, best_score
    
    def print_metrics(self):
        """Print evaluation metrics"""
        logger.info("\n" + "="*50)
        logger.info("Model Evaluation Results")
        logger.info("="*50)
        
        metrics = self.results.get('metrics', {})
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        return metrics