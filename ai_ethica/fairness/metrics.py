"""
Fairness Metrics Module

This module provides various fairness metrics for evaluating model fairness
across different protected groups.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from sklearn.metrics import confusion_matrix


class FairnessMetrics:
    """
    A class for calculating various fairness metrics.
    
    Supported metrics:
    - Demographic Parity (Statistical Parity)
    - Equalized Odds
    - Equal Opportunity
    - Calibration
    - Individual Fairness
    """
    
    def __init__(self):
        """Initialize the FairnessMetrics calculator."""
        pass
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attributes: Union[pd.DataFrame, Dict[str, np.ndarray]],
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate fairness metrics for a model.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        protected_attributes : pd.DataFrame or Dict[str, np.ndarray]
            Protected attribute values for each sample
        metrics : List[str], optional
            List of metrics to calculate. If None, calculates all available metrics.
            Options: ['demographic_parity', 'equalized_odds', 'equal_opportunity', 'calibration']
        
        Returns:
        --------
        Dict containing fairness metric scores
        """
        if metrics is None:
            metrics = ['demographic_parity', 'equalized_odds', 'equal_opportunity', 'calibration']
        
        results = {}
        
        # Convert protected_attributes to dict if DataFrame
        if isinstance(protected_attributes, pd.DataFrame):
            protected_dict = {col: protected_attributes[col].values for col in protected_attributes.columns}
        else:
            protected_dict = protected_attributes
        
        for attr_name, attr_values in protected_dict.items():
            attr_results = {}
            
            if 'demographic_parity' in metrics:
                attr_results['demographic_parity'] = self.demographic_parity(
                    y_pred, attr_values
                )
            
            if 'equalized_odds' in metrics:
                attr_results['equalized_odds'] = self.equalized_odds(
                    y_true, y_pred, attr_values
                )
            
            if 'equal_opportunity' in metrics:
                attr_results['equal_opportunity'] = self.equal_opportunity(
                    y_true, y_pred, attr_values
                )
            
            if 'calibration' in metrics:
                attr_results['calibration'] = self.calibration(
                    y_true, y_pred, attr_values
                )
            
            results[attr_name] = attr_results
        
        return results
    
    def demographic_parity(
        self,
        y_pred: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict:
        """
        Calculate Demographic Parity (Statistical Parity).
        
        Demographic parity requires that the positive prediction rate
        is the same across all protected groups.
        
        Returns:
        --------
        Dict with parity ratio and violation score
        """
        groups = np.unique(protected_attr)
        positive_rates = {}
        
        for group in groups:
            group_mask = protected_attr == group
            positive_rates[str(group)] = float(y_pred[group_mask].mean())
        
        rates = list(positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)
        
        return {
            "positive_rates": positive_rates,
            "parity_ratio": float(max_rate / min_rate) if min_rate > 0 else float('inf'),
            "violation": float(max_rate - min_rate),
            "is_fair": max_rate - min_rate < 0.05  # 5% threshold
        }
    
    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict:
        """
        Calculate Equalized Odds.
        
        Equalized odds requires that true positive rate (TPR) and
        false positive rate (FPR) are equal across protected groups.
        
        Returns:
        --------
        Dict with TPR and FPR for each group and violation scores
        """
        groups = np.unique(protected_attr)
        group_metrics = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            group_metrics[str(group)] = {
                "tpr": float(tpr),
                "fpr": float(fpr),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        
        tprs = [m["tpr"] for m in group_metrics.values()]
        fprs = [m["fpr"] for m in group_metrics.values()]
        
        tpr_violation = max(tprs) - min(tprs)
        fpr_violation = max(fprs) - min(fprs)
        
        return {
            "group_metrics": group_metrics,
            "tpr_violation": float(tpr_violation),
            "fpr_violation": float(fpr_violation),
            "max_tpr_violation": float(tpr_violation),
            "max_fpr_violation": float(fpr_violation),
            "is_fair": tpr_violation < 0.05 and fpr_violation < 0.05
        }
    
    def equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict:
        """
        Calculate Equal Opportunity.
        
        Equal opportunity requires that true positive rate (TPR)
        is equal across protected groups (relaxed version of equalized odds).
        
        Returns:
        --------
        Dict with TPR for each group and violation score
        """
        groups = np.unique(protected_attr)
        tprs = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tprs[str(group)] = float(tpr)
        
        rates = list(tprs.values())
        max_tpr = max(rates)
        min_tpr = min(rates)
        
        return {
            "tprs": tprs,
            "violation": float(max_tpr - min_tpr),
            "is_fair": (max_tpr - min_tpr) < 0.05
        }
    
    def calibration(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        protected_attr: np.ndarray
    ) -> Dict:
        """
        Calculate Calibration metrics.
        
        Calibration requires that predicted probabilities are well-calibrated
        across protected groups.
        
        Parameters:
        -----------
        y_pred_proba : np.ndarray
            Predicted probabilities (not binary predictions)
        
        Returns:
        --------
        Dict with calibration metrics for each group
        """
        groups = np.unique(protected_attr)
        calibration_metrics = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_proba = y_pred_proba[group_mask]
            
            # Expected positive rate vs actual positive rate
            expected_rate = float(group_proba.mean())
            actual_rate = float(group_y_true.mean())
            
            calibration_metrics[str(group)] = {
                "expected_positive_rate": expected_rate,
                "actual_positive_rate": actual_rate,
                "calibration_error": float(abs(expected_rate - actual_rate))
            }
        
        errors = [m["calibration_error"] for m in calibration_metrics.values()]
        max_error = max(errors)
        
        return {
            "group_calibration": calibration_metrics,
            "max_calibration_error": float(max_error),
            "is_calibrated": max_error < 0.05
        }

