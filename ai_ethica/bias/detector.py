"""
Bias Detection Module

This module provides tools for detecting and measuring bias in datasets and models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from collections import defaultdict


class BiasDetector:
    """
    A class for detecting and analyzing bias in datasets and machine learning models.
    
    This detector can identify:
    - Statistical disparities across protected attributes
    - Representation bias in datasets
    - Label bias in training data
    - Measurement bias in features
    """
    
    def __init__(self):
        """Initialize the BiasDetector."""
        self.bias_reports = []
    
    def analyze(
        self,
        data: pd.DataFrame,
        protected_attributes: List[str],
        target_column: Optional[str] = None,
        sensitive_groups: Optional[Dict[str, List]] = None
    ) -> Dict:
        """
        Analyze bias in a dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to analyze
        protected_attributes : List[str]
            List of column names representing protected attributes (e.g., ['gender', 'race'])
        target_column : str, optional
            Name of the target/label column if analyzing label bias
        sensitive_groups : Dict[str, List], optional
            Dictionary mapping protected attributes to lists of sensitive group values
        
        Returns:
        --------
        Dict containing bias analysis results
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        if not protected_attributes:
            raise ValueError("protected_attributes cannot be empty")
        
        report = {
            "dataset_size": len(data),
            "protected_attributes": protected_attributes,
            "bias_metrics": {},
            "recommendations": []
        }
        
        for attr in protected_attributes:
            if attr not in data.columns:
                raise ValueError(f"Protected attribute '{attr}' not found in dataset")
            
            attr_report = self._analyze_attribute(data, attr, target_column)
            report["bias_metrics"][attr] = attr_report
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report["bias_metrics"])
        
        self.bias_reports.append(report)
        return report
    
    def _analyze_attribute(
        self,
        data: pd.DataFrame,
        attribute: str,
        target_column: Optional[str] = None
    ) -> Dict:
        """Analyze bias for a specific protected attribute."""
        value_counts = data[attribute].value_counts()
        total = len(data)
        
        analysis = {
            "attribute": attribute,
            "group_distribution": (value_counts / total).to_dict(),
            "group_counts": value_counts.to_dict(),
            "representation_bias": self._calculate_representation_bias(value_counts, total),
        }
        
        if target_column and target_column in data.columns:
            analysis["label_bias"] = self._calculate_label_bias(
                data, attribute, target_column
            )
        
        return analysis
    
    def _calculate_representation_bias(
        self,
        value_counts: pd.Series,
        total: int
    ) -> Dict:
        """Calculate representation bias across groups."""
        proportions = value_counts / total
        max_prop = proportions.max()
        min_prop = proportions.min()
        
        return {
            "max_proportion": float(max_prop),
            "min_proportion": float(min_prop),
            "disparity_ratio": float(max_prop / min_prop) if min_prop > 0 else float('inf'),
            "is_balanced": max_prop / min_prop < 1.5 if min_prop > 0 else False
        }
    
    def _calculate_label_bias(
        self,
        data: pd.DataFrame,
        attribute: str,
        target_column: str
    ) -> Dict:
        """Calculate label bias across protected groups."""
        label_bias = {}
        
        for group_value in data[attribute].unique():
            group_data = data[data[attribute] == group_value]
            if len(group_data) > 0:
                positive_rate = (group_data[target_column] == 1).mean() if group_data[target_column].dtype in [int, float] else None
                label_bias[str(group_value)] = {
                    "group_size": len(group_data),
                    "positive_rate": float(positive_rate) if positive_rate is not None else None,
                    "mean_target": float(group_data[target_column].mean()) if positive_rate is None else None
                }
        
        # Calculate disparity
        if label_bias:
            rates = [v.get("positive_rate") or v.get("mean_target", 0) for v in label_bias.values()]
            rates = [r for r in rates if r is not None]
            if rates:
                max_rate = max(rates)
                min_rate = min(rates)
                label_bias["disparity"] = {
                    "max_rate": float(max_rate),
                    "min_rate": float(min_rate),
                    "disparity_ratio": float(max_rate / min_rate) if min_rate > 0 else float('inf')
                }
        
        return label_bias
    
    def _generate_recommendations(self, bias_metrics: Dict) -> List[str]:
        """Generate recommendations based on bias analysis."""
        recommendations = []
        
        for attr, metrics in bias_metrics.items():
            rep_bias = metrics.get("representation_bias", {})
            if rep_bias.get("disparity_ratio", 1) > 2.0:
                recommendations.append(
                    f"High representation disparity detected in '{attr}'. "
                    f"Consider data collection strategies to improve balance."
                )
            
            label_bias = metrics.get("label_bias", {})
            if isinstance(label_bias, dict) and "disparity" in label_bias:
                disparity = label_bias["disparity"]
                if disparity.get("disparity_ratio", 1) > 1.5:
                    recommendations.append(
                        f"Label bias detected in '{attr}'. "
                        f"Review labeling process and consider fairness constraints."
                    )
        
        if not recommendations:
            recommendations.append("No significant bias detected. Continue monitoring.")
        
        return recommendations
    
    def detect_model_bias(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        protected_attributes: List[str]
    ) -> Dict:
        """
        Detect bias in model predictions.
        
        Parameters:
        -----------
        model : sklearn-like model
            Trained model with predict() method
        X : pd.DataFrame
            Feature data
        y : pd.Series
            True labels
        protected_attributes : List[str]
            List of protected attribute column names
        
        Returns:
        --------
        Dict containing model bias analysis
        """
        predictions = model.predict(X)
        
        bias_report = {
            "model_predictions": predictions.tolist(),
            "protected_attributes": protected_attributes,
            "group_performance": {}
        }
        
        for attr in protected_attributes:
            if attr not in X.columns:
                continue
            
            group_perf = {}
            for group_value in X[attr].unique():
                group_mask = X[attr] == group_value
                group_pred = predictions[group_mask]
                group_true = y[group_mask]
                
                if len(group_pred) > 0:
                    accuracy = (group_pred == group_true).mean()
                    positive_pred_rate = (group_pred == 1).mean() if len(np.unique(group_pred)) > 1 else None
                    
                    group_perf[str(group_value)] = {
                        "size": int(group_mask.sum()),
                        "accuracy": float(accuracy),
                        "positive_prediction_rate": float(positive_pred_rate) if positive_pred_rate is not None else None
                    }
            
            bias_report["group_performance"][attr] = group_perf
        
        return bias_report

