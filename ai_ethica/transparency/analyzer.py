"""
Transparency Analysis Module

This module provides tools for assessing model transparency, interpretability,
and explainability.
"""

import numpy as np
from typing import Dict, Optional, List, Any
import warnings


class TransparencyAnalyzer:
    """
    A class for analyzing model transparency and explainability.
    
    Evaluates:
    - Model interpretability
    - Feature importance availability
    - Explanation quality
    - Documentation completeness
    """
    
    def __init__(self):
        """Initialize the TransparencyAnalyzer."""
        pass
    
    def assess(
        self,
        model: Any,
        X: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        has_documentation: bool = False,
        has_explanations: bool = False
    ) -> Dict:
        """
        Assess overall model transparency.
        
        Parameters:
        -----------
        model : Any
            The model to assess (should have predict method)
        X : np.ndarray, optional
            Sample data for feature importance analysis
        feature_names : List[str], optional
            Names of features
        has_documentation : bool
            Whether the model has documentation
        has_explanations : bool
            Whether explanations are available
        
        Returns:
        --------
        Dict containing transparency assessment
        """
        assessment = {
            "model_type": self._get_model_type(model),
            "interpretability_score": 0.0,
            "transparency_score": 0.0,
            "factors": {}
        }
        
        # Assess interpretability
        interpretability = self._assess_interpretability(model)
        assessment["factors"]["interpretability"] = interpretability
        assessment["interpretability_score"] = interpretability["score"]
        
        # Assess feature importance availability
        if X is not None:
            feature_importance = self._assess_feature_importance(model, X, feature_names)
            assessment["factors"]["feature_importance"] = feature_importance
        else:
            assessment["factors"]["feature_importance"] = {
                "available": False,
                "score": 0.0,
                "note": "No sample data provided for analysis"
            }
        
        # Assess documentation
        doc_score = 1.0 if has_documentation else 0.0
        assessment["factors"]["documentation"] = {
            "available": has_documentation,
            "score": doc_score
        }
        
        # Assess explanations
        exp_score = 1.0 if has_explanations else 0.0
        assessment["factors"]["explanations"] = {
            "available": has_explanations,
            "score": exp_score
        }
        
        # Calculate overall transparency score
        scores = [
            interpretability["score"],
            assessment["factors"]["feature_importance"]["score"],
            doc_score,
            exp_score
        ]
        assessment["transparency_score"] = float(np.mean(scores))
        
        # Generate recommendations
        assessment["recommendations"] = self._generate_recommendations(assessment)
        
        return assessment
    
    def _get_model_type(self, model: Any) -> str:
        """Determine the type of model."""
        model_class = type(model).__name__
        
        # Interpretable models
        interpretable = ['LinearRegression', 'LogisticRegression', 'DecisionTreeClassifier',
                        'DecisionTreeRegressor', 'RuleBased', 'LinearModel']
        
        # Black box models
        black_box = ['RandomForest', 'GradientBoosting', 'XGBoost', 'NeuralNetwork',
                    'SVM', 'KNeighbors']
        
        if any(name in model_class for name in interpretable):
            return "interpretable"
        elif any(name in model_class for name in black_box):
            return "black_box"
        else:
            return "unknown"
    
    def _assess_interpretability(self, model: Any) -> Dict:
        """Assess model interpretability."""
        model_type = self._get_model_type(model)
        
        if model_type == "interpretable":
            return {
                "score": 1.0,
                "level": "high",
                "reason": "Model is inherently interpretable"
            }
        elif model_type == "black_box":
            return {
                "score": 0.3,
                "level": "low",
                "reason": "Model is a black box, requires post-hoc explanations"
            }
        else:
            return {
                "score": 0.5,
                "level": "medium",
                "reason": "Model type unknown, interpretability unclear"
            }
    
    def _assess_feature_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """Assess availability of feature importance."""
        try:
            # Try to get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return {
                    "available": True,
                    "score": 1.0,
                    "method": "built-in",
                    "importances": importances.tolist() if hasattr(importances, 'tolist') else importances,
                    "feature_names": feature_names or [f"feature_{i}" for i in range(len(importances))]
                }
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                # Take absolute values for importance
                importances = np.abs(coef[0] if coef.ndim > 1 else coef)
                return {
                    "available": True,
                    "score": 1.0,
                    "method": "coefficients",
                    "importances": importances.tolist() if hasattr(importances, 'tolist') else importances,
                    "feature_names": feature_names or [f"feature_{i}" for i in range(len(importances))]
                }
            else:
                return {
                    "available": False,
                    "score": 0.0,
                    "method": None,
                    "note": "Model does not provide feature importance directly"
                }
        except Exception as e:
            return {
                "available": False,
                "score": 0.0,
                "error": str(e),
                "note": "Could not extract feature importance"
            }
    
    def _generate_recommendations(self, assessment: Dict) -> List[str]:
        """Generate recommendations for improving transparency."""
        recommendations = []
        
        if assessment["transparency_score"] < 0.5:
            recommendations.append(
                "Model transparency is low. Consider using more interpretable models "
                "or implementing post-hoc explanation methods."
            )
        
        if not assessment["factors"]["documentation"]["available"]:
            recommendations.append(
                "Add comprehensive documentation including model purpose, "
                "training data, limitations, and usage guidelines."
            )
        
        if not assessment["factors"]["explanations"]["available"]:
            recommendations.append(
                "Implement explanation methods (SHAP, LIME, etc.) to provide "
                "interpretable explanations for model predictions."
            )
        
        if assessment["factors"]["feature_importance"]["score"] < 0.5:
            recommendations.append(
                "Provide feature importance information to help users understand "
                "which features drive model decisions."
            )
        
        if not recommendations:
            recommendations.append("Model transparency is good. Continue maintaining documentation and explanations.")
        
        return recommendations

