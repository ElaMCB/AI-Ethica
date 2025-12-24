"""
AI-Ethica: A comprehensive framework for ethical AI evaluation and practices.
"""

__version__ = "0.1.0"

from .bias.detector import BiasDetector
from .fairness.metrics import FairnessMetrics
from .transparency.analyzer import TransparencyAnalyzer
from .privacy.evaluator import PrivacyEvaluator
from .accountability.tracker import AccountabilityTracker

__all__ = [
    "BiasDetector",
    "FairnessMetrics",
    "TransparencyAnalyzer",
    "PrivacyEvaluator",
    "AccountabilityTracker",
]

