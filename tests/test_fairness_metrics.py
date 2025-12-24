"""Tests for FairnessMetrics module."""

import pytest
import numpy as np
from ai_ethica.fairness.metrics import FairnessMetrics


def test_fairness_metrics_initialization():
    """Test FairnessMetrics initialization."""
    metrics = FairnessMetrics()
    assert metrics is not None


def test_demographic_parity():
    """Test demographic parity calculation."""
    metrics = FairnessMetrics()
    
    # Create predictions with different positive rates
    y_pred = np.array([1] * 60 + [0] * 40)  # 60% positive
    protected_attr = np.array(['A'] * 50 + ['B'] * 50)
    
    result = metrics.demographic_parity(y_pred, protected_attr)
    
    assert 'positive_rates' in result
    assert 'parity_ratio' in result
    assert 'violation' in result
    assert 'is_fair' in result


def test_equal_opportunity():
    """Test equal opportunity calculation."""
    metrics = FairnessMetrics()
    
    # Create simple test case
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    protected_attr = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    
    result = metrics.equal_opportunity(y_true, y_pred, protected_attr)
    
    assert 'tprs' in result
    assert 'violation' in result
    assert 'is_fair' in result

