"""Tests for BiasDetector module."""

import pytest
import pandas as pd
import numpy as np
from ai_ethica.bias.detector import BiasDetector


def test_bias_detector_initialization():
    """Test BiasDetector initialization."""
    detector = BiasDetector()
    assert detector is not None
    assert len(detector.bias_reports) == 0


def test_analyze_basic():
    """Test basic bias analysis."""
    detector = BiasDetector()
    
    # Create simple dataset
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'gender': ['M', 'M', 'F', 'F', 'M'],
        'target': [0, 1, 0, 1, 0]
    })
    
    report = detector.analyze(
        data=data,
        protected_attributes=['gender'],
        target_column='target'
    )
    
    assert report['dataset_size'] == 5
    assert 'gender' in report['bias_metrics']
    assert 'recommendations' in report


def test_analyze_missing_attribute():
    """Test that missing protected attribute raises error."""
    detector = BiasDetector()
    data = pd.DataFrame({'feature1': [1, 2, 3]})
    
    with pytest.raises(ValueError):
        detector.analyze(data=data, protected_attributes=['missing_attr'])


def test_representation_bias_calculation():
    """Test representation bias calculation."""
    detector = BiasDetector()
    
    # Create imbalanced dataset
    data = pd.DataFrame({
        'group': ['A'] * 90 + ['B'] * 10,
        'target': [0, 1] * 50
    })
    
    report = detector.analyze(
        data=data,
        protected_attributes=['group']
    )
    
    rep_bias = report['bias_metrics']['group']['representation_bias']
    assert rep_bias['disparity_ratio'] > 1.0
    assert not rep_bias['is_balanced']

