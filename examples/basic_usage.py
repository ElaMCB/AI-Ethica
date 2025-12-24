"""
Basic Usage Examples for AI-Ethica

This script demonstrates how to use the AI-Ethica framework
to evaluate ethical aspects of AI models and datasets.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Import AI-Ethica modules
from ai_ethica import (
    BiasDetector,
    FairnessMetrics,
    TransparencyAnalyzer,
    PrivacyEvaluator,
    AccountabilityTracker
)


def example_bias_detection():
    """Example: Detecting bias in a dataset."""
    print("=" * 60)
    print("Example 1: Bias Detection")
    print("=" * 60)
    
    # Create sample dataset with potential bias
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.7, 0.3]),  # Imbalanced
        'race': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
    
    # Initialize bias detector
    detector = BiasDetector()
    
    # Analyze bias
    bias_report = detector.analyze(
        data=data,
        protected_attributes=['gender', 'race'],
        target_column='target'
    )
    
    print(f"\nDataset size: {bias_report['dataset_size']}")
    print(f"\nBias Metrics:")
    for attr, metrics in bias_report['bias_metrics'].items():
        print(f"\n  {attr}:")
        rep_bias = metrics['representation_bias']
        print(f"    Representation disparity ratio: {rep_bias['disparity_ratio']:.2f}")
        print(f"    Is balanced: {rep_bias['is_balanced']}")
    
    print(f"\nRecommendations:")
    for rec in bias_report['recommendations']:
        print(f"  - {rec}")


def example_fairness_metrics():
    """Example: Calculating fairness metrics."""
    print("\n" + "=" * 60)
    print("Example 2: Fairness Metrics")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    # Simulate predictions with some bias
    y_true = np.random.choice([0, 1], n_samples)
    protected_attr = np.random.choice(['group_A', 'group_B'], n_samples, p=[0.6, 0.4])
    
    # Create biased predictions (group_B has lower positive rate)
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        if protected_attr[i] == 'group_A':
            y_pred[i] = np.random.choice([0, 1], p=[0.4, 0.6])
        else:
            y_pred[i] = np.random.choice([0, 1], p=[0.7, 0.3])
        if y_true[i] == 1:
            y_pred[i] = y_true[i]  # Some correlation with true labels
    
    # Initialize fairness metrics
    metrics = FairnessMetrics()
    
    # Evaluate fairness
    fairness_results = metrics.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        protected_attributes={'gender': protected_attr}
    )
    
    print("\nFairness Results:")
    for attr, results in fairness_results.items():
        print(f"\n  {attr}:")
        if 'demographic_parity' in results:
            dp = results['demographic_parity']
            print(f"    Demographic Parity:")
            print(f"      Parity ratio: {dp['parity_ratio']:.2f}")
            print(f"      Is fair: {dp['is_fair']}")
        
        if 'equal_opportunity' in results:
            eo = results['equal_opportunity']
            print(f"    Equal Opportunity:")
            print(f"      TPR violation: {eo['violation']:.3f}")
            print(f"      Is fair: {eo['is_fair']}")


def example_transparency_analysis():
    """Example: Analyzing model transparency."""
    print("\n" + "=" * 60)
    print("Example 3: Transparency Analysis")
    print("=" * 60)
    
    # Create and train a simple model
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize transparency analyzer
    analyzer = TransparencyAnalyzer()
    
    # Assess transparency
    transparency_report = analyzer.assess(
        model=model,
        X=X_test[:10],  # Sample data
        feature_names=[f'feature_{i}' for i in range(5)],
        has_documentation=True,
        has_explanations=False
    )
    
    print(f"\nModel Type: {transparency_report['model_type']}")
    print(f"Interpretability Score: {transparency_report['interpretability_score']:.2f}")
    print(f"Overall Transparency Score: {transparency_report['transparency_score']:.2f}")
    
    print("\nFactors:")
    for factor, details in transparency_report['factors'].items():
        if isinstance(details, dict) and 'score' in details:
            print(f"  {factor}: {details['score']:.2f}")
    
    print("\nRecommendations:")
    for rec in transparency_report['recommendations']:
        print(f"  - {rec}")


def example_privacy_evaluation():
    """Example: Evaluating privacy measures."""
    print("\n" + "=" * 60)
    print("Example 4: Privacy Evaluation")
    print("=" * 60)
    
    # Create sample dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'id': range(100),  # Unique identifier - privacy risk!
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'sensitive_info': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Initialize privacy evaluator
    evaluator = PrivacyEvaluator()
    
    # Evaluate privacy
    privacy_report = evaluator.evaluate(
        data=data,
        sensitive_columns=['sensitive_info'],
        has_anonymization=False,
        has_differential_privacy=False,
        has_access_controls=True
    )
    
    print(f"\nPrivacy Score: {privacy_report['privacy_score']:.2f}")
    print(f"\nRisks Identified:")
    for risk in privacy_report['risks']:
        print(f"  - {risk}")
    
    print("\nRecommendations:")
    for rec in privacy_report['recommendations']:
        print(f"  - {rec}")


def example_accountability_tracking():
    """Example: Tracking model decisions for accountability."""
    print("\n" + "=" * 60)
    print("Example 5: Accountability Tracking")
    print("=" * 60)
    
    # Initialize accountability tracker
    tracker = AccountabilityTracker(log_dir="example_audit_logs")
    
    # Log some decisions
    for i in range(5):
        decision_id = tracker.log_decision(
            model_id="model_v1",
            input_data={"feature1": 0.5, "feature2": 0.3},
            prediction=1,
            confidence=0.85,
            metadata={"user_id": f"user_{i}"}
        )
        print(f"Logged decision: {decision_id}")
    
    # Log an incident
    incident_id = tracker.log_incident(
        incident_type="bias_detected",
        description="High demographic parity violation detected",
        severity="high",
        model_id="model_v1"
    )
    print(f"\nLogged incident: {incident_id}")
    
    # Generate report
    report = tracker.generate_report(model_id="model_v1", period_days=1)
    print(f"\nAccountability Report:")
    print(f"  Total decisions: {report['summary']['total_decisions']}")
    print(f"  Total incidents: {report['summary']['total_incidents']}")
    print(f"  Open incidents: {report['summary']['open_incidents']}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")


if __name__ == "__main__":
    # Run all examples
    example_bias_detection()
    example_fairness_metrics()
    example_transparency_analysis()
    example_privacy_evaluation()
    example_accountability_tracking()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

