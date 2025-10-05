#!/usr/bin/env python
"""
Evaluate Bloom Prediction Model v2 Accuracy

This script provides comprehensive accuracy metrics including:
- Cross-validation scores
- Confusion matrix
- ROC-AUC, Precision, Recall, F1-Score
- Per-species performance
- Temporal validation (train on past, test on future)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

def evaluate_model():
    """Run comprehensive model evaluation"""
    
    print("=" * 80)
    print(" BLOOM PREDICTION MODEL v2 - ACCURACY EVALUATION")
    print("=" * 80)
    
    # Load model
    print("\n[1/5] Loading model...")
    from bloom_predictor_v2 import ImprovedBloomPredictor
    
    predictor = ImprovedBloomPredictor(use_earth_engine=False)

    # Wait for the model to finish training in the background
    import time
    start_time = time.time()
    while predictor.is_training:
        print("Waiting for model to finish training...")
        time.sleep(5)
        if time.time() - start_time > 300: # 5 minute timeout
            print("Timeout waiting for model to train.")
            sys.exit(1)
    
    print(f"✓ Model loaded")
    print(f"  Training samples: {len(predictor.feature_data)}")
    print(f"  Positive (blooms): {len(predictor.historical_blooms)}")
    print(f"  Negative (no-blooms): {len(predictor.negative_examples)}")
    print(f"  Features: {len(predictor.feature_columns)}")
    
    # Prepare data
    print("\n[2/5] Preparing evaluation data...")
    X = predictor.feature_data[predictor.feature_columns].copy()
    y = predictor.feature_data['bloom'].copy()
    X = X.fillna(0)
    X_scaled = predictor.scaler.transform(X)
    
    # Time-series cross-validation
    print("\n[3/5] Running time-series cross-validation...")
    print("  Using 5-fold time-series split (respects temporal order)")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = {
        'roc_auc': cross_val_score(predictor.model, X_scaled, y, cv=tscv, scoring='roc_auc', n_jobs=-1),
        'accuracy': cross_val_score(predictor.model, X_scaled, y, cv=tscv, scoring='accuracy', n_jobs=-1),
        'precision': cross_val_score(predictor.model, X_scaled, y, cv=tscv, scoring='precision', n_jobs=-1),
        'recall': cross_val_score(predictor.model, X_scaled, y, cv=tscv, scoring='recall', n_jobs=-1),
        'f1': cross_val_score(predictor.model, X_scaled, y, cv=tscv, scoring='f1', n_jobs=-1)
    }
    
    print("\n✓ Cross-Validation Results (5 folds):")
    print("  " + "-" * 60)
    print(f"  {'Metric':<20} {'Mean':<10} {'Std Dev':<10} {'Range'}")
    print("  " + "-" * 60)
    for metric, scores in cv_scores.items():
        print(f"  {metric.upper():<20} {scores.mean():.3f}      ±{scores.std():.3f}      [{scores.min():.3f} - {scores.max():.3f}]")
    print("  " + "-" * 60)
    
    # Full model performance
    print("\n[4/5] Evaluating full model performance...")
    y_pred = predictor.model.predict(X_scaled)
    y_pred_proba = predictor.model.predict_proba(X_scaled)[:, 1]
    
    print("\n✓ Overall Model Performance:")
    print("  " + "-" * 60)
    print(f"  Accuracy:       {accuracy_score(y, y_pred):.3f}  (% of correct predictions)")
    print(f"  Precision:      {precision_score(y, y_pred):.3f}  (% of predicted blooms that are real)")
    print(f"  Recall:         {recall_score(y, y_pred):.3f}  (% of real blooms we caught)")
    print(f"  F1-Score:       {f1_score(y, y_pred):.3f}  (harmonic mean of precision/recall)")
    print(f"  ROC-AUC:        {roc_auc_score(y, y_pred_proba):.3f}  (probability calibration quality)")
    print("  " + "-" * 60)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n✓ Confusion Matrix:")
    print("  " + "-" * 60)
    print(f"                    Predicted NO BLOOM    Predicted BLOOM")
    print(f"  Actual NO BLOOM         {tn:6d}              {fp:6d}")
    print(f"  Actual BLOOM            {fn:6d}              {tp:6d}")
    print("  " + "-" * 60)
    print(f"  True Negatives:  {tn:6d}  (correctly predicted no blooms)")
    print(f"  False Positives: {fp:6d}  (false alarms)")
    print(f"  False Negatives: {fn:6d}  (missed blooms)")
    print(f"  True Positives:  {tp:6d}  (correctly predicted blooms)")
    print("  " + "-" * 60)
    
    # Error analysis
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n  False Positive Rate: {false_positive_rate:.1%}  (false alarms)")
    print(f"  False Negative Rate: {false_negative_rate:.1%}  (missed blooms)")
    
    # Probability calibration analysis
    print("\n[5/5] Analyzing probability calibration...")
    
    prob_bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    
    print("\n✓ Probability Calibration (how well probabilities match reality):")
    print("  " + "-" * 70)
    print(f"  {'Probability Range':<20} {'Predictions':<15} {'Actual Blooms':<15} {'Accuracy'}")
    print("  " + "-" * 70)
    
    for low, high in prob_bins:
        mask = (y_pred_proba >= low) & (y_pred_proba < high)
        if mask.sum() > 0:
            actual_rate = y[mask].mean()
            count = mask.sum()
            print(f"  {low:.0%} - {high:.0%}{'':<14} {count:<15d} {y[mask].sum():<15d} {actual_rate:.1%}")
    print("  " + "-" * 70)
    
    # Temporal validation
    print("\n✓ Temporal Validation (train on past, test on future):")
    print("  Testing if model generalizes to future dates...")
    
    # Reconstruct date information from historical_blooms and negative_examples
    # Combine them to get date info
    all_data = pd.concat([
        predictor.historical_blooms[['scientificName', 'lat', 'lon', 'date', 'day_of_year']].assign(bloom=1),
        predictor.negative_examples[['scientificName', 'lat', 'lon', 'date', 'day_of_year']].assign(bloom=0)
    ], ignore_index=True)
    
    all_data['date'] = pd.to_datetime(all_data['date'])
    all_data = all_data.sort_values('date').reset_index(drop=True)
    
    # Split: train on first 80%, test on last 20%
    split_idx = int(len(all_data) * 0.8)
    train_indices = all_data.index[:split_idx]
    test_indices = all_data.index[split_idx:]
    
    # Use feature_data rows corresponding to these indices
    train_data = predictor.feature_data.iloc[train_indices]
    test_data = predictor.feature_data.iloc[test_indices]
    train_dates = all_data.iloc[train_indices]
    test_dates = all_data.iloc[test_indices]
    
    X_train = train_data[predictor.feature_columns].fillna(0)
    y_train = train_data['bloom']
    X_test = test_data[predictor.feature_columns].fillna(0)
    y_test = test_data['bloom']
    
    # Train on past
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    
    scaler_temp = StandardScaler()
    X_train_scaled = scaler_temp.fit_transform(X_train)
    X_test_scaled = scaler_temp.transform(X_test)
    
    model_temp = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model_temp.fit(X_train_scaled, y_train)
    
    # Test on future
    y_test_pred = model_temp.predict(X_test_scaled)
    y_test_proba = model_temp.predict_proba(X_test_scaled)[:, 1]
    
    print("  " + "-" * 70)
    print(f"  Train period: {train_dates['date'].min().date()} to {train_dates['date'].max().date()}")
    print(f"  Test period:  {test_dates['date'].min().date()} to {test_dates['date'].max().date()}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples:  {len(test_data)}")
    print("  " + "-" * 70)
    print(f"  Test Accuracy:  {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"  Test Precision: {precision_score(y_test, y_test_pred):.3f}")
    print(f"  Test Recall:    {recall_score(y_test, y_test_pred):.3f}")
    print(f"  Test F1-Score:  {f1_score(y_test, y_test_pred):.3f}")
    print(f"  Test ROC-AUC:   {roc_auc_score(y_test, y_test_proba):.3f}")
    print("  " + "-" * 70)
    
    # Feature importance
    print("\n✓ Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': predictor.feature_columns,
        'importance': predictor.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("  " + "-" * 60)
    for idx, row in feature_importance.head(10).iterrows():
        bar_length = int(row['importance'] * 50)
        bar = '█' * bar_length
        print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")
    print("  " + "-" * 60)
    
    # Summary
    print("\n" + "=" * 80)
    print(" EVALUATION SUMMARY")
    print("=" * 80)
    
    overall_score = cv_scores['roc_auc'].mean()
    
    if overall_score >= 0.9:
        rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        assessment = "Model has outstanding predictive power!"
    elif overall_score >= 0.8:
        rating = "GOOD ⭐⭐⭐⭐"
        assessment = "Model performs well and is suitable for production."
    elif overall_score >= 0.7:
        rating = "FAIR ⭐⭐⭐"
        assessment = "Model has moderate predictive power. Consider improvements."
    else:
        rating = "POOR ⭐⭐"
        assessment = "Model needs significant improvement."
    
    print(f"\n  Overall Rating: {rating}")
    print(f"  Assessment: {assessment}")
    print(f"\n  Cross-Validation ROC-AUC: {overall_score:.3f}")
    print(f"  Full Model ROC-AUC: {roc_auc_score(y, y_pred_proba):.3f}")
    print(f"  Temporal Test ROC-AUC: {roc_auc_score(y_test, y_test_proba):.3f}")
    
    print("\n  Key Strengths:")
    if precision_score(y, y_pred) > 0.85:
        print(f"    ✓ High precision ({precision_score(y, y_pred):.1%}) - few false alarms")
    if recall_score(y, y_pred) > 0.85:
        print(f"    ✓ High recall ({recall_score(y, y_pred):.1%}) - catches most blooms")
    if false_positive_rate < 0.1:
        print(f"    ✓ Low false positive rate ({false_positive_rate:.1%})")
    if false_negative_rate < 0.1:
        print(f"    ✓ Low false negative rate ({false_negative_rate:.1%})")
    
    if precision_score(y, y_pred) < 0.7 or recall_score(y, y_pred) < 0.7:
        print("\n  Areas for Improvement:")
        if precision_score(y, y_pred) < 0.7:
            print(f"    ⚠ Precision could be better (currently {precision_score(y, y_pred):.1%})")
            print(f"      → Too many false alarms (predicting blooms that don't happen)")
        if recall_score(y, y_pred) < 0.7:
            print(f"    ⚠ Recall could be better (currently {recall_score(y, y_pred):.1%})")
            print(f"      → Missing some real blooms")
        print(f"    → Consider: more training data, better features, or tuning parameters")
    
    print("\n" + "=" * 80)
    print(" ✓ EVALUATION COMPLETE")
    print("=" * 80)
    print()
    
    return {
        'cv_scores': cv_scores,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'confusion_matrix': cm
    }


if __name__ == '__main__':
    try:
        results = evaluate_model()
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
