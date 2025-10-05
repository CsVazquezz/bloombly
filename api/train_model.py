#!/usr/bin/env python
"""
Training script for Bloom Prediction Model v2

This script allows you to:
1. Train the model with custom parameters
2. Save the trained model for later use
3. View training metrics and feature importance
"""

import sys
import os
import argparse
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from bloom_predictor_v2 import ImprovedBloomPredictor

def train_and_save_model(data_path='backend/data.csv', 
                         model_output='app/bloom_model_v2.pkl',
                         use_earth_engine=False,
                         n_estimators=200,
                         max_depth=5,
                         learning_rate=0.05):
    """
    Train bloom prediction model and save it
    
    Args:
        data_path: Path to historical bloom data CSV
        model_output: Where to save the trained model
        use_earth_engine: Whether to use Google Earth Engine for environmental data
        n_estimators: Number of trees in Gradient Boosting
        max_depth: Maximum depth of each tree
        learning_rate: Learning rate for Gradient Boosting
    """
    
    print("=" * 80)
    print(" TRAINING BLOOM PREDICTION MODEL v2")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data path: {data_path}")
    print(f"  Output model: {model_output}")
    print(f"  Use Earth Engine: {use_earth_engine}")
    print(f"  Model parameters:")
    print(f"    - n_estimators: {n_estimators}")
    print(f"    - max_depth: {max_depth}")
    print(f"    - learning_rate: {learning_rate}")
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Initialize and train
    print("\n[1/4] Initializing predictor...")
    predictor = ImprovedBloomPredictor(
        data_path=data_path,
        use_earth_engine=use_earth_engine
    )

    # Wait for the model to finish training in the background
    import time
    start_time = time.time()
    while predictor.is_training:
        print("Waiting for model to finish training...")
        time.sleep(5)
        if time.time() - start_time > 300: # 5 minute timeout
            print("Timeout waiting for model to train.")
            sys.exit(1)
    
    # Training happens automatically in __init__
    # But if you want to retrain with different parameters:
    if n_estimators != 200 or max_depth != 5 or learning_rate != 0.05:
        print("\n[2/4] Retraining with custom parameters...")
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.metrics import (accuracy_score, precision_score, 
                                     recall_score, f1_score, roc_auc_score)
        
        X = predictor.feature_data[predictor.feature_columns].copy()
        y = predictor.feature_data['bloom'].copy()
        X = X.fillna(0)
        
        X_scaled = predictor.scaler.transform(X)
        
        # Create new model with custom parameters
        predictor.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
            verbose=1
        )
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(predictor.model, X_scaled, y, 
                                   cv=tscv, scoring='roc_auc', n_jobs=-1)
        print(f"  Cross-val ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Train final model
        predictor.model.fit(X_scaled, y)
        
        # Evaluate
        y_pred = predictor.model.predict(X_scaled)
        y_pred_proba = predictor.model.predict_proba(X_scaled)[:, 1]
        
        print(f"\n  Model Performance:")
        print(f"    Accuracy:  {accuracy_score(y, y_pred):.3f}")
        print(f"    Precision: {precision_score(y, y_pred):.3f}")
        print(f"    Recall:    {recall_score(y, y_pred):.3f}")
        print(f"    F1-Score:  {f1_score(y, y_pred):.3f}")
        print(f"    ROC-AUC:   {roc_auc_score(y, y_pred_proba):.3f}")
    else:
        print("\n[2/4] Using default training (already completed)")
    
    # Feature importance
    print("\n[3/4] Analyzing feature importance...")
    import pandas as pd
    feature_importance = pd.DataFrame({
        'feature': predictor.feature_columns,
        'importance': predictor.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")
    
    # Save model
    print(f"\n[4/4] Saving model to {model_output}...")
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    predictor.save_model(model_output)
    
    print("\n" + "=" * 80)
    print(" ✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print(f"\nModel Statistics:")
    print(f"  Training data:")
    print(f"    - Positive examples (blooms): {len(predictor.historical_blooms)}")
    print(f"    - Negative examples (no-blooms): {len(predictor.negative_examples)}")
    print(f"    - Total samples: {len(predictor.feature_data)}")
    print(f"  Features: {len(predictor.feature_columns)}")
    print(f"  Species: {len(predictor.species_bloom_windows)}")
    
    print(f"\nTo use this model:")
    print(f"  1. The model is automatically loaded when the API starts")
    print(f"  2. Or load manually:")
    print(f"     predictor = ImprovedBloomPredictor()")
    print(f"     predictor.load_model('{model_output}')")
    
    return predictor


def main():
    parser = argparse.ArgumentParser(
        description='Train Bloom Prediction Model v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py
  python train_model.py --n_estimators 300 --max_depth 7
  python train_model.py --use_earth_engine
  python train_model.py --data custom_data.csv --output custom_model.pkl
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='../backend/data.csv',
        help='Path to bloom observation CSV file (default: ../backend/data.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='app/bloom_model_v2.pkl',
        help='Path where trained model will be saved (default: app/bloom_model_v2.pkl)'
    )
    
    parser.add_argument(
        '--use_earth_engine',
        action='store_true',
        help='Use Google Earth Engine for environmental data (requires authentication)'
    )
    
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=200,
        help='Number of boosting stages (trees) in the model (default: 200)'
    )
    
    parser.add_argument(
        '--max_depth',
        type=int,
        default=5,
        help='Maximum depth of each tree (default: 5)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.05,
        help='Learning rate for gradient boosting (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    try:
        predictor = train_and_save_model(
            data_path=args.data,
            model_output=args.output,
            use_earth_engine=args.use_earth_engine,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate
        )
        
        print(f"\n✓ Model saved successfully to: {args.output}")
        print(f"  File size: {os.path.getsize(args.output) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
