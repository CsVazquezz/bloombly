#!/usr/bin/env python3
"""
Training script for Sakura Bloom Prediction Models
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))


class SakuraModelTrainer:
    def __init__(self, data_path='../data/processed/bloom_features_ml.csv'):
        self.data_path = data_path
        self.data = None
        self.global_model = None
        self.japan_model = None
        self.global_scaler = None
        self.japan_scaler = None
        self.feature_columns = []
        
    def load_data(self):
        print(f"\n[1/6] Loading data from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        print(f"  Loaded {len(self.data)} records")
        print(f"  Date range: {self.data['year'].min()} - {self.data['year'].max()}")
        print(f"  Unique species: {self.data['scientific_name'].nunique()}")
        
    def prepare_features(self, df):
        exclude_cols = [
            'record_id', 'scientific_name', 'family', 'genus', 'species', 
            'common_name', 'region', 'prefecture', 'location_grid',
            'trait', 'is_prediction', 'bloom_date', 'bloom_day_of_year'
        ]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"  Selected {len(self.feature_columns)} features")
        
        X = df[self.feature_columns].copy()
        # Replace inf with nan, then fill with median
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median().fillna(0))
        
        y = df['bloom_day_of_year'].copy()
        # Remove rows where target is NaN
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"  Using {len(X)} samples with valid target values")
        
        return X, y
    
    def train_global_model(self, n_estimators=300, max_depth=10, learning_rate=0.05):
        print(f"\n[2/6] Training Global Model...")
        
        prunus_data = self.data[self.data['genus'] == 'Prunus'].copy()
        print(f"  Using {len(prunus_data)} Prunus records")
        
        if len(prunus_data) < 10:
            print("  Warning: Very few records for global model")
            return
        
        X, y = self.prepare_features(prunus_data)
        
        self.global_scaler = StandardScaler()
        X_scaled = self.global_scaler.fit_transform(X)
        
        self.global_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        self.global_model.fit(X_scaled, y)
        
        y_pred = self.global_model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print(f"  MAE: {mae:.2f} days, RMSE: {rmse:.2f} days, R²: {r2:.3f}")
        
    def train_japan_model(self, n_estimators=300, max_depth=10, learning_rate=0.05):
        print(f"\n[3/6] Training Japan Model...")
        
        japan_data = self.data[
            (self.data['scientific_name'] == 'Prunus × yedoensis') |
            ((self.data['genus'] == 'Prunus') & (self.data['species'] == 'yedoensis'))
        ].copy()
        
        print(f"  Using {len(japan_data)} Japanese cherry records")
        
        if len(japan_data) < 5:
            print("  Warning: Very few records for Japan model")
            return
        
        X, y = self.prepare_features(japan_data)
        
        self.japan_scaler = StandardScaler()
        X_scaled = self.japan_scaler.fit_transform(X)
        
        self.japan_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        self.japan_model.fit(X_scaled, y)
        
        y_pred = self.japan_model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print(f"  MAE: {mae:.2f} days, RMSE: {rmse:.2f} days, R²: {r2:.3f}")
    
    def save_models(self, output_dir='app/models'):
        print(f"\n[4/6] Saving models to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.global_model is not None:
            global_path = os.path.join(output_dir, 'sakura_global_model.pkl')
            joblib.dump({
                'model': self.global_model,
                'scaler': self.global_scaler,
                'feature_columns': self.feature_columns,
                'metadata': {
                    'trained_date': datetime.now().isoformat(),
                    'model_type': 'GradientBoostingRegressor',
                    'target': 'bloom_day_of_year'
                }
            }, global_path)
            print(f"  Saved global model: {global_path}")
        
        if self.japan_model is not None:
            japan_path = os.path.join(output_dir, 'sakura_japan_model.pkl')
            joblib.dump({
                'model': self.japan_model,
                'scaler': self.japan_scaler,
                'feature_columns': self.feature_columns,
                'metadata': {
                    'trained_date': datetime.now().isoformat(),
                    'model_type': 'GradientBoostingRegressor',
                    'target': 'bloom_day_of_year'
                }
            }, japan_path)
            print(f"  Saved Japan model: {japan_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Sakura Models')
    parser.add_argument('--data', type=str, default='../data/processed/bloom_features_ml.csv')
    parser.add_argument('--output-dir', type=str, default='app/models')
    parser.add_argument('--n_estimators', type=int, default=300)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--global-only', action='store_true')
    parser.add_argument('--japan-only', action='store_true')
    
    args = parser.parse_args()
    
    try:
        print("=" * 80)
        print(" SAKURA BLOOM PREDICTION MODEL TRAINING")
        print("=" * 80)
        
        trainer = SakuraModelTrainer(data_path=args.data)
        trainer.load_data()
        
        if not args.japan_only:
            trainer.train_global_model(args.n_estimators, args.max_depth, args.learning_rate)
        
        if not args.global_only:
            trainer.train_japan_model(args.n_estimators, args.max_depth, args.learning_rate)
        
        trainer.save_models(output_dir=args.output_dir)
        
        print(f"\n[6/6] Training completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
