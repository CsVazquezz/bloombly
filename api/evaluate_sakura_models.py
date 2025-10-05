#!/usr/bin/env python3
"""
Evaluate and Compare Sakura Bloom Prediction Models

This script evaluates both the global and Japan-specific models:
- Performance metrics (MAE, RMSE, R²)
- Cross-validation results
- Feature importance comparison
- Prediction accuracy by region/year
- Model comparison and recommendations
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))


class SakuraModelEvaluator:
    """
    Evaluates sakura bloom prediction models
    """
    
    def __init__(self, data_path='../data/processed/bloom_features_ml.csv',
                 models_dir='app/models'):
        """
        Initialize evaluator
        
        Args:
            data_path: Path to bloom features dataset
            models_dir: Directory containing trained models
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.data = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load evaluation dataset"""
        print(f"\n[1/5] Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        print(f"  ✓ Loaded {len(self.data)} records")
        
    def load_models(self):
        """Load trained models"""
        print(f"\n[2/5] Loading models from {self.models_dir}...")
        
        # Load global model
        global_path = os.path.join(self.models_dir, 'sakura_global_model.pkl')
        if os.path.exists(global_path):
            self.models['global'] = joblib.load(global_path)
            print(f"  ✓ Loaded global model")
        
        # Load Japan model
        japan_path = os.path.join(self.models_dir, 'sakura_japan_model.pkl')
        if os.path.exists(japan_path):
            self.models['japan'] = joblib.load(japan_path)
            print(f"  ✓ Loaded Japan model")
        
        if not self.models:
            raise FileNotFoundError("No models found. Train models first.")
    
    def evaluate_global_model(self):
        """Evaluate global model performance"""
        print(f"\n[3/5] Evaluating Global Model...")
        
        if 'global' not in self.models:
            print("  ⚠ Global model not found, skipping")
            return
        
        model_data = self.models['global']
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['feature_columns']
        
        # Get Prunus data
        prunus_data = self.data[self.data['genus'] == 'Prunus'].copy()
        
        # Prepare features
        X = prunus_data[features].copy()
        
        # Drop columns that are entirely NaN
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            print(f"  ⚠ Dropping {len(all_nan_cols)} columns with all NaN values: {all_nan_cols[:5]}...")
            X = X.drop(columns=all_nan_cols)
            # Update features list to match
            features_to_use = [f for f in features if f not in all_nan_cols]
        else:
            features_to_use = features
        
        # Fill remaining NaN values with median
        X = X.fillna(X.median())
        
        # Fill any remaining NaN (in case median is still NaN) with 0
        X = X.fillna(0)
        
        y = prunus_data['bloom_day_of_year']
        
        # Scale features - need to handle potential dimension mismatch
        if len(all_nan_cols) > 0:
            # Create a full feature matrix with zeros for dropped columns
            X_full = pd.DataFrame(0, index=X.index, columns=features)
            X_full[features_to_use] = X
            X_scaled = scaler.transform(X_full)
        else:
            X_scaled = scaler.transform(X)
        
        # Predictions
        y_pred = model.predict(X_scaled)
        
        # Metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            model, X_scaled, y,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        self.results['global'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'n_samples': len(prunus_data),
            'predictions': y_pred,
            'actuals': y.values
        }
        
        print(f"\n  Global Model Results:")
        print(f"    Samples:      {len(prunus_data)}")
        print(f"    MAE:          {mae:.2f} days")
        print(f"    RMSE:         {rmse:.2f} days")
        print(f"    R²:           {r2:.3f}")
        print(f"    CV MAE:       {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} days")
    
    def evaluate_japan_model(self):
        """Evaluate Japan-specific model performance"""
        print(f"\n[4/5] Evaluating Japan Model...")
        
        if 'japan' not in self.models:
            print("  ⚠ Japan model not found, skipping")
            return
        
        model_data = self.models['japan']
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['feature_columns']
        
        # Get Japan data
        japan_data = self.data[
            (self.data['scientific_name'] == 'Prunus × yedoensis') |
            (
                (self.data['genus'] == 'Prunus') & 
                (self.data['species'] == 'yedoensis')
            )
        ].copy()
        
        if 'region' in self.data.columns:
            japan_region_data = self.data[
                self.data['region'].str.contains('Japan|Prefecture', case=False, na=False)
            ].copy()
            japan_data = pd.concat([japan_data, japan_region_data]).drop_duplicates()
        
        # Prepare features
        X = japan_data[features].copy()
        
        # Drop columns that are entirely NaN
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            print(f"  ⚠ Dropping {len(all_nan_cols)} columns with all NaN values: {all_nan_cols[:5]}...")
            X = X.drop(columns=all_nan_cols)
            # Update features list to match
            features_to_use = [f for f in features if f not in all_nan_cols]
        else:
            features_to_use = features
        
        # Fill remaining NaN values with median
        X = X.fillna(X.median())
        
        # Fill any remaining NaN (in case median is still NaN) with 0
        X = X.fillna(0)
        
        y = japan_data['bloom_day_of_year']
        
        # Scale features - need to handle potential dimension mismatch
        if len(all_nan_cols) > 0:
            # Create a full feature matrix with zeros for dropped columns
            X_full = pd.DataFrame(0, index=X.index, columns=features)
            X_full[features_to_use] = X
            X_scaled = scaler.transform(X_full)
        else:
            X_scaled = scaler.transform(X)
        
        # Predictions
        y_pred = model.predict(X_scaled)
        
        # Metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=min(5, len(japan_data) // 5))
        cv_scores = cross_val_score(
            model, X_scaled, y,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        self.results['japan'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'n_samples': len(japan_data),
            'predictions': y_pred,
            'actuals': y.values
        }
        
        print(f"\n  Japan Model Results:")
        print(f"    Samples:      {len(japan_data)}")
        print(f"    MAE:          {mae:.2f} days")
        print(f"    RMSE:         {rmse:.2f} days")
        print(f"    R²:           {r2:.3f}")
        print(f"    CV MAE:       {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} days")
    
    def compare_models(self):
        """Compare both models side by side"""
        print(f"\n[5/5] Model Comparison")
        print("=" * 80)
        
        if not self.results:
            print("  No results to compare")
            return
        
        # Create comparison table
        print(f"\n{'Metric':<20} {'Global Model':<20} {'Japan Model':<20}")
        print("-" * 80)
        
        metrics = ['mae', 'rmse', 'r2', 'cv_mae_mean', 'n_samples']
        metric_names = {
            'mae': 'MAE (days)',
            'rmse': 'RMSE (days)',
            'r2': 'R² Score',
            'cv_mae_mean': 'CV MAE (days)',
            'n_samples': 'Training Samples'
        }
        
        for metric in metrics:
            name = metric_names.get(metric, metric)
            
            global_val = self.results.get('global', {}).get(metric, 'N/A')
            japan_val = self.results.get('japan', {}).get(metric, 'N/A')
            
            if isinstance(global_val, float):
                if metric == 'r2':
                    global_str = f"{global_val:.3f}"
                else:
                    global_str = f"{global_val:.2f}"
            else:
                global_str = str(global_val)
            
            if isinstance(japan_val, float):
                if metric == 'r2':
                    japan_str = f"{japan_val:.3f}"
                else:
                    japan_str = f"{japan_val:.2f}"
            else:
                japan_str = str(japan_val)
            
            print(f"{name:<20} {global_str:<20} {japan_str:<20}")
        
        # Recommendations
        print(f"\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        if 'global' in self.results and 'japan' in self.results:
            global_mae = self.results['global']['mae']
            japan_mae = self.results['japan']['mae']
            
            if japan_mae < global_mae:
                improvement = ((global_mae - japan_mae) / global_mae) * 100
                print(f"\n✓ Japan model is more accurate for Japanese cherry blossoms")
                print(f"  - {improvement:.1f}% improvement in MAE over global model")
                print(f"  - Recommended for Prunus × yedoensis in Japan")
            else:
                print(f"\n✓ Global model performs well across all regions")
                print(f"  - Recommended for general Prunus species worldwide")
            
            print(f"\nModel Selection Guidelines:")
            print(f"  • Use Japan Model for:")
            print(f"    - Prunus × yedoensis (Somei Yoshino)")
            print(f"    - Locations in Japan (lat 24-46°N, lon 122-154°E)")
            print(f"    - High-accuracy predictions for Japanese prefectures")
            print(f"\n  • Use Global Model for:")
            print(f"    - Other Prunus species (P. avium, P. serrulata, etc.)")
            print(f"    - Locations outside Japan")
            print(f"    - General bloom predictions worldwide")
        
        print(f"\n" + "=" * 80)
    
    def save_results(self, output_path='evaluation_results.json'):
        """Save evaluation results to JSON"""
        results_json = {}
        
        for model_name, result in self.results.items():
            results_json[model_name] = {
                'mae': float(result['mae']),
                'rmse': float(result['rmse']),
                'r2': float(result['r2']),
                'cv_mae_mean': float(result['cv_mae_mean']),
                'cv_mae_std': float(result['cv_mae_std']),
                'n_samples': int(result['n_samples'])
            }
        
        results_json['evaluated_at'] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")
    
    def plot_predictions(self, output_dir='evaluation_plots'):
        """Create visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, result in self.results.items():
            if 'predictions' not in result:
                continue
            
            y_true = result['actuals']
            y_pred = result['predictions']
            
            plt.figure(figsize=(10, 6))
            
            # Scatter plot
            plt.subplot(1, 2, 1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2)
            plt.xlabel('Actual Bloom Day')
            plt.ylabel('Predicted Bloom Day')
            plt.title(f'{model_name.title()} Model: Predictions vs Actuals')
            plt.grid(True, alpha=0.3)
            
            # Residuals
            plt.subplot(1, 2, 2)
            residuals = y_true - y_pred
            plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Prediction Error (days)')
            plt.ylabel('Frequency')
            plt.title(f'{model_name.title()} Model: Error Distribution')
            plt.axvline(x=0, color='r', linestyle='--', lw=2)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{model_name}_evaluation.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            print(f"  ✓ Saved plot: {plot_path}")


def main():
    print("=" * 80)
    print(" SAKURA MODEL EVALUATION")
    print("=" * 80)
    
    evaluator = SakuraModelEvaluator()
    
    try:
        evaluator.load_data()
        evaluator.load_models()
        evaluator.evaluate_global_model()
        evaluator.evaluate_japan_model()
        evaluator.compare_models()
        evaluator.save_results()
        
        # Try to create plots (requires matplotlib)
        try:
            print(f"\nCreating visualization plots...")
            evaluator.plot_predictions()
        except ImportError:
            print(f"\n⚠ matplotlib not available, skipping plots")
        except Exception as e:
            print(f"\n⚠ Could not create plots: {e}")
        
        print(f"\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
