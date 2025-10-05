#!/usr/bin/env python3
"""
Quick test script for sakura models

Tests both training and prediction functionality
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_training():
    """Test model training"""
    print("=" * 80)
    print("TEST 1: Model Training")
    print("=" * 80)
    
    from train_sakura_model import SakuraModelTrainer
    
    try:
        trainer = SakuraModelTrainer(data_path='../data/processed/bloom_features_ml.csv')
        trainer.load_data()
        
        print("\n✓ Data loaded successfully")
        print(f"  Records: {len(trainer.data)}")
        
        # Train with minimal parameters for testing
        print("\nTraining models (this may take a few minutes)...")
        trainer.train_global_model(n_estimators=50, max_depth=5)
        trainer.train_japan_model(n_estimators=50, max_depth=5)
        
        # Save to test directory
        test_dir = 'test_models'
        trainer.save_models(output_dir=test_dir)
        
        print("\n✓ Model training test passed!")
        return test_dir
        
    except Exception as e:
        print(f"\n✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prediction(models_dir):
    """Test model predictions"""
    print("\n" + "=" * 80)
    print("TEST 2: Model Predictions")
    print("=" * 80)
    
    from app.sakura_predictor import SakuraBloomPredictor
    
    try:
        # Initialize predictor
        predictor = SakuraBloomPredictor(models_dir=models_dir)
        
        print(f"\n✓ Models loaded successfully")
        
        # Test prediction for Tokyo
        print(f"\nTesting prediction for Tokyo...")
        prediction = predictor.predict_bloom_date(
            latitude=35.6762,
            longitude=139.6503,
            year=2025,
            species="Prunus × yedoensis"
        )
        
        print(f"\n  Prediction Results:")
        print(f"    Location: Tokyo (35.68°N, 139.65°E)")
        print(f"    Year: {prediction['year']}")
        print(f"    Bloom date: {prediction['bloom_date']}")
        print(f"    Day of year: {prediction['bloom_day_of_year']}")
        print(f"    Model used: {prediction['model_used']}")
        print(f"    Confidence: {prediction['confidence']:.2%}")
        
        # Test with window
        print(f"\nTesting prediction with window...")
        window_pred = predictor.predict_bloom_window(
            latitude=35.6762,
            longitude=139.6503,
            year=2025,
            species="Prunus × yedoensis"
        )
        
        if 'bloom_window' in window_pred:
            window = window_pred['bloom_window']
            print(f"\n  Bloom Window:")
            print(f"    Early date: {window['early_date']}")
            print(f"    Peak date: {window['peak_date']}")
            print(f"    Late date: {window['late_date']}")
            print(f"    Window size: {window['window_days']} days")
        
        # Test batch prediction
        print(f"\nTesting batch prediction...")
        locations = [
            {'latitude': 35.68, 'longitude': 139.65, 'name': 'Tokyo'},
            {'latitude': 34.69, 'longitude': 135.50, 'name': 'Osaka'},
            {'latitude': 43.06, 'longitude': 141.35, 'name': 'Sapporo'}
        ]
        
        batch_results = predictor.batch_predict(locations, year=2025)
        
        print(f"\n  Batch Results ({len(batch_results)} locations):")
        for result in batch_results:
            if 'bloom_date' in result:
                print(f"    {result['location_name']}: {result['bloom_date']} (Day {result['bloom_day_of_year']})")
            else:
                print(f"    {result['location_name']}: Error - {result.get('error', 'Unknown')}")
        
        print(f"\n✓ Prediction tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_info(models_dir):
    """Test model information retrieval"""
    print("\n" + "=" * 80)
    print("TEST 3: Model Information")
    print("=" * 80)
    
    from app.sakura_predictor import SakuraBloomPredictor
    
    try:
        predictor = SakuraBloomPredictor(models_dir=models_dir)
        
        # Get model info
        info = predictor.get_model_info()
        
        print(f"\n  Models loaded: {', '.join(info['models_loaded'])}")
        print(f"  Feature count: {info['feature_count']}")
        print(f"\n  First 10 features:")
        for i, feature in enumerate(info['features'][:10], 1):
            print(f"    {i}. {feature}")
        
        # Get feature importance (if available)
        if 'japan' in info['models_loaded']:
            print(f"\n  Top 10 Japan Model Features:")
            importance = predictor.get_feature_importance('japan')
            for idx, row in importance.head(10).iterrows():
                print(f"    {row['feature']:30s} {row['importance']:.4f}")
        
        print(f"\n✓ Model info test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Model info test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_models(test_dir):
    """Clean up test models"""
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"\n✓ Cleaned up test directory: {test_dir}")


def main():
    print("\n" + "=" * 80)
    print(" SAKURA MODEL TESTING SUITE")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Test model training")
    print("  2. Test predictions")
    print("  3. Test model information retrieval")
    print("\nNote: This uses a small subset of estimators for quick testing")
    
    # Run tests
    models_dir = test_training()
    
    if models_dir:
        test_prediction(models_dir)
        test_model_info(models_dir)
        
        # Cleanup
        print("\n" + "=" * 80)
        response = input("\nDelete test models? (y/n): ")
        if response.lower() == 'y':
            cleanup_test_models(models_dir)
    
    print("\n" + "=" * 80)
    print(" TESTING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Train production models: python train_sakura_model.py")
    print("  2. Evaluate models: python evaluate_sakura_models.py")
    print("  3. Start API server with sakura endpoints")


if __name__ == '__main__':
    main()
