#!/usr/bin/env python3
"""
Script to retrain the v2 model with the corrected save/load logic.
This ensures feature_data is properly saved in the model file.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.bloom_predictor_v2 import ImprovedBloomPredictor

def main():
    print("=" * 60)
    print("Retraining Bloom Predictor v2 Model")
    print("=" * 60)
    
    # Train new model (will save with feature_data included)
    predictor = ImprovedBloomPredictor(
        data_path='../backend/data.csv',
        use_earth_engine=False,  # Use fallback for faster training
        load_pretrained=None  # Don't load, train fresh
    )
    
    # Wait for background training to complete
    # (In production, you'd want better synchronization)
    print("\nWaiting for training to complete...")
    import time
    while predictor.is_training:
        time.sleep(2)
        print("  Still training...")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Verify the model was saved correctly
    if os.path.exists('bloom_model_v2.pkl'):
        print(f"✓ Model file saved: bloom_model_v2.pkl")
        
        # Test loading
        print("\nTesting model load...")
        test_predictor = ImprovedBloomPredictor(
            data_path='../backend/data.csv',
            use_earth_engine=False,
            load_pretrained='bloom_model_v2.pkl'
        )
        
        # Check if feature_data was loaded
        if hasattr(test_predictor, 'feature_data') and test_predictor.feature_data is not None:
            print(f"✓ feature_data loaded successfully: {len(test_predictor.feature_data)} samples")
        else:
            print("✗ Warning: feature_data not loaded properly")
        
        print(f"✓ Model info:")
        print(f"  - Features: {len(test_predictor.feature_columns)}")
        print(f"  - Bloom observations: {len(test_predictor.historical_blooms)}")
        print(f"  - Negative examples: {len(test_predictor.negative_examples)}")
        print(f"  - Species: {len(test_predictor.species_bloom_windows)}")
    else:
        print("✗ Model file not found!")

if __name__ == '__main__':
    main()
