#!/usr/bin/env python3
"""
Quick retrain script to fix the feature_data issue.
This will reload the model, ensure feature_data is populated, and save it properly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.bloom_predictor_v2 import ImprovedBloomPredictor
import pickle

def main():
    print("=" * 60)
    print("RETRAINING BLOOM MODEL V2")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ImprovedBloomPredictor()
    
    print("\n1. Loading historical bloom data...")
    predictor.load_historical_blooms()
    print(f"   ✓ Loaded {len(predictor.historical_blooms)} bloom observations")
    
    print("\n2. Generating negative examples...")
    predictor.generate_negative_examples()
    print(f"   ✓ Generated {len(predictor.negative_examples)} negative examples")
    
    print("\n3. Building temporal features...")
    predictor.build_temporal_features()
    print(f"   ✓ Built features: {predictor.feature_data.shape}")
    print(f"   ✓ Feature columns: {len(predictor.feature_columns)}")
    print(f"   ✓ Features: {predictor.feature_columns[:5]}...")
    
    print("\n4. Training model...")
    predictor.train_model()
    print(f"   ✓ Model trained successfully")
    
    print("\n5. Saving model...")
    predictor.save_model()
    print(f"   ✓ Model saved to bloom_model_v2.pkl")
    
    # Verify the save worked
    print("\n6. Verifying saved model...")
    with open('bloom_model_v2.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    
    if 'feature_data' in saved_data:
        print(f"   ✓ feature_data saved: {saved_data['feature_data'].shape}")
    else:
        print(f"   ✗ feature_data NOT in saved file!")
        
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Restart the API server")
    print("2. Test with: curl http://localhost:5001/api/model-info")
    print()

if __name__ == "__main__":
    main()
