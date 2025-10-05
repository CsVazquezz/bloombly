#!/usr/bin/env python3
"""
Debug script to test bloom prediction with the new 31-feature model
"""

import sys
import os
from datetime import datetime

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from bloom_predictor_v2 import ImprovedBloomPredictor

def main():
    print("=" * 70)
    print("Testing Bloom Prediction v2 with 31 Features")
    print("=" * 70)
    
    # Load the model
    model_path = 'app/bloom_model_v2.pkl'
    print(f"\n1. Loading model from: {model_path}")
    predictor = ImprovedBloomPredictor(load_pretrained=model_path, use_earth_engine=False)
    
    # Check model info
    print(f"\n2. Model Info:")
    print(f"   - Features: {len(predictor.feature_columns)} features")
    print(f"   - Feature names: {predictor.feature_columns[:5]}... (showing first 5)")
    print(f"   - Species: {len(predictor.species_bloom_windows)}")
    print(f"   - Historical blooms: {len(predictor.historical_blooms)}")
    print(f"   - Scaler expects: {predictor.scaler.n_features_in_} features")
    
    # Test prediction for Queretaro, Mexico
    # Queretaro approximate bounds
    aoi_bounds = {
        'min_lat': 20.0,
        'max_lat': 21.5,
        'min_lon': -100.5,
        'max_lon': -99.0
    }
    
    target_date = datetime(2024, 4, 15)
    
    print(f"\n3. Testing prediction:")
    print(f"   - Date: {target_date}")
    print(f"   - AOI: Queretaro, Mexico {aoi_bounds}")
    print(f"   - Confidence threshold: 0.3")
    print(f"   - Number of predictions requested: 200")
    
    print("\n" + "=" * 70)
    predictions = predictor.predict_blooms_for_date(
        target_date=target_date,
        aoi_bounds=aoi_bounds,
        num_predictions=200,
        confidence_threshold=0.3
    )
    print("=" * 70)
    
    print(f"\n4. Results:")
    print(f"   - Total predictions: {len(predictions)}")
    
    if predictions:
        print(f"\n5. Sample predictions (top 5):")
        for i, pred in enumerate(predictions[:5], 1):
            props = pred['properties']
            coords = pred['geometry']['coordinates'][0][0][0]  # Get first coordinate
            print(f"   {i}. {props['Site']}")
            print(f"      Probability: {props['bloom_probability']}")
            print(f"      Location: ({coords[1]:.3f}, {coords[0]:.3f})")
            print(f"      Temperature: {props['environmental_factors']['temperature']}°C")
            print(f"      NDVI: {props['environmental_factors']['ndvi']}")
    else:
        print("\n⚠ No predictions generated!")
        print("\nPossible reasons:")
        print("  1. All probabilities are below the confidence threshold (0.3)")
        print("  2. Species bloom windows don't match the target date")
        print("  3. Environmental conditions are unfavorable")
        
        # Test a single prediction to see what probabilities we're getting
        print("\n6. Testing single location prediction:")
        test_lat, test_lon = 20.5, -100.0
        for species in list(predictor.species_bloom_windows.keys())[:3]:
            prob = predictor.predict_bloom_probability(test_lat, test_lon, target_date, species)
            bloom_window = predictor.species_bloom_windows[species]
            print(f"   - {species[:30]:30s}: {prob:.4f} (bloom window: {bloom_window['mean_day']} ± {bloom_window['mean_day'] - bloom_window['min_day']} days)")

if __name__ == '__main__':
    main()
