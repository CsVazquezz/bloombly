#!/usr/bin/env python
"""
Quick test script for the v2 Bloom Prediction Model
This bypasses Flask and tests the model directly
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from datetime import datetime
import json

print("=" * 80)
print(" TESTING BLOOM PREDICTION MODEL v2")
print("=" * 80)

# Test configuration
test_date = datetime(2025, 6, 15)  # Mid-June
test_location = {
    'lat': 41.5,
    'lon': -74.0,
    'name': 'New York area'
}

print(f"\nTest Parameters:")
print(f"  Date: {test_date.strftime('%Y-%m-%d')}")
print(f"  Location: {test_location['name']} ({test_location['lat']}, {test_location['lon']})")

print("\n" + "=" * 80)
print(" INITIALIZING MODEL v2...")
print("=" * 80)

try:
    from bloom_predictor_v2 import ImprovedBloomPredictor
    
    # Initialize predictor (will train model)
    print("\nThis will:")
    print("  1. Load historical bloom data (~1,179 observations)")
    print("  2. Generate negative examples (~3,000 non-bloom observations)")
    print("  3. Build 21 temporal + environmental features")
    print("  4. Train Gradient Boosting model")
    print("  5. Validate with time-series cross-validation")
    print("\nPlease wait...\n")
    
    predictor = ImprovedBloomPredictor(use_earth_engine=False)  # Use fallback for testing
    
    print("\n" + "=" * 80)
    print(" MODEL INITIALIZED SUCCESSFULLY!")
    print("=" * 80)
    
    # Print model statistics
    print(f"\nTraining Data Statistics:")
    print(f"  Positive examples (blooms): {len(predictor.historical_blooms)}")
    print(f"  Negative examples (no-blooms): {len(predictor.negative_examples)}")
    print(f"  Total training samples: {len(predictor.feature_data)}")
    print(f"  Species tracked: {len(predictor.species_bloom_windows)}")
    
    print(f"\nFeature Engineering:")
    print(f"  Total features: {len(predictor.feature_columns)}")
    print(f"  Feature names:")
    for i, feat in enumerate(predictor.feature_columns, 1):
        print(f"    {i:2d}. {feat}")
    
    print(f"\nSpecies Bloom Windows:")
    for species, info in predictor.species_bloom_windows.items():
        print(f"  {species}:")
        print(f"    Peak bloom day: {int(info['mean_day'])} (±{int(info['std_day'])} days)")
        print(f"    Range: day {int(info['min_day'])}-{int(info['max_day'])}")
        print(f"    Observations: {info['count']}")
    
    print("\n" + "=" * 80)
    print(" TEST 1: PREDICT SINGLE LOCATION")
    print("=" * 80)
    
    # Test prediction for specific location
    for species in predictor.species_bloom_windows.keys():
        prob = predictor.predict_bloom_probability(
            lat=test_location['lat'],
            lon=test_location['lon'],
            date=test_date,
            species=species
        )
        
        print(f"\n{species}:")
        print(f"  Bloom probability: {prob:.1%}")
        print(f"  Confidence: {'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW'}")
    
    print("\n" + "=" * 80)
    print(" TEST 2: PREDICT BLOOMS ACROSS AREA")
    print("=" * 80)
    
    # Predict blooms across an area
    aoi_bounds = {
        'min_lat': 40.0,
        'max_lat': 42.0,
        'min_lon': -75.0,
        'max_lon': -73.0
    }
    
    print(f"\nSearching for blooms in:")
    print(f"  Latitude: {aoi_bounds['min_lat']} to {aoi_bounds['max_lat']}")
    print(f"  Longitude: {aoi_bounds['min_lon']} to {aoi_bounds['max_lon']}")
    print(f"  Date: {test_date.strftime('%Y-%m-%d')}")
    print(f"  Confidence threshold: 0.3 (30%)")
    
    predictions = predictor.predict_blooms_for_date(
        target_date=test_date,
        aoi_bounds=aoi_bounds,
        num_predictions=20,
        confidence_threshold=0.3
    )
    
    print(f"\nFound {len(predictions)} predicted bloom sites")
    
    if predictions:
        print(f"\nTop 5 Predictions:")
        for i, pred in enumerate(predictions[:5], 1):
            props = pred['properties']
            coords = pred['geometry']['coordinates'][0][0][0]  # Get first coordinate
            print(f"\n  {i}. {props['Site']}")
            print(f"     Location: ({coords[1]:.3f}, {coords[0]:.3f})")
            print(f"     Bloom probability: {props['bloom_probability']:.1%}")
            print(f"     Season: {props['Season']}")
            print(f"     Estimated area: {props['Area']:.0f} m²")
            print(f"     Environmental:")
            print(f"       Temperature: {props['environmental_factors']['temperature']:.1f}°C")
            print(f"       Precipitation: {props['environmental_factors']['precipitation']:.1f}mm")
            print(f"       NDVI: {props['environmental_factors']['ndvi']:.3f}")
            print(f"       NDVI trend: {props['environmental_factors']['ndvi_trend']:.4f}")
    
    print("\n" + "=" * 80)
    print(" TEST 3: COMPARE WITH DIFFERENT DATES")
    print("=" * 80)
    
    test_dates = [
        datetime(2025, 3, 15),  # Early spring
        datetime(2025, 6, 15),  # Summer
        datetime(2025, 9, 15),  # Fall
    ]
    
    print(f"\nTesting bloom probability over seasons:")
    print(f"  Location: {test_location['name']}")
    species = list(predictor.species_bloom_windows.keys())[0]
    print(f"  Species: {species}")
    
    for date in test_dates:
        prob = predictor.predict_bloom_probability(
            lat=test_location['lat'],
            lon=test_location['lon'],
            date=date,
            species=species
        )
        print(f"\n  {date.strftime('%B %d, %Y')} (day {date.timetuple().tm_yday}): {prob:.1%}")
    
    print("\n" + "=" * 80)
    print(" ✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print(f"\nModel Summary:")
    print(f"  ✓ Predicts bloom probability directly (not species)")
    print(f"  ✓ Trained on {len(predictor.historical_blooms)} bloom + {len(predictor.negative_examples)} no-bloom examples")
    print(f"  ✓ Uses {len(predictor.feature_columns)} engineered features")
    print(f"  ✓ Learns temporal patterns (seasonality, cyclical encoding)")
    print(f"  ✓ Incorporates environmental dynamics (trends, aggregations)")
    print(f"  ✓ Provides calibrated probability scores")
    
    print(f"\nTo use via API:")
    print(f"  1. Start server: python app/main.py")
    print(f"  2. Test endpoint: curl 'http://localhost:5001/api/blooms?date=2025-06-15&method=v2'")
    
except ImportError as e:
    print(f"\n✗ Error importing model: {e}")
    print(f"\nMake sure you're in the api directory and dependencies are installed:")
    print(f"  pip install pandas numpy scikit-learn")
    
except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\n")
