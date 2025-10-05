#!/usr/bin/env python
"""Quick debug test for v2 model"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from datetime import datetime
from bloom_predictor_v2 import ImprovedBloomPredictor

print("Initializing v2 predictor...")
predictor = ImprovedBloomPredictor(use_earth_engine=False)

print("\nTesting predictions...")
aoi_bounds = {
    'min_lat': 40.0,
    'max_lat': 42.0,
    'min_lon': -75.0,
    'max_lon': -73.0
}

target_date = datetime(2024, 9, 15)

predictions = predictor.predict_blooms_for_date(
    target_date,
    aoi_bounds,
    num_predictions=10,
    confidence_threshold=0.3
)

print(f"\nResults:")
print(f"  Number of predictions: {len(predictions)}")
if predictions:
    print(f"  First prediction: {predictions[0]}")
else:
    print("  NO PREDICTIONS!")
    print("\n  Debugging info:")
    print(f"    Model trained: {predictor.model is not None}")
    print(f"    Training samples: {len(predictor.feature_data) if hasattr(predictor, 'feature_data') else 'N/A'}")
    print(f"    Species: {list(predictor.species_bloom_windows.keys()) if hasattr(predictor, 'species_bloom_windows') else 'N/A'}")
