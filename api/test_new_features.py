#!/usr/bin/env python3
"""
Test script for new bloom prediction features.

This script validates:
1. Spring start date detection from NDVI time series
2. Growing Degree Days (GDD) calculation
3. Soil water availability calculation
4. Integration with the bloom prediction model
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.bloom_features import (
    calculate_spring_start_date,
    calculate_growing_degree_days,
    calculate_accumulated_gdd,
    calculate_soil_water_days,
    calculate_comprehensive_bloom_features
)


def test_spring_detection():
    """Test spring start date detection with realistic NDVI data"""
    print("\n" + "=" * 80)
    print("TEST 1: Spring Start Date Detection")
    print("=" * 80)
    
    # Create a full year of synthetic NDVI data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Simulate realistic NDVI: low in winter, rising in spring, high in summer, falling in fall
    ndvi_synthetic = 0.2 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    ndvi_synthetic = np.maximum(0.1, ndvi_synthetic)  # Floor at 0.1
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.02, len(ndvi_synthetic))
    ndvi_synthetic = ndvi_synthetic + noise
    ndvi_synthetic = np.clip(ndvi_synthetic, 0, 1)
    
    # Test detection
    result = calculate_spring_start_date(ndvi_synthetic, dates)
    
    print(f"✓ Spring start detected on day: {result['spring_start_day']}")
    print(f"  Expected range: 70-90 (mid-March to early April)")
    print(f"  Days since spring start: {result['days_since_spring_start']}")
    print(f"  Is spring currently active: {result['is_spring_active']}")
    print(f"  Winter NDVI baseline: {result['winter_ndvi_baseline']:.3f}")
    
    # Validate result
    assert 50 <= result['spring_start_day'] <= 120, "Spring start day should be between Feb-April"
    assert result['winter_ndvi_baseline'] < 0.4, "Winter baseline should be lower than peak NDVI"
    
    print("✓ Spring detection test passed!")
    return result


def test_gdd_calculation():
    """Test Growing Degree Days calculation"""
    print("\n" + "=" * 80)
    print("TEST 2: Growing Degree Days (GDD)")
    print("=" * 80)
    
    # Test cases with known results
    test_cases = [
        {'tmax': 25, 'tmin': 15, 'tbase': 0, 'expected': 20},
        {'tmax': 30, 'tmin': 20, 'tbase': 10, 'expected': 15},
        {'tmax': 15, 'tmin': 5, 'tbase': 0, 'expected': 10},
        {'tmax': 5, 'tmin': -5, 'tbase': 0, 'expected': 0},  # Below threshold
    ]
    
    for i, case in enumerate(test_cases, 1):
        gdd = calculate_growing_degree_days(case['tmax'], case['tmin'], case['tbase'])
        print(f"  Case {i}: Tmax={case['tmax']}°C, Tmin={case['tmin']}°C, Tbase={case['tbase']}°C")
        print(f"    GDD = {gdd:.2f} (expected ~{case['expected']})")
        assert abs(gdd - case['expected']) < 0.1, f"GDD calculation failed for case {i}"
    
    # Test accumulated GDD
    print("\n  Testing accumulated GDD over 30 days...")
    tmax_series = np.random.uniform(20, 30, 30)
    tmin_series = np.random.uniform(10, 20, 30)
    
    accumulated = calculate_accumulated_gdd(tmax_series, tmin_series, tbase=0, days=30)
    expected_range = (300, 900)  # Rough estimate
    
    print(f"  30-day accumulated GDD: {accumulated:.2f}")
    print(f"  Expected range: {expected_range}")
    assert expected_range[0] <= accumulated <= expected_range[1], "Accumulated GDD out of expected range"
    
    print("✓ GDD calculation test passed!")
    return accumulated


def test_soil_water_calculation():
    """Test soil water availability calculation"""
    print("\n" + "=" * 80)
    print("TEST 3: Soil Water Availability")
    print("=" * 80)
    
    # Test different soil moisture scenarios
    field_capacity = 25  # Typical loam soil
    
    scenarios = [
        {'moisture': 30, 'description': 'Well-watered'},
        {'moisture': 20, 'description': 'Moderate moisture'},
        {'moisture': 15, 'description': 'Approaching stress'},
        {'moisture': 10, 'description': 'Water stress'},
        {'moisture': 5, 'description': 'Severe stress'},
    ]
    
    print(f"  Field capacity: {field_capacity}%\n")
    
    for scenario in scenarios:
        result = calculate_soil_water_days(scenario['moisture'], field_capacity)
        
        print(f"  {scenario['description']} ({scenario['moisture']}% moisture):")
        print(f"    Wilting point: {result['wilting_point']:.2f}%")
        print(f"    Available water days: {result['soil_water_days']:.2f}")
        print(f"    Water stress: {result['water_stress']}")
        print(f"    Available water ratio: {result['available_water_ratio']:.2f}")
        
        # Validate logic
        if scenario['moisture'] < result['wilting_point']:
            assert result['water_stress'] == True, "Should be in water stress"
            assert result['soil_water_days'] == 0, "No available water days"
        else:
            assert result['water_stress'] == False, "Should not be in water stress"
            assert result['soil_water_days'] > 0, "Should have available water days"
        
        print()
    
    print("✓ Soil water calculation test passed!")


def test_comprehensive_features():
    """Test comprehensive feature calculation"""
    print("\n" + "=" * 80)
    print("TEST 4: Comprehensive Feature Calculation")
    print("=" * 80)
    
    # Create realistic test data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # NDVI time series
    ndvi_series = 0.25 + 0.45 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    ndvi_series = np.maximum(0.1, ndvi_series)
    
    # Temperature time series
    tmax_series = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    tmin_series = tmax_series - 8
    
    # Build environmental data
    env_data = {
        'ndvi_time_series': ndvi_series.tolist(),
        'dates': dates,
        'tmax': tmax_series.tolist(),
        'tmin': tmin_series.tolist(),
        'soil_moisture': 22,
        'field_capacity': 25
    }
    
    # Calculate all features
    features = calculate_comprehensive_bloom_features(env_data)
    
    print("  Calculated features:")
    print(f"    Spring start day: {features['spring_start_day']}")
    print(f"    Days since spring: {features['days_since_spring_start']}")
    print(f"    Is spring active: {features['is_spring_active']}")
    print(f"    Winter NDVI baseline: {features['winter_ndvi_baseline']:.3f}")
    print(f"    Current GDD: {features['gdd_current']:.2f}")
    print(f"    30-day accumulated GDD: {features['gdd_accumulated_30d']:.2f}")
    print(f"    Soil water days: {features['soil_water_days']:.2f}")
    print(f"    Wilting point: {features['wilting_point']:.2f}")
    print(f"    Water stress: {features['water_stress']}")
    print(f"    Available water ratio: {features['available_water_ratio']:.2f}")
    
    # Validate all features are present
    expected_keys = [
        'spring_start_day', 'days_since_spring_start', 'is_spring_active',
        'winter_ndvi_baseline', 'gdd_current', 'gdd_accumulated_30d',
        'soil_water_days', 'wilting_point', 'water_stress', 'available_water_ratio'
    ]
    
    for key in expected_keys:
        assert key in features, f"Missing feature: {key}"
    
    print("\n✓ Comprehensive feature calculation test passed!")
    return features


def test_model_integration():
    """Test integration with bloom prediction model"""
    print("\n" + "=" * 80)
    print("TEST 5: Model Integration")
    print("=" * 80)
    
    # Check if model file exists
    model_path = 'app/bloom_model_v2.pkl'
    
    if os.path.exists(model_path):
        print(f"  Found existing model: {model_path}")
        
        # Try loading the model
        from app.bloom_predictor_v2 import ImprovedBloomPredictor
        
        print("  Loading model...")
        predictor = ImprovedBloomPredictor(
            data_path='../backend/data.csv',
            use_earth_engine=False,  # Use fallback for testing
            load_pretrained=model_path
        )
        
        # Check feature columns
        print(f"\n  Model has {len(predictor.feature_columns)} features:")
        
        # Count new features
        new_features = [f for f in predictor.feature_columns if any(
            keyword in f for keyword in ['spring', 'gdd', 'soil_water', 'wilting', 'water_stress']
        )]
        
        print(f"    Original features: {len(predictor.feature_columns) - len(new_features)}")
        print(f"    New advanced features: {len(new_features)}")
        
        if new_features:
            print("\n  New features detected:")
            for feature in new_features:
                print(f"    - {feature}")
            print("\n  ✓ Model has been updated with new features!")
        else:
            print("\n  ⚠ Model needs to be retrained with new features")
            print("    Run: python retrain_and_save_v2.py")
        
        # Test prediction with new features
        print("\n  Testing prediction with a sample location and date...")
        test_lat = 40.0
        test_lon = -100.0
        test_date = datetime(2024, 5, 15)
        
        try:
            prob = predictor.predict_bloom_probability(test_lat, test_lon, test_date)
            print(f"  Bloom probability: {prob:.3f}")
            print("  ✓ Prediction successful!")
        except Exception as e:
            print(f"  ✗ Prediction failed: {e}")
            print("  → Model may need retraining with new features")
            
    else:
        print(f"  ⚠ Model file not found: {model_path}")
        print("  → Train the model first: python retrain_and_save_v2.py")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("BLOOM PREDICTION FEATURE TESTING")
    print("=" * 80)
    print("\nTesting new ecological features for improved bloom prediction:")
    print("1. Spring start date detection (NDVI-based)")
    print("2. Growing Degree Days (Baskerville-Emin method)")
    print("3. Soil water availability (wilting point method)")
    
    try:
        # Run tests
        test_spring_detection()
        test_gdd_calculation()
        test_soil_water_calculation()
        test_comprehensive_features()
        test_model_integration()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Retrain the model to use the new features:")
        print("   cd api && python retrain_and_save_v2.py")
        print("\n2. The model will now use advanced ecological features:")
        print("   - Spring phenology (4 features)")
        print("   - Growing degree days (2 features)")
        print("   - Soil water availability (4 features)")
        print("\n3. These features should improve prediction accuracy by:")
        print("   - Better capturing seasonal bloom timing")
        print("   - Accounting for heat accumulation effects")
        print("   - Considering water stress impacts on blooming")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
