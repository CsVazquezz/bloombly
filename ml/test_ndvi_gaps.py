#!/usr/bin/env python3
"""
Test script for NDVI retrieval with temporal gap-filling
Shows how the new features work without needing full dataset processing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from features import BloomFeatureEngineer
from datetime import datetime

def test_ndvi_temporal_fallback():
    """Test NDVI retrieval with temporal gap-filling"""
    
    print("\n" + "="*70)
    print(" NDVI TEMPORAL GAP-FILLING TEST")
    print("="*70)
    
    # Initialize feature engineer
    engineer = BloomFeatureEngineer()
    
    # Test location: Tokyo, Japan
    latitude = 35.6762
    longitude = 139.6503
    
    # Test different dates to show gap-filling
    test_dates = [
        datetime(2023, 3, 25),  # Recent - should have data
        datetime(2015, 4, 5),   # Mid-range - may need fallback
        datetime(2013, 3, 20),  # Early Landsat 8 - may need fallback
    ]
    
    for bloom_date in test_dates:
        print(f"\n{'='*70}")
        print(f"Testing: {bloom_date.strftime('%Y-%m-%d')}")
        print(f"Location: Tokyo ({latitude}, {longitude})")
        print(f"{'='*70}")
        
        # Test with GEE + temporal fallback (no AppEEARS needed)
        print("\n1️⃣  Testing GEE with temporal gap-filling...")
        ndvi_features = engineer.get_ndvi_with_temporal_fallback(
            latitude, longitude, bloom_date, use_appeears=False
        )
        
        print("\n   Results:")
        for key, value in ndvi_features.items():
            if value is not None:
                print(f"   ✓ {key}: {value:.4f}")
            else:
                print(f"   ✗ {key}: No data")
        
        # Test with AppEEARS if credentials available
        if engineer.appeears_username and engineer.appeears_password:
            print("\n2️⃣  Testing AppEEARS with temporal gap-filling...")
            ndvi_features_appeears = engineer.get_ndvi_with_temporal_fallback(
                latitude, longitude, bloom_date, use_appeears=True
            )
            
            print("\n   Results:")
            for key, value in ndvi_features_appeears.items():
                if value is not None:
                    print(f"   ✓ {key}: {value:.4f}")
                else:
                    print(f"   ✗ {key}: No data")
        else:
            print("\n2️⃣  AppEEARS credentials not found - skipping")
            print("   (Set APPEEARS_USERNAME and APPEEARS_PASSWORD in .env to test)")
    
    print("\n" + "="*70)
    print(" TEST COMPLETE")
    print("="*70)
    print("\nKey improvements:")
    print("  • Temporal gap-filling tries ±7, ±14, ±30 days if exact date fails")
    print("  • Works with both GEE and AppEEARS")
    print("  • Automatically falls back from AppEEARS → GEE → temporal gaps")
    print("  • Should dramatically improve NDVI data completeness!")
    print()

if __name__ == "__main__":
    test_ndvi_temporal_fallback()
