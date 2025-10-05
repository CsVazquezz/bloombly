#!/usr/bin/env python3
"""
Convert Kaggle cherry blossom forecast CSV to GeoJSON format

This script converts Kaggle forecast data to GeoJSON so you can
display it on the map alongside ML predictions.
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime

def csv_to_geojson(csv_file, output_file='kaggle_blooms_2024.geojson', year=2024):
    """
    Convert Kaggle CSV to GeoJSON
    
    Expected CSV columns:
    - location or city: Location name
    - lat, latitude: Latitude
    - lon, longitude: Longitude
    - bloom_date or predicted_date: Bloom date
    - country (optional): Country name
    """
    print(f"\n📁 Converting Kaggle CSV to GeoJSON...")
    print(f"  Input: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    print(f"  ✓ Loaded {len(df)} records")
    
    # Normalize column names (handle different formats)
    column_mapping = {
        'city': 'location_name',
        'location': 'location_name',
        'latitude': 'lat',
        'longitude': 'lon',
        'predicted_date': 'bloom_date',
        'forecast_date': 'bloom_date'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Required columns
    required = ['location_name', 'lat', 'lon', 'bloom_date']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add country if not present
    if 'country' not in df.columns:
        df['country'] = 'Unknown'
    
    # Parse dates
    df['bloom_date'] = pd.to_datetime(df['bloom_date'])
    df['bloom_day_of_year'] = df['bloom_date'].dt.dayofyear
    
    # Create GeoJSON features
    features = []
    
    for idx, row in df.iterrows():
        lon, lat = row['lon'], row['lat']
        offset = 0.05  # ~5km radius
        
        feature = {
            "type": "Feature",
            "id": f"kaggle_{row['location_name'].lower().replace(' ', '_')}",
            "properties": {
                "source": "kaggle_forecast",
                "location_name": row['location_name'],
                "city": row['location_name'],
                "country": row['country'],
                "species": "Prunus serrulata",
                "predicted_bloom_date": row['bloom_date'].strftime('%Y-%m-%d'),
                "bloom_day_of_year": int(row['bloom_day_of_year']),
                "forecast_made_on": f"{year}-01-01",
                "color": "#0066FF",  # Blue for Kaggle
                "opacity": 0.6,
                "display_label": f"Kaggle: {row['bloom_date'].strftime('%b %d')}"
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[[
                    [lon - offset, lat - offset],
                    [lon + offset, lat - offset],
                    [lon + offset, lat + offset],
                    [lon - offset, lat + offset],
                    [lon - offset, lat - offset]
                ]]]
            }
        }
        
        features.append(feature)
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "source": "kaggle_cherry_blossom_forecast",
            "description": f"Kaggle predictions for {year} cherry blossom bloom dates",
            "locations": len(features),
            "year": year,
            "generated_on": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        "features": features
    }
    
    # Save GeoJSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, '..', 'data', 'geojson', output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"  ✓ GeoJSON saved to {output_path}")
    print(f"  ✓ Created {len(features)} features")
    
    return geojson


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Kaggle CSV to GeoJSON')
    parser.add_argument('csv_file', help='Path to Kaggle CSV file')
    parser.add_argument('--output', default='kaggle_blooms_2024.geojson',
                       help='Output GeoJSON filename')
    parser.add_argument('--year', type=int, default=2024,
                       help='Year of forecast (default: 2024)')
    
    args = parser.parse_args()
    
    try:
        csv_to_geojson(args.csv_file, args.output, args.year)
        print("\n✓ Conversion complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
