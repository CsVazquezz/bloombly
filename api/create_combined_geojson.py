#!/usr/bin/env python3
"""
Create Combined GeoJSON with both Kaggle forecasts and ML model predictions
Output: Single GeoJSON with color-coded features for globe.gl visualization
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))


class CombinedGeoJSONGenerator:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
    def load_kaggle_data(self, forecasts_path, places_path):
        """Load and merge Kaggle forecast data with location info"""
        print("\n[1/4] Loading Kaggle forecast data...")
        
        # Load forecasts
        forecasts = pd.read_csv(forecasts_path)
        # Load places with coordinates
        places = pd.read_csv(places_path)
        
        # Remove prefix from code in places (01370053 -> 1370053)
        places['code'] = places['code'].astype(str).str.lstrip('0').astype(int)
        
        # Merge
        kaggle = forecasts.merge(places, left_on='place_code', right_on='code', how='inner')
        
        # Filter for 2024 predictions (kaika_date is bloom start date)
        kaggle['bloom_date'] = pd.to_datetime(kaggle['kaika_date'])
        kaggle_2024 = kaggle[kaggle['bloom_date'].dt.year == 2024].copy()
        
        # Calculate day of year
        kaggle_2024['bloom_day'] = kaggle_2024['bloom_date'].dt.dayofyear
        
        print(f"  ✓ Loaded {len(kaggle_2024)} Kaggle forecasts for 2024")
        print(f"  Date range: {kaggle_2024['bloom_date'].min()} to {kaggle_2024['bloom_date'].max()}")
        
        return kaggle_2024[['spot_name', 'prefecture_en', 'lat', 'lon', 'bloom_date', 'bloom_day']]
    
    def load_ml_predictions(self, predictions_path):
        """Load ML model predictions from existing GeoJSON"""
        print("\n[2/4] Loading ML model predictions...")
        
        if not os.path.exists(predictions_path):
            print(f"  ✗ ML predictions not found at {predictions_path}")
            print(f"  Please run: python api/generate_predictions_geojson.py --skip-comparison")
            return None
        
        with open(predictions_path, 'r') as f:
            geojson = json.load(f)
        
        ml_data = []
        for feature in geojson['features']:
            props = feature['properties']
            coords = feature['geometry']['coordinates'][0][0]  # Get center of polygon
            
            # Calculate center point
            lons = [p[0] for p in coords]
            lats = [p[1] for p in coords]
            center_lon = sum(lons) / len(lons)
            center_lat = sum(lats) / len(lats)
            
            ml_data.append({
                'city': props['city'],
                'country': props['country'],
                'lat': center_lat,
                'lon': center_lon,
                'bloom_date': props['predicted_bloom_date'],
                'bloom_day': props['bloom_day_of_year']
            })
        
        ml_df = pd.DataFrame(ml_data)
        print(f"  ✓ Loaded {len(ml_df)} ML predictions")
        print(f"  Locations: {', '.join(ml_df['city'].head(5).tolist())}...")
        
        return ml_df
    
    def create_point_feature(self, lat, lon, name, bloom_date, bloom_day, source, 
                            country=None, prefecture=None):
        """Create a GeoJSON Point feature"""
        # Color coding
        colors = {
            'kaggle': '#0066FF',  # Blue
            'ml_model': '#FF0066'  # Red/Pink
        }
        
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)]
            },
            "properties": {
                "name": name,
                "country": country or "Japan",
                "prefecture": prefecture,
                "bloom_date": str(bloom_date),
                "bloom_day_of_year": int(bloom_day),
                "source": source,
                "color": colors[source],
                "size": 8 if source == 'ml_model' else 6,
                "label": f"{source.upper()}: {bloom_date}",
                "description": f"Predicted bloom: {bloom_date} (Day {bloom_day})"
            }
        }
    
    def create_combined_geojson(self, kaggle_df, ml_df, output_path):
        """Create combined GeoJSON with both datasets"""
        print("\n[3/4] Creating combined GeoJSON...")
        
        features = []
        
        # Add Kaggle forecasts (BLUE)
        if kaggle_df is not None:
            for _, row in kaggle_df.iterrows():
                feature = self.create_point_feature(
                    lat=row['lat'],
                    lon=row['lon'],
                    name=row['spot_name'],
                    bloom_date=row['bloom_date'].strftime('%Y-%m-%d'),
                    bloom_day=row['bloom_day'],
                    source='kaggle',
                    prefecture=row['prefecture_en']
                )
                features.append(feature)
            print(f"  ✓ Added {len(kaggle_df)} Kaggle forecasts (BLUE)")
        
        # Add ML model predictions (RED)
        if ml_df is not None:
            for _, row in ml_df.iterrows():
                feature = self.create_point_feature(
                    lat=row['lat'],
                    lon=row['lon'],
                    name=row['city'],
                    bloom_date=row['bloom_date'],
                    bloom_day=row['bloom_day'],
                    source='ml_model',
                    country=row['country']
                )
                features.append(feature)
            print(f"  ✓ Added {len(ml_df)} ML predictions (RED)")
        
        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "title": "Cherry Blossom Bloom Predictions 2024 - Comparison",
                "description": "Combined visualization of Kaggle forecasts (BLUE) and ML model predictions (RED)",
                "year": 2024,
                "generated": datetime.now().isoformat(),
                "sources": {
                    "kaggle": {
                        "count": len(kaggle_df) if kaggle_df is not None else 0,
                        "color": "#0066FF",
                        "description": "Kaggle cherry blossom forecasts"
                    },
                    "ml_model": {
                        "count": len(ml_df) if ml_df is not None else 0,
                        "color": "#FF0066",
                        "description": "ML model predictions (NDVI + environmental features)"
                    }
                },
                "total_features": len(features)
            },
            "features": features
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"  ✓ Combined GeoJSON saved to {output_path}")
        
        return geojson
    
    def create_summary_stats(self, kaggle_df, ml_df):
        """Create comparison statistics"""
        print("\n[4/4] Generating summary statistics...")
        
        stats = {
            "kaggle": {
                "count": len(kaggle_df) if kaggle_df is not None else 0,
                "earliest_bloom": kaggle_df['bloom_date'].min().strftime('%Y-%m-%d') if kaggle_df is not None and len(kaggle_df) > 0 else None,
                "latest_bloom": kaggle_df['bloom_date'].max().strftime('%Y-%m-%d') if kaggle_df is not None and len(kaggle_df) > 0 else None,
                "avg_bloom_day": float(kaggle_df['bloom_day'].mean()) if kaggle_df is not None and len(kaggle_df) > 0 else None
            },
            "ml_model": {
                "count": len(ml_df) if ml_df is not None else 0,
                "earliest_bloom": ml_df['bloom_date'].min() if ml_df is not None and len(ml_df) > 0 else None,
                "latest_bloom": ml_df['bloom_date'].max() if ml_df is not None and len(ml_df) > 0 else None,
                "avg_bloom_day": float(ml_df['bloom_day'].mean()) if ml_df is not None and len(ml_df) > 0 else None
            }
        }
        
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        
        if stats['kaggle']['count'] > 0:
            print(f"\n🔵 KAGGLE FORECASTS:")
            print(f"   Locations: {stats['kaggle']['count']}")
            print(f"   Earliest:  {stats['kaggle']['earliest_bloom']}")
            print(f"   Latest:    {stats['kaggle']['latest_bloom']}")
            print(f"   Avg day:   {stats['kaggle']['avg_bloom_day']:.1f}")
        
        if stats['ml_model']['count'] > 0:
            print(f"\n🔴 ML MODEL PREDICTIONS:")
            print(f"   Locations: {stats['ml_model']['count']}")
            print(f"   Earliest:  {stats['ml_model']['earliest_bloom']}")
            print(f"   Latest:    {stats['ml_model']['latest_bloom']}")
            print(f"   Avg day:   {stats['ml_model']['avg_bloom_day']:.1f}")
        
        print("\n" + "=" * 70)
        
        return stats


def main():
    print("=" * 70)
    print("CREATE COMBINED GEOJSON - KAGGLE + ML MODEL")
    print("=" * 70)
    
    generator = CombinedGeoJSONGenerator()
    script_dir = generator.script_dir
    
    # Paths
    forecasts_path = os.path.join(script_dir, '../data/raw/cherry_blossom_forecasts.csv')
    places_path = os.path.join(script_dir, '../data/raw/cherry_blossom_places.csv')
    ml_predictions_path = os.path.join(script_dir, '../data/geojson/ml_predictions_2024.geojson')
    output_path = os.path.join(script_dir, '../data/geojson/combined_predictions_2024.geojson')
    
    # Load data
    kaggle_df = generator.load_kaggle_data(forecasts_path, places_path)
    ml_df = generator.load_ml_predictions(ml_predictions_path)
    
    # Create combined GeoJSON
    geojson = generator.create_combined_geojson(kaggle_df, ml_df, output_path)
    
    # Generate stats
    stats = generator.create_summary_stats(kaggle_df, ml_df)
    
    print("\n" + "=" * 70)
    print("✓ COMBINED GEOJSON CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput file: {output_path}")
    print(f"Total predictions: {len(geojson['features'])}")
    print("\nVisualization colors:")
    print("  🔵 BLUE (#0066FF)  = Kaggle forecasts")
    print("  🔴 RED  (#FF0066)  = ML model predictions")
    print("\nLoad this GeoJSON in your globe.gl frontend!")


if __name__ == "__main__":
    main()
