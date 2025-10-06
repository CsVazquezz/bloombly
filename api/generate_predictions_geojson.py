#!/usr/bin/env python3
"""
Generate GeoJSON predictions from trained model and compare with Kaggle forecasts

This script:
1. Loads trained Sakura model (.pkl)
2. Generates predictions for cherry blossom bloom dates
3. Exports predictions as GeoJSON
4. Compares with Kaggle forecast data (if available)
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import joblib

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))


class PredictionGenerator:
    def __init__(self, model_path='../app/models/sakura_global_model.pkl', 
                 data_path='../data/processed/bloom_features_ml.csv'):
        """Initialize with trained model"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if not os.path.isabs(model_path):
            self.model_path = os.path.join(script_dir, model_path)
        else:
            self.model_path = model_path
        
        if not os.path.isabs(data_path):
            self.data_path = os.path.join(script_dir, data_path)
        else:
            self.data_path = data_path
            
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.training_data = None
        
    def load_model(self):
        """Load trained model from pickle file"""
        print(f"\n[1/5] Loading model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_columns = self.model_data['feature_columns']
        
        print(f"  ✓ Model loaded successfully")
        print(f"  Features: {len(self.feature_columns)}")
        
        # Load training data for feature lookups
        print(f"\n  Loading training data from {self.data_path}...")
        if os.path.exists(self.data_path):
            self.training_data = pd.read_csv(self.data_path)
            print(f"  ✓ Loaded {len(self.training_data)} training records")
            print(f"  Date range: {self.training_data['year'].min()}-{self.training_data['year'].max()}")
        else:
            print(f"  ⚠ Training data not found, will use estimated values")
            self.training_data = None
        
    def load_locations(self, locations_file=None):
        """
        Load locations for prediction
        
        Args:
            locations_file: CSV file with columns: location_name, lat, lon, country
                           If None, uses default cherry blossom locations
        """
        print(f"\n[2/5] Loading prediction locations...")
        
        if locations_file and os.path.exists(locations_file):
            locations_df = pd.read_csv(locations_file)
            print(f"  ✓ Loaded {len(locations_df)} locations from {locations_file}")
        else:
            # Default locations: Famous cherry blossom spots
            locations_df = pd.DataFrame([
                # Japan
                {'location_name': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503, 'country': 'Japan'},
                {'location_name': 'Kyoto', 'lat': 35.0116, 'lon': 135.7681, 'country': 'Japan'},
                {'location_name': 'Osaka', 'lat': 34.6937, 'lon': 135.5023, 'country': 'Japan'},
                {'location_name': 'Hiroshima', 'lat': 34.3853, 'lon': 132.4553, 'country': 'Japan'},
                {'location_name': 'Fukuoka', 'lat': 33.5904, 'lon': 130.4017, 'country': 'Japan'},
                {'location_name': 'Sendai', 'lat': 38.2682, 'lon': 140.8694, 'country': 'Japan'},
                {'location_name': 'Nagoya', 'lat': 35.1815, 'lon': 136.9066, 'country': 'Japan'},
                {'location_name': 'Sapporo', 'lat': 43.0642, 'lon': 141.3469, 'country': 'Japan'},
                
                # USA
                {'location_name': 'Washington DC', 'lat': 38.9072, 'lon': -77.0369, 'country': 'USA'},
                {'location_name': 'Portland', 'lat': 45.5152, 'lon': -122.6784, 'country': 'USA'},
                {'location_name': 'Seattle', 'lat': 47.6062, 'lon': -122.3321, 'country': 'USA'},
                {'location_name': 'San Francisco', 'lat': 37.7749, 'lon': -122.4194, 'country': 'USA'},
                {'location_name': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'country': 'USA'},
                
                # South Korea
                {'location_name': 'Seoul', 'lat': 37.5665, 'lon': 126.9780, 'country': 'South Korea'},
                {'location_name': 'Busan', 'lat': 35.1796, 'lon': 129.0756, 'country': 'South Korea'},
                
                # China
                {'location_name': 'Beijing', 'lat': 39.9042, 'lon': 116.4074, 'country': 'China'},
                {'location_name': 'Shanghai', 'lat': 31.2304, 'lon': 121.4737, 'country': 'China'},
            ])
            print(f"  ✓ Using {len(locations_df)} default cherry blossom locations")
        
        return locations_df
    
    def create_feature_vector(self, lat, lon, year=2024):
        """
        Create feature vector for prediction using actual training data
        Finds the closest matching location and most recent year from training data
        """
        if self.training_data is not None:
            # Find closest location in training data
            self.training_data['distance'] = np.sqrt(
                (self.training_data['latitude'] - lat)**2 + 
                (self.training_data['longitude'] - lon)**2
            )
            
            # Get closest matches
            closest_matches = self.training_data.nsmallest(10, 'distance')
            
            # Prefer most recent year
            most_recent = closest_matches.sort_values('year', ascending=False).iloc[0]
            
            distance = most_recent['distance']
            
            print(f"    Using features from: {most_recent.get('region', 'Unknown')} "
                  f"({most_recent['year']}) - {distance:.2f}° away")
            
            # Create feature vector from actual data
            # First get fallback estimates
            fallback_features = self._create_estimated_features(lat, lon, year)
            
            features = {}
            for col in self.feature_columns:
                if col in most_recent.index:
                    value = most_recent[col]
                    # Use actual value if not NaN, otherwise use fallback estimate
                    if pd.notna(value):
                        features[col] = value
                    else:
                        features[col] = fallback_features.get(col, 0)
                else:
                    features[col] = fallback_features.get(col, 0)
            
            # Update year to prediction year
            features['year'] = year
            features['year_normalized'] = (year - 2010) / 12.0
            
            # Estimate day_of_year based on latitude (for display purposes)
            # The model will predict the actual bloom day
            if lat > 40:
                est_bloom_day = 120
            elif lat > 35:
                est_bloom_day = 90
            elif lat > 30:
                est_bloom_day = 75
            else:
                est_bloom_day = 45
            
            features['day_of_year'] = est_bloom_day
            features['month'] = (est_bloom_day // 30) + 1
            
            # Clean up temporary distance column
            if 'distance' in self.training_data.columns:
                self.training_data.drop('distance', axis=1, inplace=True)
            
        else:
            # Fallback to estimated values if no training data
            print(f"    Using estimated feature values (no training data)")
            features = self._create_estimated_features(lat, lon, year)
        
        # Create DataFrame
        feature_df = pd.DataFrame([features])
        
        # Ensure we have all required columns in correct order
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        feature_df = feature_df[self.feature_columns]
        
        return feature_df
    
    def _create_estimated_features(self, lat, lon, year=2024):
        """
        Fallback method: Create estimated feature vector when training data unavailable
        """
        # Estimate typical bloom month based on latitude
        if lat > 40:
            bloom_month = 5
            bloom_day = 120
        elif lat > 35:
            bloom_month = 4
            bloom_day = 90
        elif lat > 30:
            bloom_month = 3
            bloom_day = 75
        else:
            bloom_month = 2
            bloom_day = 45
        
        # Typical spring temperatures at bloom time
        temp_avg = 10.0 + (40 - abs(lat)) * 0.3  # Warmer closer to equator
        temp_min = temp_avg - 3
        temp_max = temp_avg + 5
        
        # Calculate realistic feature values
        features = {
            # Basic location/time
            'year': year,
            'year_normalized': (year - 2010) / 12.0,  # Normalize based on training range
            'day_of_year': bloom_day,
            'month': bloom_month,
            'decade': (year // 10) * 10,
            'latitude': lat,
            'longitude': lon,
            
            # Temperature features
            'temperature_avg': temp_avg,
            'temperature_min': temp_min,
            'temperature_max': temp_max,
            'precipitation': 50.0,
            'deviation_from_baseline': 0.0,
            
            # 7-day aggregates
            'temp_avg_7d': temp_avg - 1,
            'temp_max_7d': temp_max,
            'temp_min_7d': temp_min,
            'precip_total_7d': 15.0,
            'precip_avg_7d': 2.1,
            'humidity_avg_7d': 70.0,
            'solar_avg_7d': 8.0,
            'gdd_7d': max(0, (temp_avg - 5) * 7),
            
            # 14-day aggregates
            'temp_avg_14d': temp_avg - 2,
            'temp_max_14d': temp_max,
            'temp_min_14d': temp_min - 1,
            'precip_total_14d': 30.0,
            'precip_avg_14d': 2.1,
            'humidity_avg_14d': 70.0,
            'solar_avg_14d': 7.5,
            'gdd_14d': max(0, (temp_avg - 2 - 5) * 14),
            
            # 30-day aggregates
            'temp_avg_30d': temp_avg - 3,
            'temp_max_30d': temp_max - 1,
            'temp_min_30d': temp_min - 2,
            'precip_total_30d': 60.0,
            'precip_avg_30d': 2.0,
            'humidity_avg_30d': 68.0,
            'solar_avg_30d': 7.0,
            'gdd_30d': max(0, (temp_avg - 3 - 5) * 30),
            
            # 90-day aggregates
            'temp_avg_90d': temp_avg - 5,
            'temp_max_90d': temp_max - 2,
            'temp_min_90d': temp_min - 3,
            'precip_total_90d': 200.0,
            'precip_avg_90d': 2.2,
            'humidity_avg_90d': 65.0,
            'solar_avg_90d': 6.5,
            'gdd_90d': max(0, (temp_avg - 5 - 5) * 90),
            'temp_variance_90d': 25.0,
            'temp_range_90d': 15.0,
            'frost_days_90d': max(0, 10 - (temp_avg - 5)),
            
            # Soil properties (typical values for cherry growing regions)
            'soil_ph_0-5cm': 6.5,
            'soil_ph_5-15cm': 6.5,
            'soil_clay_0-5cm': 28.0,
            'soil_clay_5-15cm': 30.0,
            'soil_sand_0-5cm': 42.0,
            'soil_sand_5-15cm': 42.0,
            'soil_organic_carbon_0-5cm': 2.0,
            'soil_organic_carbon_5-15cm': 1.8,
            
            # NDVI - Vegetation greenness (CRITICAL for bloom prediction!)
            # Spring greenup typically shows NDVI around 0.3-0.5
            'ndvi_mean_5d': 0.35 + (temp_avg - 8) * 0.02,  # Higher NDVI with warmer temps
            'ndvi_mean_10d': 0.35 + (temp_avg - 8) * 0.02,  # Similar to 5d
            
            # Elevation (approximate based on latitude)
            'elevation_m': 50.0 if abs(lat - 35) < 5 else 100.0,
            
            # Photoperiod (day length) - critical for phenology
            # Spring equinox is around 12 hours, increasing
            'photoperiod_at_bloom': 12.0 + (bloom_day - 80) * 0.03,  # Increasing day length
            'photoperiod_30d_before': 12.0 + (bloom_day - 110) * 0.03,
            'photoperiod_change_rate': 0.05,  # Minutes per day increase
        }
        
        # Create DataFrame with all required features
        feature_df = pd.DataFrame([features])
        
        # Ensure correct column order
        feature_df = feature_df[self.feature_columns]
        
        return feature_df
    
    def predict_bloom_date(self, lat, lon, year=2024):
        """
        Predict bloom day of year for a location
        
        Returns:
            day_of_year: Predicted bloom day (1-365)
        """
        # Create feature vector
        features = self.create_feature_vector(lat, lon, year)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        day_of_year = self.model.predict(features_scaled)[0]
        
        return int(round(day_of_year))
    
    def day_of_year_to_date(self, day_of_year, year=2024):
        """Convert day of year to date"""
        return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    
    def generate_predictions(self, locations_df, year=2024):
        """Generate bloom predictions for all locations"""
        print(f"\n[3/5] Generating predictions for year {year}...")
        
        predictions = []
        
        for idx, row in locations_df.iterrows():
            try:
                # Predict bloom day
                bloom_day = self.predict_bloom_date(row['lat'], row['lon'], year)
                bloom_date = self.day_of_year_to_date(bloom_day, year)
                
                predictions.append({
                    'location_name': row['location_name'],
                    'country': row['country'],
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'year': year,
                    'bloom_day_of_year': bloom_day,
                    'predicted_bloom_date': bloom_date.strftime('%Y-%m-%d'),
                    'model': 'bloombly_sakura_v1'
                })
                
                print(f"  {row['location_name']:20s} → {bloom_date.strftime('%B %d, %Y')} (Day {bloom_day})")
                
            except Exception as e:
                print(f"  ✗ Error predicting for {row['location_name']}: {e}")
        
        predictions_df = pd.DataFrame(predictions)
        print(f"  ✓ Generated {len(predictions_df)} predictions")
        
        return predictions_df
    
    def create_geojson(self, predictions_df, output_file='ml_predictions_2024.geojson'):
        """Convert predictions to GeoJSON format"""
        print(f"\n[4/5] Creating GeoJSON...")
        
        features = []
        
        for idx, row in predictions_df.iterrows():
            # Create polygon around location (small area)
            lon, lat = row['lon'], row['lat']
            offset = 0.05  # ~5km radius
            
            feature = {
                "type": "Feature",
                "id": f"ml_{row['location_name'].lower().replace(' ', '_')}",
                "properties": {
                    "source": "ml_model",
                    "location_name": row['location_name'],
                    "city": row['location_name'],
                    "country": row['country'],
                    "species": "Prunus serrulata",
                    "predicted_bloom_date": row['predicted_bloom_date'],
                    "bloom_day_of_year": int(row['bloom_day_of_year']),
                    "model_version": row['model'],
                    "forecast_made_on": datetime.now().strftime('%Y-%m-%d'),
                    "color": "#FF0066",  # Red for ML predictions
                    "opacity": 0.6,
                    "display_label": f"ML: {row['predicted_bloom_date']}"
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
                "source": "bloombly_ml_model",
                "description": f"ML model predictions for {predictions_df['year'].iloc[0]} cherry blossom bloom dates",
                "model_version": "sakura_v1",
                "generated_on": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "locations": len(features),
                "year": int(predictions_df['year'].iloc[0])
            },
            "features": features
        }
        
        # Save to file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, '..', 'data', 'geojson', output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"  ✓ GeoJSON saved to {output_path}")
        
        return geojson
    
    def compare_with_kaggle(self, predictions_df, kaggle_file=None):
        """Compare ML predictions with Kaggle forecasts"""
        print(f"\n[5/5] Comparing with Kaggle forecasts...")
        
        if not kaggle_file:
            kaggle_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..',
                'data',
                'raw',
                'kaggle_cherry_blossom_2024.csv'
            )
        
        if not os.path.exists(kaggle_file):
            print(f"  ⚠ Kaggle file not found: {kaggle_file}")
            print(f"  Skipping comparison. To compare:")
            print(f"  1. Download Kaggle cherry blossom forecast CSV")
            print(f"  2. Save to: {kaggle_file}")
            print(f"  3. Re-run this script")
            return None
        
        # Load Kaggle data
        kaggle_df = pd.read_csv(kaggle_file)
        print(f"  ✓ Loaded {len(kaggle_df)} Kaggle forecasts")
        
        # Merge predictions with Kaggle data
        comparison = predictions_df.merge(
            kaggle_df,
            left_on='location_name',
            right_on='location',  # Adjust column name as needed
            how='inner',
            suffixes=('_ml', '_kaggle')
        )
        
        if len(comparison) == 0:
            print(f"  ⚠ No matching locations found between ML and Kaggle data")
            return None
        
        # Calculate differences
        comparison['kaggle_date'] = pd.to_datetime(comparison['bloom_date'])  # Adjust column name
        comparison['ml_date'] = pd.to_datetime(comparison['predicted_bloom_date'])
        comparison['difference_days'] = (comparison['ml_date'] - comparison['kaggle_date']).dt.days
        comparison['absolute_error'] = comparison['difference_days'].abs()
        comparison['within_3_days'] = comparison['absolute_error'] <= 3
        comparison['within_7_days'] = comparison['absolute_error'] <= 7
        
        # Print comparison results
        print(f"\n  📊 COMPARISON RESULTS")
        print(f"  {'='*70}")
        print(f"  Locations compared: {len(comparison)}")
        print(f"  Mean Absolute Error: {comparison['absolute_error'].mean():.1f} days")
        print(f"  Median Absolute Error: {comparison['absolute_error'].median():.0f} days")
        print(f"  Accuracy within 3 days: {comparison['within_3_days'].mean():.1%}")
        print(f"  Accuracy within 7 days: {comparison['within_7_days'].mean():.1%}")
        print(f"  Best match: {comparison['absolute_error'].min():.0f} days")
        print(f"  Worst match: {comparison['absolute_error'].max():.0f} days")
        print(f"  {'='*70}")
        
        # Print detailed comparison
        print(f"\n  Detailed Comparison:")
        print(f"  {'-'*70}")
        for idx, row in comparison.iterrows():
            emoji = "✅" if row['within_3_days'] else ("⚠️" if row['within_7_days'] else "❌")
            print(f"  {emoji} {row['location_name']:20s} | Kaggle: {row['kaggle_date'].strftime('%b %d')} | ML: {row['ml_date'].strftime('%b %d')} | Diff: {row['difference_days']:+3.0f} days")
        print(f"  {'-'*70}")
        
        # Save comparison CSV
        comparison_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'data',
            'processed',
            'comparison_kaggle_vs_ml_2024.csv'
        )
        
        comparison[['location_name', 'country', 'kaggle_date', 'ml_date', 
                   'difference_days', 'absolute_error', 'within_3_days', 'within_7_days']].to_csv(
            comparison_file, index=False
        )
        
        print(f"\n  ✓ Comparison saved to {comparison_file}")
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description='Generate GeoJSON predictions from trained model')
    parser.add_argument('--model', default='../app/models/sakura_global_model.pkl',
                       help='Path to trained model (.pkl file)')
    parser.add_argument('--locations', default=None,
                       help='CSV file with locations (location_name, lat, lon, country)')
    parser.add_argument('--year', type=int, default=2024,
                       help='Year to predict (default: 2024)')
    parser.add_argument('--output', default='ml_predictions_2024.geojson',
                       help='Output GeoJSON filename')
    parser.add_argument('--kaggle', default=None,
                       help='Kaggle forecast CSV file for comparison')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='Skip Kaggle comparison')
    
    args = parser.parse_args()
    
    print("="*80)
    print(" GENERATE ML PREDICTIONS GEOJSON")
    print("="*80)
    
    try:
        # Initialize generator
        generator = PredictionGenerator(model_path=args.model)
        
        # Load model
        generator.load_model()
        
        # Load locations
        locations_df = generator.load_locations(args.locations)
        
        # Generate predictions
        predictions_df = generator.generate_predictions(locations_df, year=args.year)
        
        # Create GeoJSON
        geojson = generator.create_geojson(predictions_df, output_file=args.output)
        
        # Compare with Kaggle (if not skipped)
        if not args.skip_comparison:
            comparison = generator.compare_with_kaggle(predictions_df, kaggle_file=args.kaggle)
        
        print("\n" + "="*80)
        print(" ✓ PREDICTION GENERATION COMPLETE")
        print("="*80)
        print(f"\n  Generated predictions for {len(predictions_df)} locations")
        print(f"  GeoJSON file: data/geojson/{args.output}")
        if not args.skip_comparison:
            print(f"  Comparison CSV: data/processed/comparison_kaggle_vs_ml_2024.csv")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
