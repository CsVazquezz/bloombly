import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import ee
from collections import defaultdict
import requests
import json
from google.oauth2 import service_account

class EnhancedBloomPredictor:
    """
    Enhanced bloom predictor that incorporates environmental factors
    for more accurate future predictions
    """
    def __init__(self, data_path='../backend/data.csv', use_earth_engine=True):
        self.data_path = data_path
        self.use_earth_engine = use_earth_engine
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.species_patterns = {}
        self.environmental_cache = {}

        # Check if Earth Engine is available (already initialized elsewhere)
        if self.use_earth_engine:
            try:
                # Test if Earth Engine is already initialized
                ee.Number(1).getInfo()
                print("Earth Engine is available and initialized")
            except:
                # Try to initialize Earth Engine ourselves
                try:
                    # Try with default credentials first (for development)
                    ee.Initialize()
                    print("Earth Engine initialized with default credentials")
                except:
                    # If that fails, try with service account from environment
                    try:
                        import os
                        import json

                        credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
                        project_id = os.getenv('EE_PROJECT')

                        if credentials_json and project_id:
                            credentials_dict = json.loads(credentials_json)
                            SCOPES = [
                                'https://www.googleapis.com/auth/earthengine',
                                'https://www.googleapis.com/auth/cloud-platform',
                                'https://www.googleapis.com/auth/userinfo.email'
                            ]

                            credentials = service_account.Credentials.from_service_account_info(
                                credentials_dict,
                                scopes=SCOPES
                            )

                            ee.Initialize(credentials=credentials, project=project_id)
                            print("Earth Engine initialized with service account credentials")
                        else:
                            raise Exception("No credentials available")
                    except:
                        print("Earth Engine not available, using fallback methods")
                        self.use_earth_engine = False

        self.load_and_process_data()
        self.build_environmental_features()
        self.train_model()

    def load_and_process_data(self):
        """Load and process historical bloom data"""
        try:
            # Load CSV data
            df = pd.read_csv(self.data_path)

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_year'] = df['date'].dt.dayofyear
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month

            # Extract location and species info
            df['lat'] = df['latitude'].astype(float)
            df['lon'] = df['longitude'].astype(float)

            # Create bloom indicator (1 if flowering observed)
            df['bloom'] = 1

            # Add state information for regional analysis
            df['state'] = self.get_state_from_coords(df['lat'], df['lon'])

            # Store processed data
            self.historical_data = df

            # Analyze patterns
            self.analyze_patterns()

        except Exception as e:
            print(f"Error loading data: {e}")
            self.historical_data = pd.DataFrame()

    def get_state_from_coords(self, lat, lon):
        """Simple state assignment based on coordinates (US only)"""
        states = []

        for la, lo in zip(lat, lon):
            if 25 <= la <= 32 and -88 <= lo <= -80:
                states.append('Florida')
            elif 32 <= la <= 42 and -125 <= lo <= -114:
                states.append('California')
            elif 25 <= la <= 37 and -107 <= lo <= -93:
                states.append('Texas')
            elif 40 <= la <= 45 and -80 <= lo <= -71:
                states.append('New York')
            else:
                states.append('Other')

        return states

    def analyze_patterns(self):
        """Analyze historical patterns for prediction"""
        if self.historical_data.empty:
            return

        # Species-specific patterns
        for species in self.historical_data['scientificName'].unique():
            species_data = self.historical_data[self.historical_data['scientificName'] == species]
            self.species_patterns[species] = {
                'mean_day': species_data['day_of_year'].mean(),
                'std_day': species_data['day_of_year'].std(),
                'lat_range': (species_data['lat'].min(), species_data['lat'].max()),
                'lon_range': (species_data['lon'].min(), species_data['lon'].max()),
                'count': len(species_data),
                'states': species_data['state'].value_counts().to_dict()
            }

    def get_environmental_data_ee(self, lat, lon, date):
        """Get environmental data using Google Earth Engine"""
        try:
            point = ee.Geometry.Point([lon, lat])

            # Temperature data (MODIS LST)
            lst = ee.ImageCollection('MODIS/061/MOD11A1') \
                .filterDate(date - timedelta(days=7), date + timedelta(days=1)) \
                .select('LST_Day_1km') \
                .mean()

            # Precipitation (CHIRPS)
            precip = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                .filterDate(date - timedelta(days=30), date) \
                .sum()

            # NDVI (vegetation index)
            ndvi = ee.ImageCollection('MODIS/061/MOD13Q1') \
                .filterDate(date - timedelta(days=30), date) \
                .select('NDVI') \
                .mean()

            # Elevation
            elevation = ee.Image('USGS/SRTMGL1_003')

            # Get values at point
            values = {
                'temperature': lst.reduceRegion(ee.Reducer.mean(), point, 1000).get('LST_Day_1km'),
                'precipitation': precip.reduceRegion(ee.Reducer.mean(), point, 1000).get('precipitation'),
                'ndvi': ndvi.reduceRegion(ee.Reducer.mean(), point, 1000).get('NDVI'),
                'elevation': elevation.reduceRegion(ee.Reducer.mean(), point, 1000).get('elevation')
            }

            # Convert to Python values
            result = {}
            for key, value in values.items():
                try:
                    result[key] = float(ee.Number(value).getInfo())
                except:
                    result[key] = 0.0

            # Convert temperature from Kelvin to Celsius
            if 'temperature' in result and result['temperature'] > 0:
                result['temperature'] = result['temperature'] * 0.02 - 273.15

            return result

        except Exception as e:
            print(f"Error getting EE environmental data: {e}")
            return self.get_environmental_data_fallback(lat, lon, date)

    def get_environmental_data_fallback(self, lat, lon, date):
        """Fallback environmental data when EE is not available"""
        # Use simplified climate normals based on location and season
        month = date.month
        day_of_year = date.timetuple().tm_yday

        # Base temperature by latitude and season
        base_temp = 15 - abs(lat) * 0.3  # Cooler at higher latitudes

        # Seasonal temperature variation
        seasonal_temp = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # Precipitation patterns
        if month in [12, 1, 2]:  # Winter
            precip = 50 + np.random.normal(0, 20)
        elif month in [6, 7, 8]:  # Summer
            precip = 80 + np.random.normal(0, 30)
        else:  # Spring/Fall
            precip = 60 + np.random.normal(0, 25)

        # NDVI based on season (higher in summer)
        ndvi = 0.3 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # Elevation estimate (rough)
        elevation = 100 + abs(lat - 40) * 50

        return {
            'temperature': base_temp + seasonal_temp,
            'precipitation': max(0, precip),
            'ndvi': max(0, min(1, ndvi)),
            'elevation': elevation
        }

    def get_environmental_data(self, lat, lon, date):
        """Get environmental data with caching"""
        cache_key = f"{lat:.2f}_{lon:.2f}_{date.strftime('%Y-%m-%d')}"

        if cache_key in self.environmental_cache:
            return self.environmental_cache[cache_key]

        if self.use_earth_engine:
            data = self.get_environmental_data_ee(lat, lon, date)
        else:
            data = self.get_environmental_data_fallback(lat, lon, date)

        self.environmental_cache[cache_key] = data
        return data

    def build_environmental_features(self):
        """Build environmental features for historical data"""
        if self.historical_data.empty:
            return

        print("Building environmental features...")

        # Sample environmental data for historical observations
        environmental_features = []

        for idx, row in self.historical_data.iterrows():
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(self.historical_data)} observations...")

            env_data = self.get_environmental_data(row['lat'], row['lon'], row['date'])

            features = {
                'lat': row['lat'],
                'lon': row['lon'],
                'day_of_year': row['day_of_year'],
                'month': row['month'],
                'year': row['year'],
                'species': row['scientificName'],
                'family': row['family'],
                'genus': row['genus'],
                'state': row['state'],
                **env_data
            }

            environmental_features.append(features)

        self.feature_data = pd.DataFrame(environmental_features)

        # Define feature columns for ML
        self.feature_columns = [
            'lat', 'lon', 'day_of_year', 'month', 'year',
            'temperature', 'precipitation', 'ndvi', 'elevation'
        ]

        print(f"Built features for {len(self.feature_data)} observations")

    def train_model(self):
        """Train machine learning model for bloom prediction"""
        if self.feature_data is None or self.feature_data.empty:
            print("No feature data available for training")
            return

        print("Training bloom prediction model...")

        # Prepare features and target
        X = self.feature_data[self.feature_columns].copy()
        y = self.feature_data['species'].copy()  # Predict species as proxy for bloom

        # Encode categorical target
        if 'species_encoder' not in self.label_encoders:
            self.label_encoders['species_encoder'] = LabelEncoder()

        y_encoded = self.label_encoders['species_encoder'].fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model trained with accuracy: {accuracy:.3f}")

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def predict_blooms_enhanced(self, target_date, aoi_bounds=None, num_predictions=50):
        """
        Enhanced prediction using environmental factors and ML model
        """
        if self.model is None:
            print("Model not trained, falling back to statistical method")
            return self.predict_blooms_statistical(target_date, aoi_bounds, num_predictions)

        predictions = []
        target_day = target_date.timetuple().tm_yday
        target_month = target_date.month
        target_year = target_date.year

        # Set default AOI
        if aoi_bounds is None:
            aoi_bounds = {
                'min_lat': 25, 'max_lat': 50,
                'min_lon': -125, 'max_lon': -65
            }

        # Generate candidate locations within AOI
        candidate_lats = np.random.uniform(aoi_bounds['min_lat'], aoi_bounds['max_lat'], 200)
        candidate_lons = np.random.uniform(aoi_bounds['min_lon'], aoi_bounds['max_lon'], 200)

        for lat, lon in zip(candidate_lats, candidate_lons):
            # Get environmental conditions
            env_data = self.get_environmental_data(lat, lon, target_date)

            # Create feature vector
            features = np.array([[
                lat, lon, target_day, target_month, target_year,
                env_data['temperature'], env_data['precipitation'],
                env_data['ndvi'], env_data['elevation']
            ]])

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Get prediction probabilities for all species
            species_probs = self.model.predict_proba(features_scaled)[0]

            # Find most likely species
            best_species_idx = np.argmax(species_probs)
            best_species_prob = species_probs[best_species_idx]

            species_name = self.label_encoders['species_encoder'].inverse_transform([best_species_idx])[0]

            # Get species info
            species_info = self.species_patterns.get(species_name, {})
            historical_mean_day = species_info.get('mean_day', target_day)

            # Calculate bloom probability based on environmental suitability
            day_diff_penalty = 1 - min(1, abs(target_day - historical_mean_day) / 60)
            environmental_factor = np.clip(best_species_prob * day_diff_penalty, 0, 1)

            # Temperature suitability (species-specific)
            temp_suitability = self.calculate_temperature_suitability(
                env_data['temperature'], species_info
            )

            # Precipitation suitability
            precip_suitability = self.calculate_precipitation_suitability(
                env_data['precipitation'], target_month
            )

            # Combined probability
            bloom_probability = environmental_factor * temp_suitability * precip_suitability

            if bloom_probability > 0.01:  # Lower threshold for testing
                # Get species details
                species_row = self.historical_data[
                    self.historical_data['scientificName'] == species_name
                ].iloc[0] if len(self.historical_data[self.historical_data['scientificName'] == species_name]) > 0 else None

                if species_row is not None:
                    # Determine season
                    season = self.determine_season(target_day)

                    # Estimate area based on environmental conditions
                    area = self.estimate_bloom_area(env_data, bloom_probability)

                    feature = {
                        "type": "Feature",
                        "properties": {
                            "Site": species_name,
                            "Family": species_row['family'],
                            "Genus": species_row['genus'],
                            "Season": season,
                            "Area": area,
                            "prediction_confidence": bloom_probability,
                            "predicted_date": target_date.strftime('%Y-%m-%d'),
                            "is_prediction": True,
                            "environmental_factors": {
                                "temperature": env_data['temperature'],
                                "precipitation": env_data['precipitation'],
                                "ndvi": env_data['ndvi'],
                                "elevation": env_data['elevation']
                            }
                        },
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": [[[
                                [lon - 0.01, lat - 0.01],
                                [lon + 0.01, lat - 0.01],
                                [lon + 0.01, lat + 0.01],
                                [lon - 0.01, lat + 0.01],
                                [lon - 0.01, lat - 0.01]
                            ]]]
                        }
                    }

                    predictions.append(feature)

                    if len(predictions) >= num_predictions:
                        break

        return predictions

    def predict_blooms_statistical(self, target_date, aoi_bounds=None, num_predictions=100):
        """Fallback statistical prediction method"""
        if self.historical_data.empty:
            return []

        target_day = target_date.timetuple().tm_yday

        if aoi_bounds is None:
            aoi_bounds = {
                'min_lat': 25, 'max_lat': 50,
                'min_lon': -125, 'max_lon': -65
            }

        predictions = []

        for _ in range(num_predictions):
            candidates = self.historical_data[
                (self.historical_data['lat'] >= aoi_bounds['min_lat']) &
                (self.historical_data['lat'] <= aoi_bounds['max_lat']) &
                (self.historical_data['lon'] >= aoi_bounds['min_lon']) &
                (self.historical_data['lon'] <= aoi_bounds['max_lon'])
            ]

            if candidates.empty:
                continue

            sample = candidates.sample(n=1).iloc[0]
            day_diff = abs(sample['day_of_year'] - target_day)
            base_prob = max(0, 1 - (day_diff / 60))
            environmental_factor = np.random.normal(1, 0.2)
            probability = min(1, base_prob * environmental_factor)

            if np.random.random() < probability:
                lat_noise = np.random.normal(0, 0.5)
                lon_noise = np.random.normal(0, 0.5)

                predicted_lat = sample['lat'] + lat_noise
                predicted_lon = sample['lon'] + lon_noise

                predicted_lat = np.clip(predicted_lat, aoi_bounds['min_lat'], aoi_bounds['max_lat'])
                predicted_lon = np.clip(predicted_lon, aoi_bounds['min_lon'], aoi_bounds['max_lon'])

                area = np.random.lognormal(10, 1)
                season = self.determine_season(target_day)

                feature = {
                    "type": "Feature",
                    "properties": {
                        "Site": sample['scientificName'],
                        "Family": sample['family'],
                        "Genus": sample['genus'],
                        "Season": season,
                        "Area": area,
                        "prediction_confidence": probability,
                        "predicted_date": target_date.strftime('%Y-%m-%d'),
                        "is_prediction": True
                    },
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [[[
                            [predicted_lon - 0.01, predicted_lat - 0.01],
                            [predicted_lon + 0.01, predicted_lat - 0.01],
                            [predicted_lon + 0.01, predicted_lat + 0.01],
                            [predicted_lon - 0.01, predicted_lat + 0.01],
                            [predicted_lon - 0.01, predicted_lat - 0.01]
                        ]]]
                    }
                }

                predictions.append(feature)

        return predictions

    def calculate_temperature_suitability(self, temperature, species_info):
        """Calculate how suitable temperature is for blooming"""
        # Simple temperature suitability model
        optimal_temp = 20  # Assume 20°C is optimal for most blooms
        temp_range = 15   # Acceptable range

        suitability = 1 - min(1, abs(temperature - optimal_temp) / temp_range)
        return max(0.1, suitability)  # Minimum suitability

    def calculate_precipitation_suitability(self, precipitation, month):
        """Calculate precipitation suitability"""
        # Seasonal precipitation expectations
        if month in [12, 1, 2]:  # Winter - lower precipitation
            optimal_precip = 40
        elif month in [6, 7, 8]:  # Summer - higher precipitation
            optimal_precip = 70
        else:  # Spring/Fall - moderate
            optimal_precip = 55

        precip_range = 30
        suitability = 1 - min(1, abs(precipitation - optimal_precip) / precip_range)
        return max(0.1, suitability)

    def estimate_bloom_area(self, env_data, probability):
        """Estimate bloom area based on environmental conditions"""
        # Base area influenced by environmental factors
        base_area = 1000  # sq meters

        # Temperature effect
        temp_factor = 1 + (env_data['temperature'] - 15) * 0.02

        # Precipitation effect
        precip_factor = 1 + (env_data['precipitation'] - 50) * 0.005

        # NDVI effect (vegetation health)
        ndvi_factor = 1 + env_data['ndvi'] * 0.5

        # Probability effect
        prob_factor = probability

        estimated_area = base_area * temp_factor * precip_factor * ndvi_factor * prob_factor

        return max(100, min(10000, estimated_area))

    def determine_season(self, day_of_year):
        """Determine season from day of year"""
        if 80 <= day_of_year <= 172:
            return 'Spring'
        elif 173 <= day_of_year <= 265:
            return 'Summer'
        elif 266 <= day_of_year <= 355:
            return 'Fall'
        else:
            return 'Winter'

    def predict_blooms_for_date(self, target_date, aoi_bounds=None, num_predictions=50):
        """Main prediction method - uses enhanced method if available"""
        return self.predict_blooms_enhanced(target_date, aoi_bounds, num_predictions)

    def predict_blooms_time_series(self, start_date, end_date, aoi_bounds=None, interval_days=7):
        """Generate bloom predictions for a time series"""
        predictions = {}
        current_date = start_date

        while current_date <= end_date:
            daily_predictions = self.predict_blooms_for_date(current_date, aoi_bounds)
            predictions[current_date.strftime('%Y-%m-%d')] = daily_predictions
            current_date += timedelta(days=interval_days)

        return predictions

def create_geojson_from_predictions(predictions_dict):
    """Convert predictions dict to GeoJSON FeatureCollection"""
    all_features = []

    for date, features in predictions_dict.items():
        for feature in features:
            feature['properties']['date'] = date
            all_features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": all_features
    }
