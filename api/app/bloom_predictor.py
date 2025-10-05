import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import ee
import config

class EnhancedBloomPredictor:
    """
    Enhanced bloom predictor that incorporates environmental factors
    for more accurate future predictions.
    """
    def __init__(self, data_path='../backend/data.csv', use_earth_engine=True):
        """
        Initializes the EnhancedBloomPredictor.

        Args:
            data_path (str, optional): The path to the historical data file. Defaults to '../backend/data.csv'.
            use_earth_engine (bool, optional): Whether to use Google Earth Engine for environmental data. Defaults to True.
        """
        self.data_path = data_path
        self.use_earth_engine = use_earth_engine
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.species_patterns = {}
        self.environmental_cache = {}

        if self.use_earth_engine:
            try:
                ee.Number(1).getInfo()
                logging.info("Earth Engine is available and initialized.")
            except ee.EEException:
                logging.warning("Earth Engine not available, using fallback methods.")
                self.use_earth_engine = False

        self.load_and_process_data()
        self.build_environmental_features()
        self.train_model()

    def load_and_process_data(self):
        """Loads and processes historical bloom data from a CSV file."""
        try:
            df = pd.read_csv(self.data_path)
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_year'] = df['date'].dt.dayofyear
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['lat'] = df['latitude'].astype(float)
            df['lon'] = df['longitude'].astype(float)
            df['bloom'] = 1
            df['state'] = self.get_state_from_coords(df['lat'], df['lon'])
            self.historical_data = df
            self.analyze_patterns()
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            self.historical_data = pd.DataFrame()

    def get_state_from_coords(self, lat, lon):
        """Assigns a state to a given latitude and longitude."""
        # This is a simplified implementation. A more robust solution would use a reverse geocoding library.
        for state, bounds in config.STATE_BOUNDS.items():
            if bounds['min_lat'] <= lat <= bounds['max_lat'] and bounds['min_lon'] <= lon <= bounds['max_lon']:
                return state
        return 'Other'

    def analyze_patterns(self):
        """Analyzes historical data to find patterns in bloom occurrences."""
        if self.historical_data.empty:
            return

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
        """Retrieves environmental data from Google Earth Engine."""
        try:
            point = ee.Geometry.Point([lon, lat])
            
            # Temperature
            lst_collection = ee.ImageCollection('MODIS/061/MOD11A1').filterDate(date - timedelta(days=7), date + timedelta(days=1))
            lst = lst_collection.select('LST_Day_1km').mean()
            temperature = lst.reduceRegion(ee.Reducer.mean(), point, 1000).get('LST_Day_1km').getInfo()
            if temperature:
                temperature = temperature * 0.02 - 273.15

            # Precipitation
            precip_collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(date - timedelta(days=30), date)
            precip = precip_collection.sum()
            precipitation = precip.reduceRegion(ee.Reducer.mean(), point, 1000).get('precipitation').getInfo()

            # NDVI
            ndvi_collection = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(date - timedelta(days=30), date)
            ndvi = ndvi_collection.select('NDVI').mean()
            ndvi_value = ndvi.reduceRegion(ee.Reducer.mean(), point, 1000).get('NDVI').getInfo()
            if ndvi_value:
                ndvi_value /= 10000.0

            # Elevation
            elevation = ee.Image('USGS/SRTMGL1_003').reduceRegion(ee.Reducer.mean(), point, 1000).get('elevation').getInfo()

            return {
                'temperature': temperature or 0.0,
                'precipitation': precipitation or 0.0,
                'ndvi': ndvi_value or 0.0,
                'elevation': elevation or 0.0
            }

        except Exception as e:
            logging.error(f"Error getting EE data for {lat}, {lon} on {date}: {e}")
            return self.get_environmental_data_fallback(lat, lon, date)

    def get_environmental_data_fallback(self, lat, lon, date):
        """Provides fallback environmental data when Earth Engine is not available."""
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        base_temp = 15 - abs(lat) * 0.3
        seasonal_temp = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        precip = 60 + np.random.normal(0, 25)
        if month in [12, 1, 2]:
            precip = 50 + np.random.normal(0, 20)
        elif month in [6, 7, 8]:
            precip = 80 + np.random.normal(0, 30)
            
        ndvi = 0.3 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        elevation = 100 + abs(lat - 40) * 50

        return {
            'temperature': base_temp + seasonal_temp,
            'precipitation': max(0, precip),
            'ndvi': max(0, min(1, ndvi)),
            'elevation': elevation
        }

    def get_environmental_data(self, lat, lon, date):
        """Retrieves environmental data, using a cache if available."""
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
        """Builds environmental features for the historical data."""
        if self.historical_data.empty:
            return

        logging.info("Building environmental features...")
        environmental_features = []
        for idx, row in self.historical_data.iterrows():
            if idx % 100 == 0:
                logging.info(f"Processing {idx}/{len(self.historical_data)} observations...")
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
                'bloom': row['bloom'],
                **env_data
            }
            environmental_features.append(features)

        self.feature_data = pd.DataFrame(environmental_features)
        self.feature_columns = ['lat', 'lon', 'day_of_year', 'month', 'year', 'temperature', 'precipitation', 'ndvi', 'elevation']
        logging.info(f"Built features for {len(self.feature_data)} observations.")

    def train_model(self):
        """Trains the bloom prediction model using GridSearchCV to find the best hyperparameters."""
        if self.feature_data is None or self.feature_data.empty:
            logging.warning("No feature data available for training.")
            return

        logging.info("Training bloom prediction model...")
        X = self.feature_data[self.feature_columns].copy()
        y = self.feature_data['bloom'].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Create a GridSearchCV object
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        # Set the best estimator as the model
        self.model = grid_search.best_estimator_

        # Evaluate the model on the test set
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model trained with accuracy: {accuracy:.3f}")
        logging.info(f"Best hyperparameters: {grid_search.best_params_}")

    def predict_blooms_enhanced(self, target_date, aoi_bounds=None, num_predictions=50):
        """Predicts blooms using the enhanced machine learning model."""
        if self.model is None:
            logging.warning("Model not trained, falling back to statistical method.")
            return self.predict_blooms_statistical(target_date, aoi_bounds, num_predictions)

        predictions = []
        target_day = target_date.timetuple().tm_yday
        target_month = target_date.month
        target_year = target_date.year

        if aoi_bounds is None:
            aoi_bounds = {'min_lat': 25, 'max_lat': 50, 'min_lon': -125, 'max_lon': -65}

        # Generate random candidate locations within the AOI
        candidate_lats = np.random.uniform(aoi_bounds['min_lat'], aoi_bounds['max_lat'], 200)
        candidate_lons = np.random.uniform(aoi_bounds['min_lon'], aoi_bounds['max_lon'], 200)

        for lat, lon in zip(candidate_lats, candidate_lons):
            # Get environmental data for the candidate location
            env_data = self.get_environmental_data(lat, lon, target_date)
            
            # Create a feature vector for the candidate location
            features = np.array([[lat, lon, target_day, target_month, target_year, env_data['temperature'], env_data['precipitation'], env_data['ndvi'], env_data['elevation']]])
            features_scaled = self.scaler.transform(features)

            # Predict the probability of a bloom
            bloom_probability = self.model.predict_proba(features_scaled)[0][1]

            # If the bloom probability is above a certain threshold, create a GeoJSON feature for the prediction
            if bloom_probability > 0.5:
                # Since we are not predicting species anymore, we will use a generic name
                species_name = "Algal Bloom"
                family = "Unknown"
                genus = "Unknown"

                season = self.determine_season(target_day)
                area = self.estimate_bloom_area(env_data, bloom_probability)
                feature = {
                    "type": "Feature",
                    "properties": {
                        "Site": species_name,
                        "Family": family,
                        "Genus": genus,
                        "Season": season,
                        "Area": area,
                        "prediction_confidence": bloom_probability,
                        "predicted_date": target_date.strftime('%Y-%m-%d'),
                        "is_prediction": True,
                        "environmental_factors": env_data
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
        """Fallback statistical prediction method."""
        if self.historical_data.empty:
            return []

        target_day = target_date.timetuple().tm_yday
        if aoi_bounds is None:
            aoi_bounds = {'min_lat': 25, 'max_lat': 50, 'min_lon': -125, 'max_lon': -65}

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
                predicted_lat = np.clip(sample['lat'] + lat_noise, aoi_bounds['min_lat'], aoi_bounds['max_lat'])
                predicted_lon = np.clip(sample['lon'] + lon_noise, aoi_bounds['min_lon'], aoi_bounds['max_lon'])
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
        """Calculates the suitability of the temperature for a bloom."""
        optimal_temp = 20
        temp_range = 15
        suitability = 1 - min(1, abs(temperature - optimal_temp) / temp_range)
        return max(0.1, suitability)

    def calculate_precipitation_suitability(self, precipitation, month):
        """Calculates the suitability of the precipitation for a bloom."""
        if month in [12, 1, 2]:
            optimal_precip = 40
        elif month in [6, 7, 8]:
            optimal_precip = 70
        else:
            optimal_precip = 55
        precip_range = 30
        suitability = 1 - min(1, abs(precipitation - optimal_precip) / precip_range)
        return max(0.1, suitability)

    def estimate_bloom_area(self, env_data, probability):
        """Estimates the area of a bloom based on environmental conditions."""
        base_area = 1000
        temp_factor = 1 + (env_data['temperature'] - 15) * 0.02
        precip_factor = 1 + (env_data['precipitation'] - 50) * 0.005
        ndvi_factor = 1 + env_data['ndvi'] * 0.5
        prob_factor = probability
        estimated_area = base_area * temp_factor * precip_factor * ndvi_factor * prob_factor
        return max(100, min(10000, estimated_area))

    def determine_season(self, day_of_year):
        """Determines the season from the day of the year."""
        if 80 <= day_of_year <= 172:
            return 'Spring'
        elif 173 <= day_of_year <= 265:
            return 'Summer'
        elif 266 <= day_of_year <= 355:
            return 'Fall'
        else:
            return 'Winter'

    def predict_blooms_for_date(self, target_date, aoi_bounds=None, num_predictions=50):
        """Main prediction method - uses enhanced method if available."""
        return self.predict_blooms_enhanced(target_date, aoi_bounds, num_predictions)

    def predict_blooms_time_series(self, start_date, end_date, aoi_bounds=None, interval_days=7):
        """Generates bloom predictions for a time series."""
        predictions = {}
        current_date = start_date
        while current_date <= end_date:
            daily_predictions = self.predict_blooms_for_date(current_date, aoi_bounds)
            predictions[current_date.strftime('%Y-%m-%d')] = daily_predictions
            current_date += timedelta(days=interval_days)
        return predictions