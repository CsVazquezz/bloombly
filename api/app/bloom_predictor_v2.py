import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                            precision_score, recall_score, f1_score, roc_auc_score)
import joblib
import os
import ee
from collections import defaultdict
import json
from google.oauth2 import service_account
import threading

class ImprovedBloomPredictor:
    """
    Improved bloom predictor that learns actual bloom dynamics using:
    1. Binary classification (bloom vs no-bloom)
    2. Temporal features (lag, rolling averages, trends)
    3. Proper negative examples from non-bloom periods
    4. Environmental factors from Earth Engine
    5. Time-series aware validation
    """
    
    def __init__(self, data_path='../backend/data.csv', use_earth_engine=True, 
                 load_pretrained=None):
        """
        Initialize Bloom Predictor
        
        Args:
            data_path: Path to historical bloom data CSV
            use_earth_engine: Whether to use Google Earth Engine
            load_pretrained: Path to pre-trained model file (if None, trains new model)
        """
        self.data_path = data_path
        self.use_earth_engine = use_earth_engine
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.species_bloom_windows = {}
        self.environmental_cache = {}
        self.is_training = False
        
        # Initialize Earth Engine
        self._initialize_earth_engine()
        
        # Try to load pre-trained model first
        if load_pretrained and os.path.exists(load_pretrained):
            print(f"Loading pre-trained model from {load_pretrained}...")
            try:
                self.load_model(load_pretrained)
                print("✓ Pre-trained model loaded successfully!")
                # Model already has historical_blooms and everything needed
                return  # Skip training
            except Exception as e:
                print(f"⚠ Failed to load pre-trained model: {e}")
                print("  Training new model instead...")
        
        # If no pre-trained model, train in the background
        else:
            print("No pre-trained model found. Starting background training...")
            self.is_training = True
            # Load data synchronously so it's available for inspection
            self.load_and_process_data()
            
            # Run the rest of the training process in a background thread
            training_thread = threading.Thread(
                target=self._train_and_save_model_background,
                args=(load_pretrained if load_pretrained else 'bloom_model_v2.pkl',)
            )
            training_thread.daemon = True  # Allow main thread to exit
            training_thread.start()
        
    def _train_and_save_model_background(self, model_path):
        """
        Runs the full training pipeline in a background thread and saves the model.
        """
        try:
            print("Generating negative examples...")
            self.generate_negative_examples()
            
            print("Building temporal features...")
            self.build_temporal_features()
            
            print("Training bloom prediction model...")
            self.train_model()
            
            print(f"Saving model to {model_path}...")
            self.save_model(model_path)
            
        except Exception as e:
            print(f"Error during background training: {e}")
        finally:
            # Mark training as complete
            self.is_training = False
            print("Background training finished.")

    def _initialize_earth_engine(self):
        """Initialize Earth Engine with proper error handling"""
        if not self.use_earth_engine:
            return
            
        try:
            # Test if already initialized
            ee.Number(1).getInfo()
            print("✓ Earth Engine already initialized")
        except:
            try:
                # Try default credentials
                ee.Initialize()
                print("✓ Earth Engine initialized with default credentials")
            except:
                try:
                    # Try service account
                    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
                    project_id = os.getenv('EE_PROJECT')
                    
                    if credentials_json and project_id:
                        credentials_dict = json.loads(credentials_json)
                        SCOPES = [
                            'https://www.googleapis.com/auth/earthengine',
                            'https://www.googleapis.com/auth/cloud-platform',
                        ]
                        credentials = service_account.Credentials.from_service_account_info(
                            credentials_dict, scopes=SCOPES
                        )
                        ee.Initialize(credentials=credentials, project=project_id)
                        print("✓ Earth Engine initialized with service account")
                    else:
                        raise Exception("No credentials available")
                except Exception as e:
                    print(f"⚠ Earth Engine not available: {e}")
                    print("  Using fallback climate normals")
                    self.use_earth_engine = False
    
    def load_and_process_data(self):
        """Load and process historical bloom observations"""
        try:
            df = pd.read_csv(self.data_path)
            
            # Parse dates and extract temporal features
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            # Spatial features
            df['lat'] = df['latitude'].astype(float)
            df['lon'] = df['longitude'].astype(float)
            
            # All historical observations are bloom events
            df['bloom'] = 1
            
            # Sort by date for time-series processing
            df = df.sort_values('date').reset_index(drop=True)
            
            self.historical_blooms = df
            
            # Analyze bloom windows for each species
            self._analyze_bloom_windows()
            
            print(f"✓ Loaded {len(df)} bloom observations")
            print(f"  Species: {df['scientificName'].nunique()}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            self.historical_blooms = pd.DataFrame()
    
    def _analyze_bloom_windows(self):
        """Analyze typical bloom periods for each species"""
        for species in self.historical_blooms['scientificName'].unique():
            species_data = self.historical_blooms[
                self.historical_blooms['scientificName'] == species
            ]
            
            self.species_bloom_windows[species] = {
                'mean_day': species_data['day_of_year'].mean(),
                'std_day': species_data['day_of_year'].std(),
                'min_day': species_data['day_of_year'].min(),
                'max_day': species_data['day_of_year'].max(),
                'peak_months': species_data['month'].mode().tolist(),
                'lat_range': (species_data['lat'].min(), species_data['lat'].max()),
                'lon_range': (species_data['lon'].min(), species_data['lon'].max()),
                'years': sorted(species_data['year'].unique()),
                'count': len(species_data)
            }
    
    def generate_negative_examples(self):
        """
        Generate negative examples (non-bloom observations) by:
        1. Using same locations as blooms but during off-season
        2. Using random locations within species range during bloom season
        3. Using temporal offsets (before/after bloom events)
        """
        negative_examples = []
        
        for species in self.historical_blooms['scientificName'].unique():
            species_blooms = self.historical_blooms[
                self.historical_blooms['scientificName'] == species
            ]
            bloom_info = self.species_bloom_windows[species]
            
            # Strategy 1: Same locations during off-season
            for _, bloom in species_blooms.iterrows():
                bloom_day = bloom['day_of_year']
                
                # Generate 2-3 negative examples per bloom
                # Before bloom season (2-3 months earlier)
                offset_days = np.random.randint(60, 90)
                neg_date = bloom['date'] - timedelta(days=offset_days)
                
                negative_examples.append({
                    'date': neg_date,
                    'lat': bloom['lat'],
                    'lon': bloom['lon'],
                    'scientificName': species,
                    'family': bloom['family'],
                    'genus': bloom['genus'],
                    'bloom': 0,
                    'generation_method': 'temporal_offset_before'
                })
                
                # After bloom season (2-3 months later)
                offset_days = np.random.randint(60, 90)
                neg_date = bloom['date'] + timedelta(days=offset_days)
                
                negative_examples.append({
                    'date': neg_date,
                    'lat': bloom['lat'],
                    'lon': bloom['lon'],
                    'scientificName': species,
                    'family': bloom['family'],
                    'genus': bloom['genus'],
                    'bloom': 0,
                    'generation_method': 'temporal_offset_after'
                })
            
            # Strategy 2: Random locations within species range during off-season
            n_spatial_negatives = len(species_blooms) // 2
            
            for _ in range(n_spatial_negatives):
                # Random location within species range
                lat = np.random.uniform(bloom_info['lat_range'][0], 
                                       bloom_info['lat_range'][1])
                lon = np.random.uniform(bloom_info['lon_range'][0], 
                                       bloom_info['lon_range'][1])
                
                # Random date outside peak bloom window
                year = np.random.choice(bloom_info['years'])
                
                # Choose a day far from mean bloom day
                mean_day = bloom_info['mean_day']
                off_season_day = int((mean_day + 180) % 365)  # Opposite season
                off_season_day += np.random.randint(-30, 30)  # Add noise
                
                neg_date = datetime(year, 1, 1) + timedelta(days=off_season_day)
                
                negative_examples.append({
                    'date': neg_date,
                    'lat': lat,
                    'lon': lon,
                    'scientificName': species,
                    'family': species_blooms.iloc[0]['family'],
                    'genus': species_blooms.iloc[0]['genus'],
                    'bloom': 0,
                    'generation_method': 'spatial_random_off_season'
                })
        
        # Convert to DataFrame
        self.negative_examples = pd.DataFrame(negative_examples)
        
        # Parse dates
        self.negative_examples['date'] = pd.to_datetime(self.negative_examples['date'])
        self.negative_examples['year'] = self.negative_examples['date'].dt.year
        self.negative_examples['month'] = self.negative_examples['date'].dt.month
        self.negative_examples['day_of_year'] = self.negative_examples['date'].dt.dayofyear
        self.negative_examples['week_of_year'] = self.negative_examples['date'].dt.isocalendar().week
        
        print(f"✓ Generated {len(self.negative_examples)} negative examples")
        print(f"  Positive:Negative ratio = 1:{len(self.negative_examples)/len(self.historical_blooms):.1f}")
    
    def get_environmental_data_ee(self, lat, lon, date, days_before=30):
        """
        Get environmental data from Earth Engine with temporal aggregation
        Returns averages/trends over the period leading up to the date
        """
        try:
            point = ee.Geometry.Point([lon, lat])
            end_date = ee.Date(date.strftime('%Y-%m-%d'))
            start_date = end_date.advance(-days_before, 'day')
            
            # Temperature (MODIS LST)
            lst = ee.ImageCollection('MODIS/061/MOD11A1') \
                .filterDate(start_date, end_date) \
                .select('LST_Day_1km')
            
            lst_mean = lst.mean()
            lst_max = lst.max()
            lst_min = lst.min()
            
            # Precipitation (CHIRPS)
            precip = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                .filterDate(start_date, end_date)
            
            precip_total = precip.sum().select('precipitation')
            precip_mean = precip.mean().select('precipitation')
            
            # NDVI (MODIS)
            ndvi = ee.ImageCollection('MODIS/061/MOD13Q1') \
                .filterDate(start_date, end_date) \
                .select('NDVI')
            
            ndvi_mean = ndvi.mean()
            ndvi_max = ndvi.max()
            
            # Calculate NDVI trend (slope)
            def add_time_band(image):
                time = image.metadata('system:time_start').divide(1000 * 60 * 60 * 24)
                return image.addBands(time.rename('time')).float()
            
            ndvi_time = ndvi.map(add_time_band)
            ndvi_trend = ndvi_time.select(['time', 'NDVI']).reduce(ee.Reducer.linearFit())
            
            # Elevation (static)
            elevation = ee.Image('USGS/SRTMGL1_003')
            
            # Extract values
            scale = 1000  # 1km resolution
            
            result = {
                'temp_mean': self._safe_extract(lst_mean, point, scale, 'LST_Day_1km'),
                'temp_max': self._safe_extract(lst_max, point, scale, 'LST_Day_1km'),
                'temp_min': self._safe_extract(lst_min, point, scale, 'LST_Day_1km'),
                'precip_total': self._safe_extract(precip_total, point, scale, 'precipitation'),
                'precip_mean': self._safe_extract(precip_mean, point, scale, 'precipitation'),
                'ndvi_mean': self._safe_extract(ndvi_mean, point, scale, 'NDVI'),
                'ndvi_max': self._safe_extract(ndvi_max, point, scale, 'NDVI'),
                'ndvi_trend': self._safe_extract(ndvi_trend, point, scale, 'scale'),
                'elevation': self._safe_extract(elevation, point, scale, 'elevation')
            }
            
            # Convert temperature from Kelvin to Celsius and scale
            for key in ['temp_mean', 'temp_max', 'temp_min']:
                if result[key] > 100:  # Likely in Kelvin
                    result[key] = result[key] * 0.02 - 273.15
            
            # Scale NDVI to 0-1
            for key in ['ndvi_mean', 'ndvi_max']:
                if result[key] > 1:
                    result[key] = result[key] / 10000.0
            
            return result
            
        except Exception as e:
            print(f"⚠ EE error for {lat:.2f}, {lon:.2f} on {date}: {e}")
            return self.get_environmental_data_fallback(lat, lon, date)
    
    def _safe_extract(self, image, point, scale, band_name):
        """Safely extract value from EE image"""
        try:
            value = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=scale
            ).get(band_name)
            
            if value is not None:
                return float(ee.Number(value).getInfo())
            return 0.0
        except:
            return 0.0
    
    def get_environmental_data_fallback(self, lat, lon, date):
        """Fallback environmental data using climate normals"""
        day_of_year = date.timetuple().tm_yday
        month = date.month
        
        # Temperature model (latitude and seasonal)
        base_temp = 15 - abs(lat - 35) * 0.4
        seasonal_temp = 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        temp_mean = base_temp + seasonal_temp
        temp_max = temp_mean + 5
        temp_min = temp_mean - 5
        
        # Precipitation (seasonal patterns)
        if month in [6, 7, 8]:  # Summer
            precip_mean = 3.5
            precip_total = 105
        elif month in [12, 1, 2]:  # Winter
            precip_mean = 2.0
            precip_total = 60
        else:  # Spring/Fall
            precip_mean = 2.5
            precip_total = 75
        
        # NDVI (vegetation greenness)
        ndvi_mean = 0.35 + 0.35 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        ndvi_max = min(0.9, ndvi_mean + 0.15)
        ndvi_trend = 0.002 if 80 <= day_of_year <= 200 else -0.002
        
        # Elevation (rough estimate)
        elevation = max(0, 100 + abs(lat - 40) * 30)
        
        return {
            'temp_mean': temp_mean,
            'temp_max': temp_max,
            'temp_min': temp_min,
            'precip_total': precip_total,
            'precip_mean': precip_mean,
            'ndvi_mean': max(0, min(1, ndvi_mean)),
            'ndvi_max': max(0, min(1, ndvi_max)),
            'ndvi_trend': ndvi_trend,
            'elevation': elevation
        }
    
    def get_environmental_data(self, lat, lon, date):
        """Get environmental data with caching"""
        cache_key = f"{lat:.3f}_{lon:.3f}_{date.strftime('%Y-%m-%d')}"
        
        if cache_key in self.environmental_cache:
            return self.environmental_cache[cache_key]
        
        if self.use_earth_engine:
            data = self.get_environmental_data_ee(lat, lon, date)
        else:
            data = self.get_environmental_data_fallback(lat, lon, date)
        
        self.environmental_cache[cache_key] = data
        return data
    
    def build_temporal_features(self):
        """Build comprehensive feature set for bloom prediction"""
        # Combine positive and negative examples
        all_data = pd.concat([
            self.historical_blooms[['date', 'lat', 'lon', 'scientificName', 
                                   'family', 'genus', 'year', 'month', 
                                   'day_of_year', 'week_of_year', 'bloom']],
            self.negative_examples[['date', 'lat', 'lon', 'scientificName', 
                                   'family', 'genus', 'year', 'month', 
                                   'day_of_year', 'week_of_year', 'bloom']]
        ]).sort_values('date').reset_index(drop=True)
        
        print(f"Building features for {len(all_data)} total observations...")
        
        features_list = []
        
        for idx, row in all_data.iterrows():
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(all_data)}...")
            
            # Get environmental data
            env_data = self.get_environmental_data(row['lat'], row['lon'], row['date'])
            
            # Get species bloom window info
            bloom_window = self.species_bloom_windows.get(row['scientificName'], {})
            
            # Temporal features
            mean_bloom_day = bloom_window.get('mean_day', row['day_of_year'])
            days_from_mean = abs(row['day_of_year'] - mean_bloom_day)
            days_from_mean = min(days_from_mean, 365 - days_from_mean)  # Handle wrap-around
            
            # Seasonal encoding (sin/cos for cyclical nature)
            day_sin = np.sin(2 * np.pi * row['day_of_year'] / 365)
            day_cos = np.cos(2 * np.pi * row['day_of_year'] / 365)
            
            # Build feature vector
            features = {
                # Spatial
                'lat': row['lat'],
                'lon': row['lon'],
                
                # Temporal
                'day_of_year': row['day_of_year'],
                'month': row['month'],
                'week_of_year': row['week_of_year'],
                'day_sin': day_sin,
                'day_cos': day_cos,
                'days_from_species_mean': days_from_mean,
                
                # Environmental - current conditions
                'temp_mean': env_data['temp_mean'],
                'temp_max': env_data['temp_max'],
                'temp_min': env_data['temp_min'],
                'temp_range': env_data['temp_max'] - env_data['temp_min'],
                'precip_total': env_data['precip_total'],
                'precip_mean': env_data['precip_mean'],
                'ndvi_mean': env_data['ndvi_mean'],
                'ndvi_max': env_data['ndvi_max'],
                'ndvi_trend': env_data['ndvi_trend'],
                'elevation': env_data['elevation'],
                
                # Derived features
                'growing_degree_days': max(0, env_data['temp_mean'] - 10) * 30,  # Approximation
                'moisture_index': env_data['precip_total'] / (env_data['temp_mean'] + 20),
                'vegetation_health': env_data['ndvi_mean'] * (1 + env_data['ndvi_trend']),
                
                # Species encoding (one-hot would be better, but keeping simple)
                'species': row['scientificName'],
                
                # Target
                'bloom': row['bloom']
            }
            
            features_list.append(features)
        
        self.feature_data = pd.DataFrame(features_list)
        
        # Define features for model (excluding target and non-numeric)
        self.feature_columns = [
            'lat', 'lon', 'day_of_year', 'month', 'week_of_year',
            'day_sin', 'day_cos', 'days_from_species_mean',
            'temp_mean', 'temp_max', 'temp_min', 'temp_range',
            'precip_total', 'precip_mean',
            'ndvi_mean', 'ndvi_max', 'ndvi_trend',
            'elevation', 'growing_degree_days', 'moisture_index',
            'vegetation_health'
        ]
        
        print(f"✓ Built {len(self.feature_columns)} features for {len(self.feature_data)} observations")
        print(f"  Class distribution: Bloom={sum(self.feature_data['bloom'])}, "
              f"No-bloom={sum(self.feature_data['bloom']==0)}")
    
    def train_model(self):
        """Train bloom prediction model with proper validation"""
        if self.feature_data is None or self.feature_data.empty:
            print("✗ No feature data available")
            return
        
        # Prepare features and target
        X = self.feature_data[self.feature_columns].copy()
        y = self.feature_data['bloom'].copy()
        
        # Handle any missing values
        X = X.fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Use time-series split for validation (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        print("Training model with time-series cross-validation...")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, 
                                   scoring='roc_auc', n_jobs=-1)
        
        print(f"  Cross-val ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Train final model on all data
        self.model.fit(X_scaled, y)
        
        # Evaluate on training data (just for diagnostics)
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        print(f"\n✓ Model Training Complete")
        print(f"  Accuracy: {accuracy_score(y, y_pred):.3f}")
        print(f"  Precision: {precision_score(y, y_pred):.3f}")
        print(f"  Recall: {recall_score(y, y_pred):.3f}")
        print(f"  F1-Score: {f1_score(y, y_pred):.3f}")
        print(f"  ROC-AUC: {roc_auc_score(y, y_pred_proba):.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']:30s} {row['importance']:.4f}")
    
    def predict_bloom_probability(self, lat, lon, date, species=None):
        """
        Predict bloom probability for a specific location, date, and species
        Returns probability between 0 and 1
        """
        if self.model is None or self.is_training:
            return 0.0
        
        # If no species specified, use the most common one
        if species is None:
            species = self.historical_blooms['scientificName'].mode()[0]
        
        # Get environmental data
        env_data = self.get_environmental_data(lat, lon, date)
        
        # Get species info
        bloom_window = self.species_bloom_windows.get(species, {})
        
        # Calculate temporal features
        day_of_year = date.timetuple().tm_yday
        mean_bloom_day = bloom_window.get('mean_day', day_of_year)
        days_from_mean = abs(day_of_year - mean_bloom_day)
        days_from_mean = min(days_from_mean, 365 - days_from_mean)
        
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)
        
        # Build feature vector
        features = np.array([[
            lat, lon, day_of_year, date.month, date.isocalendar()[1],
            day_sin, day_cos, days_from_mean,
            env_data['temp_mean'], env_data['temp_max'], env_data['temp_min'],
            env_data['temp_max'] - env_data['temp_min'],
            env_data['precip_total'], env_data['precip_mean'],
            env_data['ndvi_mean'], env_data['ndvi_max'], env_data['ndvi_trend'],
            env_data['elevation'],
            max(0, env_data['temp_mean'] - 10) * 30,
            env_data['precip_total'] / (env_data['temp_mean'] + 20),
            env_data['ndvi_mean'] * (1 + env_data['ndvi_trend'])
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict probability
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        return float(probability)
    
    def predict_blooms_for_date(self, target_date, aoi_bounds=None, 
                                num_predictions=100, confidence_threshold=0.3):
        """
        Predict blooms for a specific date using learned dynamics
        """
        print(f"🔍 Starting prediction for {target_date}, threshold={confidence_threshold}")
        
        if self.model is None:
            print("✗ Model not trained")
            return []
        
        # Set default AOI
        if aoi_bounds is None:
            aoi_bounds = {
                'min_lat': 25, 'max_lat': 50,
                'min_lon': -125, 'max_lon': -65
            }
        
        predictions = []
        
        # OPTIMIZATION: Temporarily disable Earth Engine during predictions
        # to use fast fallback data (otherwise hundreds of slow EE API calls)
        original_ee_state = self.use_earth_engine
        self.use_earth_engine = False
        print(f"  → Using fallback environmental data (EE disabled for speed)")
        
        try:
            # Generate predictions for known species
            print(f"  → Generating predictions for {len(self.species_bloom_windows)} species...")
            for i, (species, bloom_info) in enumerate(self.species_bloom_windows.items()):
                # Early stopping if we have enough predictions
                if len(predictions) >= num_predictions * 3:  # Increased from 2x to 3x
                    print(f"    Early stopping: already have {len(predictions)} predictions")
                    break
                    
                print(f"    Processing species {i+1}/{len(self.species_bloom_windows)}: {species}")
                
                # Sample locations within the AOI bounds
                # Increased samples to get more predictions
                n_samples = min(100, max(50, num_predictions // len(self.species_bloom_windows) * 2))
                
                # Generate random locations within AOI bounds
                candidate_lats = np.random.uniform(
                    aoi_bounds['min_lat'], 
                    aoi_bounds['max_lat'], 
                    n_samples
                )
                candidate_lons = np.random.uniform(
                    aoi_bounds['min_lon'], 
                    aoi_bounds['max_lon'], 
                    n_samples
                )
                
                # Predict for each location
                for lat, lon in zip(candidate_lats, candidate_lons):
                    probability = self.predict_bloom_probability(lat, lon, target_date, species)
                    
                    if probability >= confidence_threshold:
                        # Get species details
                        species_row = self.historical_blooms[
                            self.historical_blooms['scientificName'] == species
                        ].iloc[0]
                        
                        # Estimate bloom area based on probability and environmental conditions
                        env_data = self.get_environmental_data(lat, lon, target_date)
                        area = self._estimate_bloom_area(probability, env_data)
                        
                        feature = {
                            "type": "Feature",
                            "properties": {
                                "Site": species,
                                "Family": species_row['family'],
                                "Genus": species_row['genus'],
                                "Season": self._get_season(target_date.timetuple().tm_yday),
                                "Area": area,
                                "bloom_probability": round(probability, 3),
                                "predicted_date": target_date.strftime('%Y-%m-%d'),
                                "is_prediction": True,
                                "model_version": "v2_bloom_dynamics",
                                "environmental_factors": {
                                    "temperature": round(env_data['temp_mean'], 1),
                                    "precipitation": round(env_data['precip_total'], 1),
                                    "ndvi": round(env_data['ndvi_mean'], 3),
                                    "ndvi_trend": round(env_data['ndvi_trend'], 4)
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
            
            # Sort by probability and return top predictions
            print(f"  → Sorting {len(predictions)} predictions and returning top {num_predictions}")
            predictions.sort(key=lambda x: x['properties']['bloom_probability'], reverse=True)
            result = predictions[:num_predictions]
            print(f"✓ Prediction complete! Returning {len(result)} blooms")
            return result
        
        finally:
            # Restore original Earth Engine state
            self.use_earth_engine = original_ee_state
    
    def _estimate_bloom_area(self, probability, env_data):
        """Estimate bloom area based on probability and environmental conditions"""
        base_area = 5000  # square meters
        
        # Scale by probability
        area = base_area * probability
        
        # Adjust for vegetation health
        vegetation_factor = 0.5 + env_data['ndvi_mean']
        area *= vegetation_factor
        
        # Adjust for moisture
        moisture_factor = min(2.0, env_data['precip_total'] / 50)
        area *= moisture_factor
        
        return max(100, min(50000, area))
    
    def _get_season(self, day_of_year):
        """Determine season from day of year"""
        if 80 <= day_of_year <= 172:
            return 'Spring'
        elif 173 <= day_of_year <= 265:
            return 'Summer'
        elif 266 <= day_of_year <= 355:
            return 'Fall'
        else:
            return 'Winter'
    
    def save_model(self, path='bloom_model_v2.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'species_bloom_windows': self.species_bloom_windows,
            'use_earth_engine': self.use_earth_engine,
            'historical_blooms': self.historical_blooms,  # Save historical data too
            'negative_examples': self.negative_examples,
            'feature_data': self.feature_data if hasattr(self, 'feature_data') else None
        }
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path='bloom_model_v2.pkl'):
        """Load trained model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.species_bloom_windows = model_data['species_bloom_windows']
        self.use_earth_engine = model_data.get('use_earth_engine', False)
        self.historical_blooms = model_data.get('historical_blooms', pd.DataFrame())
        self.negative_examples = model_data.get('negative_examples', pd.DataFrame())
        self.feature_data = model_data.get('feature_data', None)
        print(f"✓ Model loaded from {path}")
        print(f"  Loaded {len(self.historical_blooms)} bloom observations")
        print(f"  Loaded {len(self.species_bloom_windows)} species")
        if self.feature_data is not None:
            print(f"  Loaded {len(self.feature_data)} feature samples")
