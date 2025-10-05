"""
Feature Engineering Script for Bloombly
Enriches plant bloom datasets with climate, soil, elevation, and photoperiod data.
Uses NASA POWER, Google Earth Engine, Open-Elevation APIs and astronomical calculations.
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import math
import os
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("  python-dotenv not available. Install with: pip install python-dotenv")

# Google Earth Engine import (optional)
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("  Google Earth Engine not available. Install with: pip install earthengine-api")
    print("   Soil features will be skipped unless GEE is configured.")


class BloomFeatureEngineer:
    """Enrich bloom observation data with environmental features for ML"""
    
    def __init__(self, processed_data_dir='../../data/processed', use_gee=True):
        script_dir = Path(__file__).parent
        
        if not Path(processed_data_dir).is_absolute():
            self.processed_data_dir = (script_dir / processed_data_dir).resolve()
        else:
            self.processed_data_dir = Path(processed_data_dir)
            
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints
        self.nasa_power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.elevation_url = "https://api.open-elevation.com/api/v1/lookup"
        
        # Google Earth Engine setup
        self.use_gee = use_gee and GEE_AVAILABLE
        if self.use_gee:
            try:
                # Try to authenticate using service account credentials from .env
                credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
                project_id = os.getenv('EE_PROJECT')
                
                if credentials_json and project_id:
                    # Parse the JSON credentials
                    credentials_dict = json.loads(credentials_json)
                    
                    # Create service account credentials
                    credentials = ee.ServiceAccountCredentials(
                        credentials_dict['client_email'],
                        key_data=credentials_json
                    )
                    
                    # Initialize with service account
                    ee.Initialize(credentials, project=project_id)
                    print("✅ Google Earth Engine initialized successfully with service account")
                else:
                    # Fallback to default authentication
                    ee.Initialize()
                    print("✅ Google Earth Engine initialized successfully")
                    
            except Exception as e:
                print(f"❌ GEE initialization failed: {e}")
                print("   Make sure your .env file has valid GEE credentials")
                self.use_gee = False
        
        # Cache for API responses to avoid redundant calls
        self.weather_cache = {}
        self.soil_cache = {}
        self.elevation_cache = {}
        
    def calculate_photoperiod(self, latitude: float, date: datetime) -> float:
        """
        Calculate day length (photoperiod) in hours for given latitude and date.
        Based on astronomical calculations.
        
        Args:
            latitude: Latitude in degrees (-90 to 90)
            date: Date to calculate photoperiod for
            
        Returns:
            Day length in hours
        """
        # Day of year
        day_of_year = date.timetuple().tm_yday
        
        # Solar declination (degrees)
        declination = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
        
        # Convert latitude to radians
        lat_rad = math.radians(latitude)
        dec_rad = math.radians(declination)
        
        # Hour angle at sunrise/sunset
        try:
            cos_hour_angle = -math.tan(lat_rad) * math.tan(dec_rad)
            
            # Handle polar day/night
            if cos_hour_angle > 1:
                return 0.0  # Polar night
            elif cos_hour_angle < -1:
                return 24.0  # Polar day
            
            hour_angle = math.degrees(math.acos(cos_hour_angle))
            
            # Day length in hours
            day_length = 2 * hour_angle / 15
            
            return round(day_length, 2)
        except:
            return 12.0  # Default to 12 hours if calculation fails
    
    def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """
        Get elevation in meters from Open-Elevation API.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Elevation in meters or None if request fails
        """
        # Check cache
        cache_key = f"{latitude:.4f},{longitude:.4f}"
        if cache_key in self.elevation_cache:
            return self.elevation_cache[cache_key]
        
        try:
            params = {
                "locations": f"{latitude},{longitude}"
            }
            
            response = requests.get(self.elevation_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                elevation = data['results'][0]['elevation']
                self.elevation_cache[cache_key] = elevation
                return elevation
            else:
                print(f"  Elevation API error {response.status_code} for {latitude}, {longitude}")
                return None
                
        except Exception as e:
            print(f"  Elevation request failed: {str(e)}")
            return None
    
    def get_soil_properties_gee(self, latitude: float, longitude: float) -> Dict[str, float]:
        """
        Get soil properties from Google Earth Engine using SoilGrids250m dataset.
        Much faster than SoilGrids REST API with better rate limits.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Dictionary with soil properties at different depths
        """
        # Check cache
        cache_key = f"{latitude:.4f},{longitude:.4f}"
        if cache_key in self.soil_cache:
            return self.soil_cache[cache_key]
        
        if not self.use_gee:
            print(f"  GEE not available, skipping soil data")
            return self._get_default_soil_features()
        
        try:
            # Create point geometry
            point = ee.Geometry.Point([longitude, latitude])
            
            # SoilGrids250m dataset in GEE
            # Available bands: b0, b10, b30, b60, b100, b200 (depths in cm)
            
            soil_features = {}
            
            # Get soil properties from different datasets
            # 1. pH in H2O
            ph = ee.Image("projects/soilgrids-isric/phh2o_mean").select(['phh2o_0-5cm_mean', 'phh2o_5-15cm_mean'])
            ph_values = ph.reduceRegion(ee.Reducer.first(), point, 250).getInfo()
            
            # 2. Clay content
            clay = ee.Image("projects/soilgrids-isric/clay_mean").select(['clay_0-5cm_mean', 'clay_5-15cm_mean'])
            clay_values = clay.reduceRegion(ee.Reducer.first(), point, 250).getInfo()
            
            # 3. Sand content
            sand = ee.Image("projects/soilgrids-isric/sand_mean").select(['sand_0-5cm_mean', 'sand_5-15cm_mean'])
            sand_values = sand.reduceRegion(ee.Reducer.first(), point, 250).getInfo()
            
            # 4. Soil Organic Carbon
            soc = ee.Image("projects/soilgrids-isric/soc_mean").select(['soc_0-5cm_mean', 'soc_5-15cm_mean'])
            soc_values = soc.reduceRegion(ee.Reducer.first(), point, 250).getInfo()
            
            # 5. Nitrogen
            nitrogen = ee.Image("projects/soilgrids-isric/nitrogen_mean").select(['nitrogen_0-5cm_mean', 'nitrogen_5-15cm_mean'])
            nitrogen_values = nitrogen.reduceRegion(ee.Reducer.first(), point, 250).getInfo()
            
            # Process and convert units
            # pH: stored as pH * 10
            soil_features['soil_ph_0-5cm'] = ph_values.get('phh2o_0-5cm_mean', None)
            if soil_features['soil_ph_0-5cm'] is not None:
                soil_features['soil_ph_0-5cm'] = round(soil_features['soil_ph_0-5cm'] / 10, 2)
            
            soil_features['soil_ph_5-15cm'] = ph_values.get('phh2o_5-15cm_mean', None)
            if soil_features['soil_ph_5-15cm'] is not None:
                soil_features['soil_ph_5-15cm'] = round(soil_features['soil_ph_5-15cm'] / 10, 2)
            
            # Clay/Sand: stored as g/kg, convert to %
            soil_features['soil_clay_0-5cm'] = clay_values.get('clay_0-5cm_mean', None)
            if soil_features['soil_clay_0-5cm'] is not None:
                soil_features['soil_clay_0-5cm'] = round(soil_features['soil_clay_0-5cm'] / 10, 2)
            
            soil_features['soil_clay_5-15cm'] = clay_values.get('clay_5-15cm_mean', None)
            if soil_features['soil_clay_5-15cm'] is not None:
                soil_features['soil_clay_5-15cm'] = round(soil_features['soil_clay_5-15cm'] / 10, 2)
            
            soil_features['soil_sand_0-5cm'] = sand_values.get('sand_0-5cm_mean', None)
            if soil_features['soil_sand_0-5cm'] is not None:
                soil_features['soil_sand_0-5cm'] = round(soil_features['soil_sand_0-5cm'] / 10, 2)
            
            soil_features['soil_sand_5-15cm'] = sand_values.get('sand_5-15cm_mean', None)
            if soil_features['soil_sand_5-15cm'] is not None:
                soil_features['soil_sand_5-15cm'] = round(soil_features['soil_sand_5-15cm'] / 10, 2)
            
            # SOC: stored as dg/kg
            soil_features['soil_organic_carbon_0-5cm'] = soc_values.get('soc_0-5cm_mean', None)
            if soil_features['soil_organic_carbon_0-5cm'] is not None:
                soil_features['soil_organic_carbon_0-5cm'] = round(soil_features['soil_organic_carbon_0-5cm'] / 10, 2)
            
            soil_features['soil_organic_carbon_5-15cm'] = soc_values.get('soc_5-15cm_mean', None)
            if soil_features['soil_organic_carbon_5-15cm'] is not None:
                soil_features['soil_organic_carbon_5-15cm'] = round(soil_features['soil_organic_carbon_5-15cm'] / 10, 2)
            
            # Nitrogen: stored as cg/kg
            soil_features['soil_nitrogen_0-5cm'] = nitrogen_values.get('nitrogen_0-5cm_mean', None)
            if soil_features['soil_nitrogen_0-5cm'] is not None:
                soil_features['soil_nitrogen_0-5cm'] = round(soil_features['soil_nitrogen_0-5cm'] / 10, 2)
            
            soil_features['soil_nitrogen_5-15cm'] = nitrogen_values.get('nitrogen_5-15cm_mean', None)
            if soil_features['soil_nitrogen_5-15cm'] is not None:
                soil_features['soil_nitrogen_5-15cm'] = round(soil_features['soil_nitrogen_5-15cm'] / 10, 2)
            
            # Cache the results
            self.soil_cache[cache_key] = soil_features
            return soil_features
            
        except Exception as e:
            print(f"  GEE soil request failed: {str(e)}")
            return self._get_default_soil_features()
    
    def get_soil_properties(self, latitude: float, longitude: float) -> Dict[str, float]:
        """
        Get soil properties using Google Earth Engine (preferred) or fallback method.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Dictionary with soil properties at different depths
        """
        if self.use_gee:
            return self.get_soil_properties_gee(latitude, longitude)
        else:
            return self._get_default_soil_features()
    
    def _get_default_soil_features(self) -> Dict[str, float]:
        """Return default soil features when API fails"""
        return {
            'soil_ph_0-5cm': np.nan,
            'soil_ph_5-15cm': np.nan,
            'soil_clay_0-5cm': np.nan,
            'soil_clay_5-15cm': np.nan,
            'soil_sand_0-5cm': np.nan,
            'soil_sand_5-15cm': np.nan,
            'soil_organic_carbon_0-5cm': np.nan,
            'soil_organic_carbon_5-15cm': np.nan,
            'soil_nitrogen_0-5cm': np.nan,
            'soil_nitrogen_5-15cm': np.nan,
        }
    
    def get_ndvi_gee(self, latitude: float, longitude: float, bloom_date: datetime) -> Dict[str, float]:
        """
        Get NDVI (vegetation index) from Google Earth Engine using Landsat 8/9 or Sentinel-2.
        NDVI indicates vegetation greenness and spring green-up timing.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            bloom_date: Date of bloom observation
            
        Returns:
            Dictionary with NDVI features at different time windows
        """
        if not self.use_gee:
            print(f"  GEE not available, skipping NDVI data")
            return self._get_default_ndvi_features()
        
        try:
            # Create point geometry
            point = ee.Geometry.Point([longitude, latitude])
            
            # Create buffer for regional average (1km radius)
            region = point.buffer(1000)
            
            ndvi_features = {}
            
            # Use Sentinel-2 (better spatial/temporal resolution)
            # Backup with Landsat 8/9 if Sentinel unavailable
            
            # Define time windows before bloom
            windows = {
                '7d': 7,
                '14d': 14,
                '30d': 30,
                '60d': 60
            }
            
            for window_name, days in windows.items():
                start_date = (bloom_date - timedelta(days=days)).strftime('%Y-%m-%d')
                end_date = (bloom_date - timedelta(days=1)).strftime('%Y-%m-%d')
                
                try:
                    # Try Sentinel-2 first (10m resolution, 5-day revisit)
                    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(region) \
                        .filterDate(start_date, end_date) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                    
                    if s2.size().getInfo() > 0:
                        # Calculate NDVI for Sentinel-2
                        def calc_ndvi_s2(image):
                            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                            return image.addBands(ndvi)
                        
                        s2_ndvi = s2.map(calc_ndvi_s2).select('NDVI')
                        
                        # Get statistics
                        stats = s2_ndvi.mean().reduceRegion(
                            reducer=ee.Reducer.mean().combine(
                                ee.Reducer.stdDev(), '', True
                            ).combine(
                                ee.Reducer.max(), '', True
                            ).combine(
                                ee.Reducer.min(), '', True
                            ),
                            geometry=region,
                            scale=10,
                            maxPixels=1e9
                        ).getInfo()
                        
                        ndvi_features[f'ndvi_mean_{window_name}'] = stats.get('NDVI_mean')
                        ndvi_features[f'ndvi_stddev_{window_name}'] = stats.get('NDVI_stdDev')
                        ndvi_features[f'ndvi_max_{window_name}'] = stats.get('NDVI_max')
                        
                    else:
                        # Fallback to Landsat 8/9 (30m resolution)
                        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                            .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')) \
                            .filterBounds(region) \
                            .filterDate(start_date, end_date) \
                            .filter(ee.Filter.lt('CLOUD_COVER', 20))
                        
                        if l8.size().getInfo() > 0:
                            def calc_ndvi_l8(image):
                                ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
                                return image.addBands(ndvi)
                            
                            l8_ndvi = l8.map(calc_ndvi_l8).select('NDVI')
                            
                            stats = l8_ndvi.mean().reduceRegion(
                                reducer=ee.Reducer.mean().combine(
                                    ee.Reducer.stdDev(), '', True
                                ),
                                geometry=region,
                                scale=30,
                                maxPixels=1e9
                            ).getInfo()
                            
                            ndvi_features[f'ndvi_mean_{window_name}'] = stats.get('NDVI_mean')
                            ndvi_features[f'ndvi_stddev_{window_name}'] = stats.get('NDVI_stdDev')
                        else:
                            # No imagery available for this window
                            ndvi_features[f'ndvi_mean_{window_name}'] = None
                            ndvi_features[f'ndvi_stddev_{window_name}'] = None
                            ndvi_features[f'ndvi_max_{window_name}'] = None
                
                except Exception as e:
                    print(f"  NDVI {window_name} failed: {str(e)}")
                    ndvi_features[f'ndvi_mean_{window_name}'] = None
                    ndvi_features[f'ndvi_stddev_{window_name}'] = None
                    ndvi_features[f'ndvi_max_{window_name}'] = None
            
            # Calculate NDVI trend (green-up rate)
            if (ndvi_features.get('ndvi_mean_7d') is not None and 
                ndvi_features.get('ndvi_mean_30d') is not None):
                ndvi_features['ndvi_trend_30d'] = round(
                    (ndvi_features['ndvi_mean_7d'] - ndvi_features['ndvi_mean_30d']) / 23, 4
                )
            else:
                ndvi_features['ndvi_trend_30d'] = None
            
            return ndvi_features
            
        except Exception as e:
            print(f"  GEE NDVI request failed: {str(e)}")
            return self._get_default_ndvi_features()
    
    def _get_default_ndvi_features(self) -> Dict[str, float]:
        """Return default NDVI features when GEE not available"""
        features = {}
        for window in ['7d', '14d', '30d', '60d']:
            features[f'ndvi_mean_{window}'] = np.nan
            features[f'ndvi_stddev_{window}'] = np.nan
            features[f'ndvi_max_{window}'] = np.nan
        features['ndvi_trend_30d'] = np.nan
        return features
    
    def calculate_gdd(self, temps: list, base_temp: float = 5.0) -> float:
        """
        Calculate Growing Degree Days (GDD).
        
        Args:
            temps: List of daily average temperatures
            base_temp: Base temperature for GDD calculation (default 5°C)
            
        Returns:
            Cumulative GDD
        """
        gdd = sum([max(0, temp - base_temp) for temp in temps if not np.isnan(temp)])
        return round(gdd, 2)
    
    def get_weather_data(self, latitude: float, longitude: float, 
                        bloom_date: datetime, days_before: int = 90) -> Dict[str, float]:
        """
        Get weather data from NASA POWER API for specified period before bloom.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            bloom_date: Date of bloom observation
            days_before: Number of days before bloom to fetch data
            
        Returns:
            Dictionary with weather features
        """
        # Check cache
        cache_key = f"{latitude:.4f},{longitude:.4f},{bloom_date.strftime('%Y-%m-%d')},{days_before}"
        if cache_key in self.weather_cache:
            return self.weather_cache[cache_key]
        
        try:
            start_date = bloom_date - timedelta(days=days_before)
            end_date = bloom_date - timedelta(days=1)  # Day before bloom
            
            params = {
                "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN",
                "community": "AG",
                "longitude": longitude,
                "latitude": latitude,
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "format": "JSON"
            }
            
            response = requests.get(self.nasa_power_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                parameters = data['properties']['parameter']
                
                # Extract daily values
                temps_avg = list(parameters['T2M'].values())
                temps_max = list(parameters['T2M_MAX'].values())
                temps_min = list(parameters['T2M_MIN'].values())
                precip = list(parameters['PRECTOTCORR'].values())
                humidity = list(parameters['RH2M'].values())
                solar = list(parameters['ALLSKY_SFC_SW_DWN'].values())
                
                # Calculate features for different time windows
                weather_features = {}
                
                for window in [7, 14, 30, 90]:
                    if len(temps_avg) >= window:
                        window_temps = temps_avg[-window:]
                        window_precip = precip[-window:]
                        window_humidity = humidity[-window:]
                        window_solar = solar[-window:]
                        
                        weather_features[f'temp_avg_{window}d'] = round(np.mean(window_temps), 2)
                        weather_features[f'temp_max_{window}d'] = round(np.max(temps_max[-window:]), 2)
                        weather_features[f'temp_min_{window}d'] = round(np.min(temps_min[-window:]), 2)
                        weather_features[f'precip_total_{window}d'] = round(np.sum(window_precip), 2)
                        weather_features[f'precip_avg_{window}d'] = round(np.mean(window_precip), 2)
                        weather_features[f'humidity_avg_{window}d'] = round(np.mean(window_humidity), 2)
                        weather_features[f'solar_avg_{window}d'] = round(np.mean(window_solar), 2)
                        
                        # Growing Degree Days
                        weather_features[f'gdd_{window}d'] = self.calculate_gdd(window_temps)
                
                # Overall statistics (full 90 days)
                weather_features['temp_variance_90d'] = round(np.var(temps_avg), 2)
                weather_features['temp_range_90d'] = round(np.max(temps_max) - np.min(temps_min), 2)
                weather_features['frost_days_90d'] = sum(1 for t in temps_min if t < 0)
                
                self.weather_cache[cache_key] = weather_features
                return weather_features
                
            else:
                print(f"  NASA POWER API error {response.status_code} for {latitude}, {longitude}")
                return self._get_default_weather_features()
                
        except Exception as e:
            print(f"  NASA POWER request failed: {str(e)}")
            return self._get_default_weather_features()
    
    def _get_default_weather_features(self) -> Dict[str, float]:
        """Return default weather features when API fails"""
        features = {}
        for window in [7, 14, 30, 90]:
            features[f'temp_avg_{window}d'] = np.nan
            features[f'temp_max_{window}d'] = np.nan
            features[f'temp_min_{window}d'] = np.nan
            features[f'precip_total_{window}d'] = np.nan
            features[f'precip_avg_{window}d'] = np.nan
            features[f'humidity_avg_{window}d'] = np.nan
            features[f'solar_avg_{window}d'] = np.nan
            features[f'gdd_{window}d'] = np.nan
        
        features['temp_variance_90d'] = np.nan
        features['temp_range_90d'] = np.nan
        features['frost_days_90d'] = np.nan
        
        return features
    
    def calculate_photoperiod_features(self, latitude: float, bloom_date: datetime) -> Dict[str, float]:
        """
        Calculate photoperiod features at bloom and 30 days before.
        
        Args:
            latitude: Latitude in degrees
            bloom_date: Date of bloom observation
            
        Returns:
            Dictionary with photoperiod features
        """
        date_30d_before = bloom_date - timedelta(days=30)
        
        photoperiod_bloom = self.calculate_photoperiod(latitude, bloom_date)
        photoperiod_30d = self.calculate_photoperiod(latitude, date_30d_before)
        
        # Rate of change in day length
        photoperiod_change_rate = (photoperiod_bloom - photoperiod_30d) / 30
        
        return {
            'photoperiod_at_bloom': photoperiod_bloom,
            'photoperiod_30d_before': photoperiod_30d,
            'photoperiod_change_rate': round(photoperiod_change_rate, 4)
        }
    
    def enrich_single_observation(self, row: pd.Series) -> Dict[str, float]:
        """
        Enrich a single bloom observation with all environmental features.
        
        Args:
            row: Pandas Series containing bloom observation data
            
        Returns:
            Dictionary with all engineered features
        """
        latitude = row['latitude']
        longitude = row['longitude']
        
        # Parse bloom date
        if 'bloom_date' in row:
            bloom_date = pd.to_datetime(row['bloom_date'])
        elif 'year' in row and 'day_of_year' in row:
            bloom_date = datetime(int(row['year']), 1, 1) + timedelta(days=int(row['day_of_year']) - 1)
        else:
            raise ValueError("Need either 'bloom_date' or 'year' + 'day_of_year' columns")
        
        features = {}
        
        # 1. Weather features (NASA POWER)
        print(f"   Fetching weather data for {latitude:.2f}, {longitude:.2f}...")
        weather_features = self.get_weather_data(latitude, longitude, bloom_date, days_before=90)
        features.update(weather_features)
        
        # 2. Soil properties (Google Earth Engine)
        print(f"   Fetching soil data...")
        soil_features = self.get_soil_properties(latitude, longitude)
        features.update(soil_features)
        
        # 3. NDVI vegetation index (Google Earth Engine)
        print(f"   Fetching NDVI vegetation data...")
        ndvi_features = self.get_ndvi_gee(latitude, longitude, bloom_date)
        features.update(ndvi_features)
        
        # 4. Elevation (Open-Elevation)
        print(f"   Fetching elevation...")
        elevation = self.get_elevation(latitude, longitude)
        features['elevation_m'] = elevation
        
        # 5. Photoperiod calculations
        print(f"    Calculating photoperiod...")
        photoperiod_features = self.calculate_photoperiod_features(latitude, bloom_date)
        features.update(photoperiod_features)
        
        return features
    
    def enrich_dataset(self, input_csv: str, output_csv: str = None, 
                      sample_size: int = None) -> pd.DataFrame:
        """
        Enrich entire bloom dataset with environmental features.
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output CSV file (optional)
            sample_size: Number of rows to process (for testing, optional)
            
        Returns:
            Enriched DataFrame
        """
        print(f"\n Loading bloom dataset from {input_csv}")
        df = pd.read_csv(input_csv)
        
        print(f" Dataset shape: {df.shape[0]} observations")
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            print(f" Sampling {sample_size} observations for testing...")
            df = df.sample(n=sample_size, random_state=42)
        
        # Initialize feature columns
        feature_list = []
        
        print(f"\n🔧 Starting feature engineering for {len(df)} observations...")
        print("=" * 70)
        
        for idx, row in df.iterrows():
            print(f"\n[{idx + 1}/{len(df)}] Processing observation {row.get('record_id', idx)}:")
            print(f"  Species: {row.get('scientific_name', 'Unknown')}")
            print(f"  Location: {row['latitude']:.2f}, {row['longitude']:.2f}")
            
            try:
                features = self.enrich_single_observation(row)
                feature_list.append(features)
                print(f"   Enriched with {len(features)} features")
                
            except Exception as e:
                print(f"   Error: {str(e)}")
                # Add empty features
                feature_list.append({})
            
            # Small delay to avoid overwhelming APIs
            if idx < len(df) - 1:
                time.sleep(0.5)
        
        print("\n" + "=" * 70)
        
        # Convert feature list to DataFrame
        features_df = pd.DataFrame(feature_list)
        
        # Combine original data with new features
        df_enriched = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        
        # Add target variable (bloom day of year)
        if 'day_of_year' in df_enriched.columns:
            df_enriched['bloom_day_of_year'] = df_enriched['day_of_year']
        
        print(f"\n Feature engineering complete!")
        print(f" Original features: {df.shape[1]}")
        print(f" New features added: {features_df.shape[1]}")
        print(f" Total features: {df_enriched.shape[1]}")
        
        # Display feature summary
        print("\n New features added:")
        for col in features_df.columns:
            non_null = features_df[col].notna().sum()
            pct = (non_null / len(features_df)) * 100
            print(f"   • {col}: {non_null}/{len(features_df)} ({pct:.1f}% complete)")
        
        # Save to CSV
        if output_csv is None:
            output_csv = self.processed_data_dir / 'bloom_features_ml.csv'
        
        df_enriched.to_csv(output_csv, index=False)
        print(f"\n Enriched dataset saved to: {output_csv}")
        
        # Save feature metadata
        metadata = {
            'total_observations': len(df_enriched),
            'total_features': len(df_enriched.columns),
            'original_features': len(df.columns),
            'engineered_features': len(features_df.columns),
            'feature_categories': {
                'weather': len([c for c in features_df.columns if 'temp' in c or 'precip' in c or 'humidity' in c or 'solar' in c or 'gdd' in c or 'frost' in c]),
                'soil': len([c for c in features_df.columns if 'soil' in c]),
                'elevation': 1 if 'elevation_m' in features_df.columns else 0,
                'photoperiod': len([c for c in features_df.columns if 'photoperiod' in c])
            },
            'completeness': {
                col: f"{(features_df[col].notna().sum() / len(features_df) * 100):.1f}%"
                for col in features_df.columns
            }
        }
        
        metadata_path = self.processed_data_dir / 'feature_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f" Metadata saved to: {metadata_path}")
        
        return df_enriched


def main():
    """Example usage of BloomFeatureEngineer"""
    
    print("\n" + "=" * 70)
    print(" BLOOMBLY FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    # Initialize feature engineer
    engineer = BloomFeatureEngineer()
    
    # Path to cleaned bloom data
    input_csv = engineer.processed_data_dir / 'clean_blooms_ml.csv'
    output_csv = engineer.processed_data_dir / 'bloom_features_ml.csv'
    
    # Check if input file exists
    if not input_csv.exists():
        print(f"\n Error: Input file not found: {input_csv}")
        print("\n Please run the data cleaning script first:")
        print("   python ml/src/clean_data.py")
        return
    
    print(f"\n Input:  {input_csv}")
    print(f" Output: {output_csv}")
    
    # Ask user about sample size
    print("\n" + "=" * 70)
    print("  WARNING: Full dataset processing will make ~8,000 API calls")
    print("   This may take several hours due to rate limiting.")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Process FULL dataset (recommended for production)")
    print("  2. Process SAMPLE (10 observations - for testing)")
    print("  3. Process SAMPLE (100 observations - for validation)")
    
    choice = input("\nEnter choice (1/2/3) or press Enter for testing sample [2]: ").strip()
    
    if choice == '1':
        sample_size = None
        confirm = input("\n  Process all observations? This will take hours. (yes/no): ")
        if confirm.lower() != 'yes':
            print(" Cancelled.")
            return
    elif choice == '3':
        sample_size = 100
    else:
        sample_size = 10
    
    # Run enrichment
    try:
        df_enriched = engineer.enrich_dataset(
            input_csv=input_csv,
            output_csv=output_csv,
            sample_size=sample_size
        )
        
        print("\n" + "=" * 70)
        print(" FEATURE ENGINEERING COMPLETE!")
        print("=" * 70)
        
        print(f"\n Dataset Summary:")
        print(f"   Total observations: {len(df_enriched)}")
        print(f"   Total features: {len(df_enriched.columns)}")
        
        print(f"\n Next steps:")
        print(f"   1. Inspect features: pandas.read_csv('{output_csv}')")
        print(f"   2. Train ML model: python ml/src/train.py")
        print(f"   3. Generate predictions: python ml/src/predict.py")
        
    except KeyboardInterrupt:
        print("\n\n  Process interrupted by user.")
        print("   Partial results may be saved.")
    except Exception as e:
        print(f"\n\n Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
