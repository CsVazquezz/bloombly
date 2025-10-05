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
        self.appeears_url = "https://appeears.earthdatacloud.nasa.gov/api"
        
        # NASA AppEEARS credentials (optional, for NDVI)
        self.appeears_token = None
        self.appeears_username = os.getenv('APPEEARS_USERNAME')
        self.appeears_password = os.getenv('APPEEARS_PASSWORD')
        
        # Temporal gap-filling parameters
        self.max_temporal_gap = 30  # Maximum days to look back/forward for data
        self.temporal_windows = [0, 7, 14, 30]  # Days to try for gap filling
        
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
                    print(" Google Earth Engine initialized successfully with service account")
                else:
                    # Fallback to default authentication
                    ee.Initialize()
                    print(" Google Earth Engine initialized successfully")
                    
            except Exception as e:
                print(f" GEE initialization failed: {e}")
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
        Get soil properties from Google Earth Engine using OpenLandMap datasets.
        OpenLandMap provides global soil data with better coverage than SoilGrids.
        
        Available properties:
        - pH (H2O): Soil acidity/alkalinity
        - Clay content: g/kg
        - Sand content: g/kg  
        - Organic carbon: g/kg
        - Nitrogen: Not available in OpenLandMap (returns None)
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Dictionary with soil properties at 0-5cm and 5-15cm depths
        """
        # Check cache
        cache_key = f"{latitude:.4f},{longitude:.4f}"
        if cache_key in self.soil_cache:
            print(f"  (using cached soil data)")
            return self.soil_cache[cache_key]
        
        if not self.use_gee:
            print(f"  GEE not available, skipping soil data")
            return self._get_default_soil_features()
        
        try:
            # Create point geometry
            point = ee.Geometry.Point([longitude, latitude])
            
            soil_features = {}
            
            # Use OpenLandMap datasets (more reliable global coverage)
            # Reference: https://www.openlandmap.org/
            print(f"  Fetching soil properties from OpenLandMap...")
            
            try:
                # 1. pH in H2O (0-5cm and 5-15cm depths)
                # OpenLandMap: b0=0cm, b10=10cm, b30=30cm, b60=60cm, b100=100cm, b200=200cm
                ph_img = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")
                ph_values = ph_img.select(['b0', 'b10']).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=250,
                    bestEffort=True
                ).getInfo()
                
                # pH is stored as pH * 10, convert to actual pH
                ph_0cm = ph_values.get('b0')
                ph_10cm = ph_values.get('b10')
                
                soil_features['soil_ph_0-5cm'] = round(ph_0cm / 10, 2) if ph_0cm is not None else None
                soil_features['soil_ph_5-15cm'] = round(ph_10cm / 10, 2) if ph_10cm is not None else None
                print(f"    ✓ pH data retrieved")
            except Exception as e:
                print(f"    ✗ pH data failed: {str(e)[:80]}")
                soil_features['soil_ph_0-5cm'] = None
                soil_features['soil_ph_5-15cm'] = None
            
            try:
                # 2. Clay content (0-5cm and 5-15cm depths)
                clay_img = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02")
                clay_values = clay_img.select(['b0', 'b10']).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=250,
                    bestEffort=True
                ).getInfo()
                
                # Clay is stored as g/kg, values are already in the right range
                clay_0cm = clay_values.get('b0')
                clay_10cm = clay_values.get('b10')
                
                soil_features['soil_clay_0-5cm'] = round(clay_0cm, 2) if clay_0cm is not None else None
                soil_features['soil_clay_5-15cm'] = round(clay_10cm, 2) if clay_10cm is not None else None
                print(f"    ✓ Clay data retrieved")
            except Exception as e:
                print(f"    ✗ Clay data failed: {str(e)[:80]}")
                soil_features['soil_clay_0-5cm'] = None
                soil_features['soil_clay_5-15cm'] = None
            
            try:
                # 3. Sand content (0-5cm and 5-15cm depths)
                sand_img = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02")
                sand_values = sand_img.select(['b0', 'b10']).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=250,
                    bestEffort=True
                ).getInfo()
                
                # Sand is stored as g/kg
                sand_0cm = sand_values.get('b0')
                sand_10cm = sand_values.get('b10')
                
                soil_features['soil_sand_0-5cm'] = round(sand_0cm, 2) if sand_0cm is not None else None
                soil_features['soil_sand_5-15cm'] = round(sand_10cm, 2) if sand_10cm is not None else None
                print(f"    ✓ Sand data retrieved")
            except Exception as e:
                print(f"    ✗ Sand data failed: {str(e)[:80]}")
                soil_features['soil_sand_0-5cm'] = None
                soil_features['soil_sand_5-15cm'] = None
            
            try:
                # 4. Soil Organic Carbon (0-5cm and 5-15cm depths)
                soc_img = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02")
                soc_values = soc_img.select(['b0', 'b10']).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=250,
                    bestEffort=True
                ).getInfo()
                
                # SOC is stored as g/kg * 5
                soc_0cm = soc_values.get('b0')
                soc_10cm = soc_values.get('b10')
                
                soil_features['soil_organic_carbon_0-5cm'] = round(soc_0cm / 5, 2) if soc_0cm is not None else None
                soil_features['soil_organic_carbon_5-15cm'] = round(soc_10cm / 5, 2) if soc_10cm is not None else None
                print(f"    ✓ Organic carbon data retrieved")
            except Exception as e:
                print(f"    ✗ Organic carbon data failed: {str(e)[:80]}")
                soil_features['soil_organic_carbon_0-5cm'] = None
                soil_features['soil_organic_carbon_5-15cm'] = None
            
            try:
                # 5. Nitrogen (0-5cm and 5-15cm depths)
                nitrogen_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
                nitrogen_values = nitrogen_img.select(['b0', 'b10']).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=250,
                    bestEffort=True
                ).getInfo()
                
                # Note: OpenLandMap doesn't have direct nitrogen, using texture as proxy
                # This is a limitation - will be None for now
                soil_features['soil_nitrogen_0-5cm'] = None
                soil_features['soil_nitrogen_5-15cm'] = None
                print(f"    ⚠ Nitrogen data not available (using OpenLandMap)")
            except Exception as e:
                print(f"    ✗ Nitrogen data failed: {str(e)[:80]}")
                soil_features['soil_nitrogen_0-5cm'] = None
                soil_features['soil_nitrogen_5-15cm'] = None
            
            # Cache the results
            self.soil_cache[cache_key] = soil_features
            return soil_features
            
        except Exception as e:
            print(f"  ✗ GEE soil request failed: {str(e)}")
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
        
        Note: NDVI data only available from 2013 onwards (Landsat 8 launch date).
        
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
        
        # Check if date is before satellite data availability (Landsat 8: Feb 2013)
        if bloom_date.year < 2013:
            print(f"  ⚠ NDVI not available for {bloom_date.year} (satellite data starts in 2013)")
            return self._get_default_ndvi_features()
        
        try:
            # Create point geometry
            point = ee.Geometry.Point([longitude, latitude])
            
            # Create buffer for regional average (1km radius)
            region = point.buffer(1000)
            
            ndvi_features = {}
            
            # Define time windows before bloom - only 5d and 10d
            windows = {
                '5d': 5,
                '10d': 10
            }
            
            for window_name, days in windows.items():
                # Expand the date range to increase chances of finding imagery
                # Look further back to find cloud-free imagery
                start_date = (bloom_date - timedelta(days=days + 15)).strftime('%Y-%m-%d')
                end_date = (bloom_date - timedelta(days=max(1, days - 5))).strftime('%Y-%m-%d')
                
                ndvi_value = None
                
                try:
                    # Determine which satellite to use based on date
                    # Sentinel-2: June 2015 onwards (better resolution: 10m)
                    # Landsat 8/9: February 2013 onwards (30m resolution)
                    
                    use_sentinel = bloom_date.year > 2015 or (bloom_date.year == 2015 and bloom_date.month >= 6)
                    
                    if use_sentinel:
                        # Try Sentinel-2 first (10m resolution, 5-day revisit)
                        print(f"    Trying Sentinel-2 for {window_name} (date: {bloom_date.strftime('%Y-%m-%d')})...")
                        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                            .filterBounds(region) \
                            .filterDate(start_date, end_date) \
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                        
                        s2_count = s2.size().getInfo()
                        
                        if s2_count > 0:
                            print(f"    Found {s2_count} Sentinel-2 images for {window_name}")
                            
                            # Calculate NDVI for Sentinel-2
                            def calc_ndvi_s2(image):
                                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                                return image.addBands(ndvi)
                            
                            s2_ndvi = s2.map(calc_ndvi_s2).select('NDVI')
                            
                            # Get only mean NDVI
                            stats = s2_ndvi.mean().reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=region,
                                scale=10,
                                maxPixels=1e9,
                                bestEffort=True
                            ).getInfo()
                            
                            ndvi_value = stats.get('NDVI')
                            if ndvi_value is not None:
                                print(f"    ✓ Sentinel-2 NDVI {window_name}: {ndvi_value:.4f}")
                        else:
                            print(f"    No Sentinel-2 imagery found, trying Landsat...")
                    
                    # If no Sentinel-2 (or date is 2013-2015), use Landsat 8/9
                    if ndvi_value is None:
                        print(f"    Trying Landsat 8/9 for {window_name} (date: {bloom_date.strftime('%Y-%m-%d')})...")
                        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                            .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')) \
                            .filterBounds(region) \
                            .filterDate(start_date, end_date) \
                            .filter(ee.Filter.lt('CLOUD_COVER', 50))
                        
                        l8_count = l8.size().getInfo()
                        
                        if l8_count > 0:
                            print(f"    Found {l8_count} Landsat images for {window_name}")
                            
                            def calc_ndvi_l8(image):
                                ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
                                return image.addBands(ndvi)
                            
                            l8_ndvi = l8.map(calc_ndvi_l8).select('NDVI')
                            
                            stats = l8_ndvi.mean().reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=region,
                                scale=30,
                                maxPixels=1e9,
                                bestEffort=True
                            ).getInfo()
                            
                            ndvi_value = stats.get('NDVI')
                            if ndvi_value is not None:
                                print(f"    ✓ Landsat NDVI {window_name}: {ndvi_value:.4f}")
                        else:
                            print(f"    ✗ No Landsat imagery found for {window_name}")
                    
                    ndvi_features[f'ndvi_mean_{window_name}'] = ndvi_value
                    
                    if ndvi_value is None:
                        print(f"    ⚠ No NDVI data available for {window_name} (date range: {start_date} to {end_date})")

                
                except Exception as e:
                    print(f"    ✗ NDVI {window_name} error: {str(e)}")
                    ndvi_features[f'ndvi_mean_{window_name}'] = None
                
                # Small delay between windows
                time.sleep(0.3)
            
            return ndvi_features
            
        except Exception as e:
            print(f"  ✗ GEE NDVI request failed: {str(e)}")
            return self._get_default_ndvi_features()
    
    def _get_default_ndvi_features(self) -> Dict[str, float]:
        """Return default NDVI features when GEE not available"""
        features = {}
        for window in ['5d', '10d']:
            features[f'ndvi_mean_{window}'] = np.nan
        return features
    
    def authenticate_appeears(self) -> bool:
        """
        Authenticate with NASA AppEEARS API.
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not self.appeears_username or not self.appeears_password:
            print("  ⚠ AppEEARS credentials not found in .env file")
            print("     Add APPEEARS_USERNAME and APPEEARS_PASSWORD to enable AppEEARS")
            return False
        
        try:
            print(f"  🔐 Authenticating with AppEEARS as: {self.appeears_username}")
            
            response = requests.post(
                f"{self.appeears_url}/login",
                auth=(self.appeears_username, self.appeears_password)
            )
            
            if response.status_code == 200:
                self.appeears_token = response.json()['token']
                print(f"  ✓ AppEEARS authenticated successfully")
                print(f"     Token: {self.appeears_token[:20]}...")
                return True
            else:
                print(f"  ✗ AppEEARS authentication failed: HTTP {response.status_code}")
                print(f"  📋 Response:")
                try:
                    print(f"     {json.dumps(response.json(), indent=6)}")
                except:
                    print(f"     {response.text}")
                return False
                
        except Exception as e:
            print(f"  ✗ AppEEARS authentication error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_ndvi_appeears(self, latitude: float, longitude: float, bloom_date: datetime) -> Dict[str, float]:
        """
        Get NDVI from NASA AppEEARS (alternative to GEE).
        Uses MODIS or VIIRS data products.
        
        AppEEARS provides:
        - MODIS Terra/Aqua (2000-present, 250m-1km resolution)
        - VIIRS (2012-present, 375m resolution)
        - More reliable for historical queries than GEE
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            bloom_date: Date of bloom observation
            
        Returns:
            Dictionary with NDVI features at different time windows
        """
        # Check if we need to authenticate
        if not self.appeears_token:
            if not self.authenticate_appeears():
                print("  ⚠ Using GEE fallback for NDVI")
                return self.get_ndvi_gee(latitude, longitude, bloom_date)
        
        try:
            ndvi_features = {}
            
            # Use correct AppEEARS product names and layers
            # Reference: https://appeears.earthdatacloud.nasa.gov/products
            if bloom_date.year >= 2015:
                # MODIS Terra 16-day NDVI (most reliable, 250m resolution)
                product = "MOD13Q1.061"
                layer = "_250m_16_days_NDVI"
            elif bloom_date.year >= 2000:
                # MODIS Terra 16-day NDVI (1km resolution for older dates)
                product = "MOD13A2.061"
                layer = "_1_km_16_days_NDVI"
            else:
                print(f"  ⚠ No MODIS data before 2000 (date: {bloom_date.year})")
                return self.get_ndvi_gee(latitude, longitude, bloom_date)
            
            windows = {'5d': 5, '10d': 10}
            
            for window_name, days in windows.items():
                # AppEEARS uses point sampling, extend date range for better coverage
                start_date = (bloom_date - timedelta(days=days + 20)).strftime('%m-%d-%Y')
                end_date = (bloom_date - timedelta(days=max(1, days - 5))).strftime('%m-%d-%Y')
                
                # Create task request
                task = {
                    "task_type": "point",
                    "task_name": f"bloom_ndvi_{bloom_date.strftime('%Y%m%d')}_{window_name}",
                    "params": {
                        "dates": [{"startDate": start_date, "endDate": end_date}],
                        "layers": [{"product": product, "layer": layer}],
                        "coordinates": [{"latitude": latitude, "longitude": longitude, "id": "bloom_point"}]
                    }
                }
                
                headers = {"Authorization": f"Bearer {self.appeears_token}"}
                
                # Submit task
                print(f"    📤 Submitting AppEEARS task for {window_name}...")
                print(f"       Product: {product}, Dates: {start_date} to {end_date}")
                
                response = requests.post(
                    f"{self.appeears_url}/task",
                    json=task,
                    headers=headers
                )
                
                # AppEEARS returns 201 (Created) or 202 (Accepted) for successful task submission
                if response.status_code in [200, 201, 202]:
                    task_response = response.json()
                    task_id = task_response.get('task_id')
                    
                    if not task_id:
                        print(f"    ✗ No task_id in response: {task_response}")
                        ndvi_features[f'ndvi_mean_{window_name}'] = None
                        continue
                    
                    print(f"    ✓ Task submitted successfully (ID: {task_id})")
                    
                    # Poll for task completion (with timeout)
                    max_wait = 120  # seconds (increased for AppEEARS processing time)
                    wait_time = 0
                    poll_interval = 3  # seconds
                    
                    while wait_time < max_wait:
                        status_response = requests.get(
                            f"{self.appeears_url}/task/{task_id}",
                            headers=headers
                        )
                        
                        if status_response.status_code == 200:
                            task_status = status_response.json()
                            status = task_status.get('status')
                            print(f"    ⏳ Task status: {status} (waited {wait_time}s)")
                            
                            if status == 'done':
                                print(f"    ✓ Task completed, fetching results...")
                                # Download results
                                result_response = requests.get(
                                    f"{self.appeears_url}/bundle/{task_id}",
                                    headers=headers
                                )
                                
                                if result_response.status_code == 200:
                                    data = result_response.json()
                                    # Extract NDVI mean from results
                                    ndvi_values = [r['value'] for r in data.get('files', []) if r.get('value')]
                                    if ndvi_values:
                                        # MODIS/VIIRS NDVI scaled by 10000, convert to -1 to 1 range
                                        ndvi_mean = np.mean([v / 10000.0 for v in ndvi_values])
                                        ndvi_features[f'ndvi_mean_{window_name}'] = ndvi_mean
                                        print(f"    ✓ AppEEARS NDVI {window_name}: {ndvi_mean:.4f}")
                                    else:
                                        ndvi_features[f'ndvi_mean_{window_name}'] = None
                                break
                            elif status == 'error':
                                print(f"    ✗ AppEEARS task error for {window_name}")
                                print(f"       Task details: {task_status}")
                                ndvi_features[f'ndvi_mean_{window_name}'] = None
                                break
                        else:
                            print(f"    ⚠ Status check failed: HTTP {status_response.status_code}")
                        
                        time.sleep(poll_interval)
                        wait_time += poll_interval
                    
                    if wait_time >= max_wait:
                        print(f"    ⚠ AppEEARS timeout for {window_name}")
                        ndvi_features[f'ndvi_mean_{window_name}'] = None
                else:
                    # Detailed error logging
                    print(f"    ✗ AppEEARS task submission failed: HTTP {response.status_code}")
                    print(f"    📋 Error details:")
                    try:
                        error_data = response.json()
                        print(f"       {json.dumps(error_data, indent=8)}")
                    except:
                        print(f"       Response text: {response.text}")
                    print(f"    📋 Request sent:")
                    print(f"       URL: {self.appeears_url}/task")
                    print(f"       Task payload: {json.dumps(task, indent=8)}")
                    ndvi_features[f'ndvi_mean_{window_name}'] = None
            
            return ndvi_features
            
        except Exception as e:
            print(f"  ✗ AppEEARS NDVI request failed: {str(e)}")
            print(f"  📋 Exception traceback:")
            import traceback
            traceback.print_exc()
            print("  ⚠ Falling back to GEE for NDVI")
            return self.get_ndvi_gee(latitude, longitude, bloom_date)
    
    def get_ndvi_with_temporal_fallback(self, latitude: float, longitude: float, bloom_date: datetime, 
                                        use_appeears: bool = True) -> Dict[str, float]:
        """
        Get NDVI with temporal gap-filling.
        If no data available for exact date, tries nearby dates within max_temporal_gap.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            bloom_date: Original bloom date
            use_appeears: Whether to use AppEEARS (True) or GEE (False)
            
        Returns:
            Dictionary with NDVI features, filled from nearby dates if needed
        """
        print(f"  Fetching NDVI with temporal fallback...")
        
        # Try exact date first
        if use_appeears:
            ndvi_features = self.get_ndvi_appeears(latitude, longitude, bloom_date)
        else:
            ndvi_features = self.get_ndvi_gee(latitude, longitude, bloom_date)
        
        # Check if we got valid data
        has_data = any(v is not None and not np.isnan(v) for v in ndvi_features.values())
        
        if has_data:
            return ndvi_features
        
        # Try temporal windows if no data found
        print(f"    ⚠ No NDVI for exact date, trying nearby dates...")
        
        for gap_days in self.temporal_windows[1:]:  # Skip 0 (already tried)
            # Try earlier date
            earlier_date = bloom_date - timedelta(days=gap_days)
            print(f"    Trying {gap_days} days earlier ({earlier_date.strftime('%Y-%m-%d')})...")
            
            if use_appeears:
                temp_features = self.get_ndvi_appeears(latitude, longitude, earlier_date)
            else:
                temp_features = self.get_ndvi_gee(latitude, longitude, earlier_date)
            
            if any(v is not None and not np.isnan(v) for v in temp_features.values()):
                print(f"    ✓ Found data from {gap_days} days earlier")
                return temp_features
            
            # Try later date
            later_date = bloom_date + timedelta(days=gap_days)
            print(f"    Trying {gap_days} days later ({later_date.strftime('%Y-%m-%d')})...")
            
            if use_appeears:
                temp_features = self.get_ndvi_appeears(latitude, longitude, later_date)
            else:
                temp_features = self.get_ndvi_gee(latitude, longitude, later_date)
            
            if any(v is not None and not np.isnan(v) for v in temp_features.values()):
                print(f"    ✓ Found data from {gap_days} days later")
                return temp_features
        
        print(f"    ✗ No NDVI data found within {self.max_temporal_gap} days")
        return self._get_default_ndvi_features()
    
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
        
        # 3. NDVI vegetation index (GEE with temporal gap-filling - AppEEARS too slow)
        print(f"   Fetching NDVI vegetation data...")
        # Use GEE by default (AppEEARS is 40x slower)
        # Set use_appeears=True to enable AppEEARS (requires 1-3 min per observation)
        use_appeears = False  # Change to True if you want AppEEARS (much slower but slightly better coverage)
        ndvi_features = self.get_ndvi_with_temporal_fallback(
            latitude, longitude, bloom_date, use_appeears=use_appeears
        )
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
        Only processes historical observations (is_prediction=False) from 2013 onwards.
        This ensures all observations have NDVI satellite data available.
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output CSV file (optional)
            sample_size: Number of rows to process (for testing, optional)
            
        Returns:
            Enriched DataFrame with observations from 2013 onwards only
        """
        print(f"\n Loading bloom dataset from {input_csv}")
        df = pd.read_csv(input_csv)
        
        print(f" Dataset shape: {df.shape[0]} total observations")
        
        # Create bloom_date from year and day_of_year if it doesn't exist
        if 'bloom_date' not in df.columns:
            if 'year' in df.columns and 'day_of_year' in df.columns:
                # Convert year and day_of_year to datetime (convert to int first to remove decimals)
                df['bloom_date'] = pd.to_datetime(
                    df['year'].astype(int).astype(str) + '-' + df['day_of_year'].astype(int).astype(str), 
                    format='%Y-%j'
                )
                print(f" ✓ Created bloom_date column from year and day_of_year")
            else:
                raise ValueError("Cannot create bloom_date: missing 'year' or 'day_of_year' columns")
        else:
            # Convert bloom_date to datetime if needed
            df['bloom_date'] = pd.to_datetime(df['bloom_date'])
        
        # Filter for only historical observations (is_prediction=False) AND from 2013 onwards
        if 'is_prediction' in df.columns:
            df_predictions = df[df['is_prediction'] == True].copy()
            df_historical = df[df['is_prediction'] == False].copy()
            
            # Additional filter: only keep observations from 2013 onwards (NDVI availability)
            df_historical_with_ndvi = df_historical[df_historical['bloom_date'].dt.year >= 2013].copy()
            df_historical_before_2013 = df_historical[df_historical['bloom_date'].dt.year < 2013].copy()
            
            print(f" Splitting data:")
            print(f"   • Historical observations (is_prediction=False): {len(df_historical):,}")
            print(f"     - From 2013 onwards (WITH NDVI): {len(df_historical_with_ndvi):,} ✓")
            print(f"     - Before 2013 (NO NDVI): {len(df_historical_before_2013):,} ✗ (excluded)")
            print(f"   • Predictions for validation (is_prediction=True): {len(df_predictions):,} (excluded)")
            print(f"\n ✓ Only enriching observations from 2013+ with satellite data availability")
            print(f" ✓ Output will contain {len(df_historical_with_ndvi):,} observations\n")
            
            df_to_enrich = df_historical_with_ndvi
        else:
            print(f" Warning: 'is_prediction' column not found")
            # Still filter by date if no is_prediction column
            if 'bloom_date' in df.columns:
                df_to_enrich = df[df['bloom_date'].dt.year >= 2013].copy()
                print(f" Filtered to {len(df_to_enrich):,} observations from 2013 onwards")
            else:
                df_to_enrich = df
            df_predictions = pd.DataFrame()
        
        # Sample if requested (only from historical data)
        if sample_size and sample_size < len(df_to_enrich):
            print(f" Sampling {sample_size} historical observations for testing...")
            df_to_enrich = df_to_enrich.sample(n=sample_size, random_state=42)
        
        # Initialize feature columns
        feature_list = []
        
        print(f"\n🔧 Starting feature engineering for {len(df_to_enrich)} historical observations...")
        print("=" * 70)
        
        for idx, row in df_to_enrich.iterrows():
            print(f"\n[{idx + 1}/{len(df_to_enrich)}] Processing observation {row.get('record_id', idx)}:")
            print(f"  Species: {row.get('scientific_name', 'Unknown')}")
            print(f"  Location: {row['latitude']:.2f}, {row['longitude']:.2f}")
            print(f"  Year: {row.get('year', 'Unknown')}")
            
            try:
                features = self.enrich_single_observation(row)
                feature_list.append(features)
                print(f"   ✓ Enriched with {len(features)} features")
                
            except Exception as e:
                print(f"   ✗ Error: {str(e)}")
                # Add empty features
                feature_list.append({})
            
            # Small delay to avoid overwhelming APIs
            time.sleep(0.5)
        
        print("\n" + "=" * 70)
        
        # Convert feature list to DataFrame
        features_df = pd.DataFrame(feature_list)
        
        # Combine original historical data with new features
        df_enriched_historical = pd.concat([df_to_enrich.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        
        # Add target variable (bloom day of year)
        if 'day_of_year' in df_enriched_historical.columns:
            df_enriched_historical['bloom_day_of_year'] = df_enriched_historical['day_of_year']
        
        # No need to combine with predictions - only output 2013+ enriched data
        print(f"\n📊 Final dataset summary:")
        print(f"   • Total enriched observations: {len(df_enriched_historical):,}")
        print(f"   • Date range: {df_enriched_historical['bloom_date'].min()} to {df_enriched_historical['bloom_date'].max()}")
        print(f"   • All observations have satellite data (2013+)")
        
        df_final = df_enriched_historical
        
        print(f"\n✅ Feature engineering complete!")
        print(f" Original features: {df.shape[1]}")
        print(f" New features added: {features_df.shape[1]}")
        print(f" Total features: {df_final.shape[1]}")
        
        # Display feature summary (for enriched historical data only)
        print("\n📋 New features added (for historical data only):")
        for col in features_df.columns:
            non_null = features_df[col].notna().sum()
            pct = (non_null / len(features_df)) * 100
            print(f"   • {col}: {non_null}/{len(features_df)} ({pct:.1f}% complete)")
        
        # Save to CSV (ONLY historical data for training, exclude predictions)
        if output_csv is None:
            output_csv = self.processed_data_dir / 'bloom_features_ml.csv'
        
        df_enriched_historical.to_csv(output_csv, index=False)
        print(f"\n💾 Enriched data (2013+ only) saved to: {output_csv}")
        print(f"   📊 Total observations: {len(df_enriched_historical):,}")
        print(f"   � Date range: 2013 onwards (satellite data availability)")
        print(f"   📝 Note: Pre-2013 data and predictions excluded from output")
        
        # Save feature metadata
        metadata = {
            'total_observations': len(df_final),
            'date_range': '2013 onwards (NDVI satellite data available)',
            'excluded_observations': {
                'pre_2013': 'Historical observations before 2013 (no satellite data)',
                'predictions': 'is_prediction=True observations (for validation)'
            },
            'total_features': len(df_final.columns),
            'original_features': len(df.columns),
            'engineered_features': len(features_df.columns),
            'feature_categories': {
                'weather': len([c for c in features_df.columns if 'temp' in c or 'precip' in c or 'humidity' in c or 'solar' in c or 'gdd' in c or 'frost' in c]),
                'soil': len([c for c in features_df.columns if 'soil' in c]),
                'ndvi': len([c for c in features_df.columns if 'ndvi' in c]),
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
        
        print(f" 📄 Metadata saved to: {metadata_path}")
        
        return df_enriched_historical


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
