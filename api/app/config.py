"""
Configuration for the BloomWatch API

IMPORTANT: Set these values in your .env file instead of here:
  EE_PROJECT=your-actual-project-id
  GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}

The values below are just fallback defaults.
"""

# Earth Engine (fallback values - use .env file instead!)
EE_PROJECT = None  # Will use environment variable EE_PROJECT
GOOGLE_APPLICATION_CREDENTIALS_JSON = None  # Will use environment variable

# Model
DEFAULT_PREDICTION_METHOD = 'v2'  # Use v2 model (bloom dynamics) by default
NUM_PREDICTIONS = 100  # Increased for v2 model (filters by confidence)

# AOI
DEFAULT_AOI_TYPE = 'global'
STATE_BOUNDS = {
    'Texas': {'min_lat': 25.8, 'max_lat': 36.5, 'min_lon': -106.6, 'max_lon': -93.5},
    'California': {'min_lat': 32.5, 'max_lat': 42.0, 'min_lon': -124.4, 'max_lon': -114.1},
    'Florida': {'min_lat': 24.5, 'max_lat': 31.0, 'min_lon': -87.6, 'max_lon': -79.8},
    'New York': {'min_lat': 40.5, 'max_lat': 45.0, 'min_lon': -79.8, 'max_lon': -71.8}
}
COUNTRY_BOUNDS = {
    'united states': {'min_lat': 24.4, 'max_lat': 49.4, 'min_lon': -125.0, 'max_lon': -66.9},
    'mexico': {'min_lat': 14.5, 'max_lat': 32.7, 'min_lon': -118.4, 'max_lon': -86.7}
}

# Prediction
MAX_TIME_SERIES_DAYS = 90
TIME_SERIES_INTERVAL_DAYS = 7

# Flask
PORT = 5001
DEBUG = True
