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
    # United States
    'Texas': {'min_lat': 25.8, 'max_lat': 36.5, 'min_lon': -106.6, 'max_lon': -93.5},
    'California': {'min_lat': 32.5, 'max_lat': 42.0, 'min_lon': -124.4, 'max_lon': -114.1},
    'Florida': {'min_lat': 24.5, 'max_lat': 31.0, 'min_lon': -87.6, 'max_lon': -79.8},
    'New York': {'min_lat': 40.5, 'max_lat': 45.0, 'min_lon': -79.8, 'max_lon': -71.8},
    
    # Mexican States
    'Aguascalientes': {'min_lat': 21.6, 'max_lat': 22.4, 'min_lon': -102.8, 'max_lon': -101.8},
    'Baja California': {'min_lat': 28.0, 'max_lat': 32.7, 'min_lon': -117.1, 'max_lon': -112.5},
    'Baja California Sur': {'min_lat': 22.9, 'max_lat': 28.0, 'min_lon': -115.0, 'max_lon': -109.4},
    'Campeche': {'min_lat': 17.8, 'max_lat': 20.7, 'min_lon': -92.5, 'max_lon': -89.1},
    'Chiapas': {'min_lat': 14.5, 'max_lat': 17.9, 'min_lon': -94.1, 'max_lon': -90.4},
    'Chihuahua': {'min_lat': 25.5, 'max_lat': 31.8, 'min_lon': -109.1, 'max_lon': -103.3},
    'Coahuila': {'min_lat': 24.6, 'max_lat': 29.9, 'min_lon': -103.8, 'max_lon': -99.9},
    'Colima': {'min_lat': 18.7, 'max_lat': 19.6, 'min_lon': -104.7, 'max_lon': -103.5},
    'Durango': {'min_lat': 22.3, 'max_lat': 26.8, 'min_lon': -107.1, 'max_lon': -102.5},
    'Guanajuato': {'min_lat': 19.9, 'max_lat': 21.7, 'min_lon': -102.1, 'max_lon': -99.7},
    'Guerrero': {'min_lat': 16.6, 'max_lat': 18.8, 'min_lon': -102.2, 'max_lon': -98.0},
    'Hidalgo': {'min_lat': 19.8, 'max_lat': 21.4, 'min_lon': -99.5, 'max_lon': -97.9},
    'Jalisco': {'min_lat': 18.9, 'max_lat': 22.8, 'min_lon': -105.7, 'max_lon': -101.4},
    'Mexico': {'min_lat': 18.4, 'max_lat': 20.2, 'min_lon': -100.4, 'max_lon': -98.6},
    'Michoacan': {'min_lat': 18.1, 'max_lat': 20.4, 'min_lon': -103.7, 'max_lon': -100.0},
    'Morelos': {'min_lat': 18.4, 'max_lat': 19.2, 'min_lon': -99.6, 'max_lon': -98.6},
    'Nayarit': {'min_lat': 20.6, 'max_lat': 23.1, 'min_lon': -105.8, 'max_lon': -103.7},
    'Nuevo Leon': {'min_lat': 23.2, 'max_lat': 27.9, 'min_lon': -101.2, 'max_lon': -98.8},
    'Oaxaca': {'min_lat': 15.7, 'max_lat': 18.7, 'min_lon': -98.6, 'max_lon': -93.9},
    'Puebla': {'min_lat': 17.9, 'max_lat': 20.9, 'min_lon': -99.0, 'max_lon': -96.6},
    'Queretaro': {'min_lat': 20.0, 'max_lat': 21.7, 'min_lon': -100.5, 'max_lon': -99.0},
    'Quintana Roo': {'min_lat': 17.9, 'max_lat': 21.6, 'min_lon': -89.3, 'max_lon': -86.7},
    'San Luis Potosi': {'min_lat': 21.2, 'max_lat': 24.5, 'min_lon': -102.1, 'max_lon': -98.3},
    'Sinaloa': {'min_lat': 22.5, 'max_lat': 27.0, 'min_lon': -109.5, 'max_lon': -105.4},
    'Sonora': {'min_lat': 26.0, 'max_lat': 32.5, 'min_lon': -115.0, 'max_lon': -108.4},
    'Tabasco': {'min_lat': 17.3, 'max_lat': 18.7, 'min_lon': -94.1, 'max_lon': -91.0},
    'Tamaulipas': {'min_lat': 22.2, 'max_lat': 27.7, 'min_lon': -100.1, 'max_lon': -97.1},
    'Tlaxcala': {'min_lat': 19.1, 'max_lat': 19.8, 'min_lon': -98.8, 'max_lon': -97.6},
    'Veracruz': {'min_lat': 17.1, 'max_lat': 22.5, 'min_lon': -98.6, 'max_lon': -93.6},
    'Yucatan': {'min_lat': 19.5, 'max_lat': 21.6, 'min_lon': -90.5, 'max_lon': -87.5},
    'Zacatecas': {'min_lat': 21.0, 'max_lat': 25.1, 'min_lon': -104.4, 'max_lon': -101.0}
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
