# Data Sources for Advanced Bloom Prediction Features

## Overview

The new features use **coordinate-based** environmental data retrieved from satellite observations and Earth observation datasets. The system has two modes:

1. **Primary Mode**: Google Earth Engine (real satellite data)
2. **Fallback Mode**: Synthetic data generation (when Earth Engine unavailable)

---

## Primary Data Sources (Google Earth Engine)

When Earth Engine is available, the system fetches **real satellite data** based on the coordinates (latitude/longitude) provided:

### 1. Spring Start Date Features

**Data Source: MODIS NDVI (Vegetation Index)**

| Property | Value |
|----------|-------|
| **Collection** | `MODIS/061/MOD13Q1` |
| **Band** | `NDVI` (Normalized Difference Vegetation Index) |
| **Satellite** | Terra MODIS |
| **Resolution** | 250 meters |
| **Temporal** | 16-day composite |
| **Coverage** | Global |
| **Time Range** | 90 days before target date |

**How it works:**
```python
# In earth_engine_utils.py -> get_ndvi_time_series()
ndvi_collection = ee.ImageCollection('MODIS/061/MOD13Q1') \
    .filterDate(start_date, end_date) \
    .filterBounds(ee.Geometry.Point([lon, lat])) \
    .select('NDVI')
```

**Data Retrieved:**
- Daily NDVI values for the past 90 days
- Extracted at the exact coordinates provided
- Used to calculate 5-day moving average
- Detects when vegetation "wakes up" in spring

**Example Query:**
- Location: 40.0°N, -100.0°W
- Date: May 15, 2024
- Returns: NDVI values from Feb 15 - May 15, 2024
- Result: ~90 data points showing vegetation greenness trend

---

### 2. Growing Degree Days (GDD) Features

**Data Source: MODIS Land Surface Temperature**

| Property | Value |
|----------|-------|
| **Collection** | `MODIS/061/MOD11A1` |
| **Bands** | `LST_Day_1km` (max temp), `LST_Night_1km` (min temp) |
| **Satellite** | Terra MODIS |
| **Resolution** | 1 kilometer |
| **Temporal** | Daily |
| **Coverage** | Global |
| **Time Range** | 30 days before target date |

**How it works:**
```python
# In earth_engine_utils.py -> get_temperature_time_series()
lst_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
    .filterDate(start_date, end_date) \
    .filterBounds(ee.Geometry.Point([lon, lat])) \
    .select(['LST_Day_1km', 'LST_Night_1km'])
```

**Data Retrieved:**
- Daily maximum temperature (daytime satellite pass)
- Daily minimum temperature (nighttime satellite pass)
- Converted from Kelvin to Celsius
- Used to calculate GDD using formula: `GDD = (Tmax + Tmin)/2 - 0°C`

**Example Query:**
- Location: 35.5°N, -95.0°W
- Date: April 10, 2024
- Returns: ~30 daily temperature pairs
- Result: Accumulated GDD = 450 (indicating heat energy available for growth)

---

### 3. Soil Water Availability Features

**Data Source: NASA SMAP Soil Moisture**

| Property | Value |
|----------|-------|
| **Collection** | `NASA/SMAP/SPL4SMGP/007` |
| **Band** | `sm_surface` (surface soil moisture) |
| **Satellite** | SMAP (Soil Moisture Active Passive) |
| **Resolution** | 10 kilometers |
| **Temporal** | 3-hourly |
| **Coverage** | Global |
| **Time Range** | 30 days before target date (averaged) |

**How it works:**
```python
# In earth_engine_utils.py -> get_soil_moisture_data()
smap = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007') \
    .filterDate(start_date, end_date) \
    .filterBounds(ee.Geometry.Point([lon, lat])) \
    .select('sm_surface')
```

**Data Retrieved:**
- Surface soil moisture (0-5cm depth)
- Volumetric water content (%)
- 30-day average to smooth daily variations
- Field capacity estimated from soil moisture range

**Example Query:**
- Location: 32.0°N, -110.0°W (Arizona)
- Date: June 1, 2024
- Returns: Soil moisture = 12%
- Calculation: Wilting point = 13.5%, so water_stress = True

---

## Fallback Data Sources (Synthetic Generation)

When Earth Engine is **not available** (no authentication, quota exceeded, or offline), the system generates synthetic data using:

### Climate Models Based on Coordinates

**Location-dependent calculations:**

```python
# In bloom_predictor_v2.py -> get_environmental_data_fallback()

# Temperature varies by latitude
base_temp = 15 - abs(lat - 35) * 0.4  # Cooler at higher/lower latitudes
seasonal_temp = 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

# Add spatial variation
lat_variation = np.sin(lat * 10) * 2
lon_variation = np.cos(lon * 10) * 2
```

**Fallback data includes:**

1. **NDVI Time Series** (Synthetic):
   - Generated using sine wave based on day of year
   - Low in winter (~0.2), high in summer (~0.7)
   - Varies by latitude (higher latitudes = shorter growing season)
   - Formula: `NDVI = 0.35 + 0.35 * sin(2π * (day - 80) / 365)`

2. **Temperature Time Series** (Synthetic):
   - Based on latitude (cooler at poles)
   - Seasonal variation using sine wave
   - Daily max/min with realistic 10°C diurnal range
   - Formula: `Temp = base_temp(lat) + seasonal_variation(day)`

3. **Soil Moisture** (Synthetic):
   - Estimated from precipitation patterns
   - Varies by season and location
   - Formula: `soil_moisture = 15 + precip_total / 5`

**Accuracy:**
- ✅ Good enough for general patterns
- ✅ Captures seasonal trends
- ❌ Less accurate than real satellite data
- ❌ Doesn't capture local anomalies (droughts, heat waves)

---

## Data Flow Diagram

```
User provides coordinates (lat, lon) + date
           ↓
     Earth Engine Available?
           ↓
    ┌──────┴──────┐
   YES            NO
    ↓              ↓
Real Satellite    Synthetic
    Data          Data
    ↓              ↓
    └──────┬──────┘
           ↓
   Feature Calculation
           ↓
     ┌─────┴─────┐
     ↓           ↓
  Spring      GDD & Soil
 Detection      Water
     ↓           ↓
     └─────┬─────┘
           ↓
   Bloom Prediction
```

---

## Specific Data Retrieval Examples

### Example 1: Texas Location

```python
lat = 30.5  # Texas
lon = -97.5
date = datetime(2024, 4, 15)

# This triggers:
# 1. NDVI query: Jan 15 - Apr 15, 2024 at (30.5, -97.5)
# 2. Temp query: Mar 15 - Apr 15, 2024 at (30.5, -97.5)
# 3. Soil moisture: Mar 15 - Apr 15, 2024 at (30.5, -97.5)

# Results in:
spring_start_day = 75  # Mid-March (real spring timing for Texas)
gdd_accumulated = 520  # Good heat accumulation
soil_water_days = 8.5  # Moderate water availability
```

### Example 2: California Location

```python
lat = 37.0  # Central California
lon = -120.0
date = datetime(2024, 3, 1)

# This triggers same queries but different location
# Results in:
spring_start_day = 60  # Early March (California has earlier spring)
gdd_accumulated = 380  # Moderate heat
soil_water_days = 2.0  # Lower (California can be drier)
```

---

## Code References

### Where Data is Fetched:

1. **`api/app/earth_engine_utils.py`**
   - Lines 151-198: `get_ndvi_time_series()` - Fetches MODIS NDVI
   - Lines 201-260: `get_temperature_time_series()` - Fetches MODIS LST
   - Lines 263-313: `get_soil_moisture_data()` - Fetches SMAP soil moisture
   - Lines 316-361: `get_comprehensive_environmental_data()` - Coordinates all data retrieval

2. **`api/app/bloom_predictor_v2.py`**
   - Lines 295-380: `get_environmental_data_ee()` - Calls Earth Engine
   - Lines 382-486: `get_environmental_data_fallback()` - Generates synthetic data
   - Lines 488-499: `get_environmental_data()` - Chooses EE or fallback

### Where Features are Calculated:

**`api/app/bloom_features.py`**
- Lines 25-133: `calculate_spring_start_date()` - Processes NDVI time series
- Lines 136-163: `calculate_growing_degree_days()` - Uses temperature data
- Lines 166-199: `calculate_accumulated_gdd()` - Sums GDD over period
- Lines 251-292: `calculate_soil_water_days()` - Uses soil moisture data

---

## Data Quality & Validation

### Real Data (Earth Engine):
✅ **Pros:**
- Actual satellite observations
- Global coverage
- Validated by NASA/USGS
- Captures real events (droughts, early springs, etc.)

⚠️ **Considerations:**
- Requires Earth Engine authentication
- Subject to quota limits
- May have gaps in cloud-covered areas (NDVI)
- 10km resolution for soil moisture (coarse)

### Synthetic Data (Fallback):
✅ **Pros:**
- Always available
- No authentication needed
- Fast computation
- Reasonable for general patterns

⚠️ **Limitations:**
- Doesn't capture anomalies
- Less accurate for specific predictions
- Based on climate normals, not actual conditions

---

## How to Enable Real Data

### Setup Earth Engine Authentication:

1. **Get Google Cloud Project:**
   ```bash
   # Visit: https://console.cloud.google.com/
   # Create project and enable Earth Engine API
   ```

2. **Set Environment Variables:**
   ```bash
   export EE_PROJECT="your-project-id"
   export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
   ```

3. **System automatically uses real data:**
   ```python
   # No code changes needed!
   # bloom_predictor_v2.py automatically detects EE availability
   ```

### Check Which Data Source is Being Used:

```python
from app.bloom_predictor_v2 import ImprovedBloomPredictor

predictor = ImprovedBloomPredictor(use_earth_engine=True)

if predictor.use_earth_engine:
    print("✓ Using real satellite data from Earth Engine")
else:
    print("⚠ Using synthetic fallback data")
```

---

## Summary

| Feature | Primary Data Source | Fallback Source | Resolution | Temporal |
|---------|-------------------|-----------------|------------|----------|
| **Spring Start** | MODIS NDVI | Synthetic sine wave | 250m | 16-day |
| **GDD** | MODIS LST | Climate model | 1km | Daily |
| **Soil Water** | SMAP Soil Moisture | Precipitation estimate | 10km | 3-hourly |

**All data is coordinate-based** - you provide (lat, lon, date) and the system automatically:
1. Queries the appropriate satellite collections
2. Extracts data at that exact location
3. Calculates the advanced features
4. Uses them for bloom prediction

The beauty of this system is that it works **globally** - any coordinates on Earth will return data! 🌍
