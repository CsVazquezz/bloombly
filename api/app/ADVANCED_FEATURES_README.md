# Advanced Bloom Prediction Features

## Overview

This document describes the new ecological features added to the bloom prediction model to improve accuracy and better capture the biological mechanisms driving plant blooming events.

## New Features Implemented

### 1. Spring Start Date Detection (4 features)

Based on NDVI (Normalized Difference Vegetation Index) time series analysis to detect when spring vegetation growth begins.

**Method:**
- Uses 5-day moving average to smooth NDVI data and eliminate daily inconsistencies
- Calculates winter NDVI baseline (December 1 - March 1 average)
- Detects spring start when smoothed NDVI exceeds winter baseline and maintains sustained growth
- Identifies the longest consecutive growth period as the true spring onset

**Features:**
- `spring_start_day`: Day of year when spring vegetation growth begins (e.g., day 80 ≈ March 21)
- `days_since_spring_start`: Number of days elapsed since spring began
- `is_spring_active`: Boolean indicator of whether we're currently in the spring growth period
- `winter_ndvi_baseline`: Average NDVI value during winter dormancy

**Ecological Significance:**
- Spring timing is critical for bloom predictions as many plants bloom in response to spring conditions
- Earlier spring onset can trigger earlier blooming
- The strength of spring growth (NDVI rise) indicates favorable conditions for blooming

### 2. Growing Degree Days - GDD (2 features)

Implements the Baskerville-Emin (1969) method to measure heat accumulation for plant development.

**Method:**
```
GDD = [(Tmax + Tmin) / 2] - Tbase
```

Where:
- `Tmax`: Maximum daily temperature (°C)
- `Tmin`: Minimum daily temperature (°C)  
- `Tbase`: Base temperature threshold (0°C for general plant growth)

**Features:**
- `gdd_current`: Growing degree days for the current day
- `gdd_accumulated_30d`: Cumulative GDD over the past 30 days

**Ecological Significance:**
- GDD measures the amount of heat energy available for plant growth and development
- Higher GDD values = more growth potential and faster development
- Many plants require a minimum accumulated GDD threshold before blooming
- Example: A plant requiring 500 accumulated GDD will bloom earlier in a warm spring than a cool spring

### 3. Soil Water Availability (4 features)

Estimates plant-available water in soil using the permanent wilting point method.

**Method:**

1. Calculate Permanent Wilting Point (PWP):
   ```
   PWP% = (Field_Capacity% × 0.74) - 5
   ```

2. Calculate Available Water:
   ```
   If soil_moisture < PWP:
       soil_water_days = 0 (water stress)
   Else:
       soil_water_days = soil_moisture - PWP
   ```

**Features:**
- `soil_water_days`: Days worth of plant-available water in soil
- `wilting_point`: Soil moisture threshold below which plants cannot extract water
- `water_stress`: Boolean indicator of water stress conditions
- `available_water_ratio`: Ratio of available water to field capacity (0-1)

**Ecological Significance:**
- Water stress inhibits blooming in many plant species
- Plants require adequate soil moisture to support bloom development
- Different soil types have different water-holding capacities
- Wilting point varies by plant species (e.g., cacti vs. mesophytic plants)

## Feature Summary

| Category | Features | Count | Data Source |
|----------|----------|-------|-------------|
| Spring Phenology | spring_start_day, days_since_spring_start, is_spring_active, winter_ndvi_baseline | 4 | NDVI time series |
| Growing Degree Days | gdd_current, gdd_accumulated_30d | 2 | Temperature data |
| Soil Water | soil_water_days, wilting_point, water_stress, available_water_ratio | 4 | Soil moisture data |
| **Total New Features** | | **10** | |
| **Total Model Features** | (Original: 21) | **31** | |

## Data Sources

### Earth Engine Data Collections Used:

1. **NDVI Time Series:**
   - Collection: `MODIS/061/MOD13Q1`
   - Band: `NDVI`
   - Resolution: 250m
   - Temporal: 16-day composite
   - Used for spring detection

2. **Temperature Time Series:**
   - Collection: `MODIS/061/MOD11A1`
   - Bands: `LST_Day_1km`, `LST_Night_1km`
   - Resolution: 1km
   - Temporal: Daily
   - Used for GDD calculation

3. **Soil Moisture:**
   - Collection: `NASA/SMAP/SPL4SMGP/007`
   - Band: `sm_surface`
   - Resolution: 10km
   - Temporal: 3-hourly
   - Used for soil water availability

### Fallback Data:

When Earth Engine is unavailable, the system generates synthetic time series based on:
- Latitude/longitude-dependent climate models
- Seasonal patterns
- Spatial variation using trigonometric functions

## Implementation Files

1. **`app/bloom_features.py`** (NEW)
   - Core feature calculation functions
   - Can be used independently or with the model
   - Includes comprehensive testing examples

2. **`app/earth_engine_utils.py`** (UPDATED)
   - Added `get_ndvi_time_series()` - Fetches NDVI time series
   - Added `get_temperature_time_series()` - Fetches temperature data
   - Added `get_soil_moisture_data()` - Fetches soil moisture
   - Added `get_comprehensive_environmental_data()` - Unified data retrieval

3. **`app/bloom_predictor_v2.py`** (UPDATED)
   - Imports bloom feature calculation functions
   - Updated `get_environmental_data_ee()` - Includes time series
   - Updated `get_environmental_data_fallback()` - Generates synthetic time series
   - Updated `build_temporal_features()` - Calculates and includes new features
   - Updated `predict_bloom_probability()` - Uses all 31 features
   - Updated feature_columns list to include 10 new features

4. **`test_new_features.py`** (NEW)
   - Comprehensive test suite for all new features
   - Validates calculations with known results
   - Tests model integration

## Usage

### Testing the New Features

```bash
cd api
python test_new_features.py
```

This will:
- Test spring start date detection
- Test GDD calculations
- Test soil water availability
- Test comprehensive feature integration
- Check model compatibility

### Retraining the Model

To retrain the model with the new features:

```bash
cd api
python retrain_and_save_v2.py
```

This will:
1. Load historical bloom data
2. Generate negative examples
3. Calculate all 31 features (21 original + 10 new)
4. Train the model
5. Save to `bloom_model_v2.pkl`

### Using the Features Directly

```python
from app.bloom_features import calculate_comprehensive_bloom_features
import pandas as pd
from datetime import datetime

# Prepare environmental data
env_data = {
    'ndvi_time_series': [0.2, 0.25, 0.3, ...],  # 90 days of NDVI
    'dates': pd.date_range('2024-01-01', '2024-03-31'),
    'tmax': [20, 22, 24, ...],  # Daily max temps
    'tmin': [10, 12, 14, ...],  # Daily min temps
    'soil_moisture': 22,  # Current soil moisture %
    'field_capacity': 25  # Soil field capacity %
}

# Calculate all features
features = calculate_comprehensive_bloom_features(env_data)

# Access individual features
print(f"Spring starts on day {features['spring_start_day']}")
print(f"Current GDD: {features['gdd_current']}")
print(f"Soil water available: {features['soil_water_days']} days")
print(f"Water stress: {features['water_stress']}")
```

## Expected Improvements

### Model Performance

The new features should improve prediction accuracy by:

1. **Better Seasonal Timing** (Spring features)
   - Captures actual spring onset rather than calendar dates
   - Accounts for year-to-year variation in spring timing
   - Detects early vs. late springs

2. **Heat Accumulation Effects** (GDD features)
   - Models plant development based on heat energy
   - Predicts earlier blooming in warm years
   - Accounts for regional climate differences

3. **Water Stress Impacts** (Soil water features)
   - Identifies conditions unfavorable for blooming
   - Reduces false positives in drought conditions
   - Better models bloom success probability

### Use Cases Enhanced

- **Climate Change Analysis**: Track shifts in spring timing over years
- **Regional Predictions**: Better account for local soil and climate
- **Drought Impact**: Predict bloom suppression in water-limited areas
- **Early Warning**: Detect conditions leading to early or late blooms

## Scientific References

1. **Spring Phenology Detection:**
   - Moving average smoothing for vegetation indices
   - White et al. (2009) - Intercomparison of spring phenology extraction methods

2. **Growing Degree Days:**
   - Baskerville, G.L., and Emin, P. (1969). "Rapid Estimation of Heat Accumulation from Maximum and Minimum Temperatures"
   - Widely used in agriculture and ecology

3. **Soil Water Availability:**
   - Permanent Wilting Point method
   - Field capacity to wilting point relationship
   - Varies by soil texture and plant species

## Configuration

### Earth Engine Requirements

To use real Earth Engine data (recommended for production):

1. Set up Earth Engine authentication
2. Configure environment variables:
   ```bash
   export EE_PROJECT="your-project-id"
   export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
   ```

### Fallback Mode

If Earth Engine is unavailable, the system automatically uses:
- Synthetic time series based on climate normals
- Latitude/longitude-based variation
- Seasonal patterns
- Less accurate but provides reasonable estimates

## Troubleshooting

### Issue: Model prediction fails with feature mismatch

**Cause:** Model was trained with 21 features, but prediction uses 31 features

**Solution:** Retrain the model:
```bash
cd api
python retrain_and_save_v2.py
```

### Issue: NDVI time series empty

**Cause:** Earth Engine quota exceeded or authentication failed

**Solution:** System automatically falls back to synthetic data. Check logs for warnings.

### Issue: Soil moisture data unavailable

**Cause:** SMAP data not available for date/location

**Solution:** System uses estimated values based on precipitation and location.

## Future Enhancements

Potential additions:
1. **Photoperiod** - Day length effects on blooming
2. **Frost Events** - Late spring frost impact
3. **Nutrient Availability** - Nitrogen/phosphorus levels
4. **Biotic Factors** - Pollinator presence, competition
5. **Species-Specific Parameters** - Customized GDD thresholds per species

## License

This code is part of the Bloombly project (NasaHack).

## Contact

For questions or issues, please refer to the main project repository.
