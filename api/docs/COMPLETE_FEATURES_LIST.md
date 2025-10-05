# Complete Model Features List - Bloom Prediction Model v2

## Overview

The bloom prediction model now includes **44 comprehensive features** covering all requested environmental variables and more. This document provides a complete breakdown of all features used in the model.

## Required Features ✅ (All Implemented)

### 1. ✅ NDVI Suavizado (Smoothed NDVI) - 3 features
- **`ndvi_smoothed_current`**: Current smoothed NDVI value (5-day moving average)
- **`ndvi_smoothed_mean`**: Mean smoothed NDVI over the analysis period
- **`ndvi_smoothed_trend`**: Trend/slope of smoothed NDVI time series

**Data Source**: MODIS MOD13Q1 (250m, 16-day composite)  
**Processing**: 5-day moving average to remove noise and reveal true vegetation growth trend

### 2. ✅ Temperatura Ambiental (Air Temperature) - 4 features
- **`temp_mean`**: Mean air temperature (°C)
- **`temp_max`**: Maximum air temperature (°C)
- **`temp_min`**: Minimum air temperature (°C)
- **`temp_range`**: Temperature range (max - min)

**Data Source**: MODIS MOD11A1 Land Surface Temperature (1km, daily)  
**Processing**: 30-day aggregation, converted from Kelvin to Celsius

### 3. ✅ Temperatura del Suelo (Soil Temperature) - 3 features
- **`soil_temp_mean`**: Mean soil temperature (°C)
- **`soil_gdd_current`**: Current day soil temperature GDD
- **`soil_gdd_accumulated_30d`**: 30-day accumulated soil temperature GDD

**Data Source**: MODIS LST (as proxy for soil temperature at 0-10cm depth)  
**Processing**: Damping factor applied (soil temp = LST × 0.7) to account for soil thermal inertia

### 4. ✅ Grados al Día de Crecimiento - Temperatura del Suelo (Soil Temperature GDD) - Included above
Uses Baskerville-Emin method with base temperature of 10°C:
```
Soil GDD = max(0, [(Tsoil_max + Tsoil_min) / 2] - 10)
```

### 5. ✅ Fecha de Inicio de Primavera (Spring Start Date) - 4 features
- **`spring_start_day`**: Day of year when spring begins (1-365)
- **`days_since_spring_start`**: Days elapsed since spring onset
- **`is_spring_active`**: Boolean flag indicating if currently in spring growth period
- **`winter_ndvi_baseline`**: Winter NDVI baseline for comparison

**Method**: 5-day smoothed NDVI analysis to detect sustained growth period  
**Algorithm**: Detects when smoothed NDVI exceeds winter baseline by 10% and maintains growth

### 6. ✅ Días de Agua Disponible en el Suelo (Soil Water Days) - 4 features
- **`soil_water_days`**: Days worth of plant-available water
- **`wilting_point`**: Permanent wilting point (%)
- **`water_stress`**: Boolean flag for water stress conditions
- **`available_water_ratio`**: Ratio of available water to field capacity (0-1)

**Data Source**: NASA SMAP SPL4SMGP (10km, 3-hourly)  
**Method**: Wilting point calculation: PWP = (FC × 0.74) - 5

### 7. ✅ Precipitación (Precipitation) - 2 features
- **`precip_total`**: Total precipitation over 30-day period (mm)
- **`precip_mean`**: Mean daily precipitation (mm/day)

**Data Source**: CHIRPS Daily (5km)  
**Processing**: 30-day aggregation

### 8. ✅ Textura del Suelo (Soil Texture) - 4 features
- **`soil_texture_code`**: Numeric soil texture code (0-8)
  - 0=sand, 1=loamy_sand, 2=sandy_loam, 3=loam, 4=silt_loam,
  - 5=sandy_clay, 6=clay_loam, 7=silty_clay, 8=clay
- **`sand_percent`**: Percentage of sand content (0-100)
- **`clay_percent`**: Percentage of clay content (0-100)
- **`silt_percent`**: Percentage of silt content (0-100)

**Data Source**: OpenLandMap SoilGrids (250m)  
**Classification**: USDA soil texture triangle

### 9. ✅ Evapotranspiración (Evapotranspiration) - 3 features
- **`et0_hargreaves`**: Reference ET calculated using Hargreaves equation (mm/day)
- **`et0_adjusted`**: Humidity-adjusted ET (mm/day)
- **`water_deficit_index`**: ET relative to available water (stress indicator)

**Data Source**: MODIS MOD16A2GF (500m, 8-day composite) + calculated from temperature  
**Method**: Hargreaves equation using extraterrestrial radiation, temperature range, and humidity

## Additional Features (Original Model Features)

### Spatial Features (2)
- **`lat`**: Latitude (decimal degrees)
- **`lon`**: Longitude (decimal degrees)

### Temporal Features (6)
- **`day_of_year`**: Day of year (1-365)
- **`month`**: Month (1-12)
- **`week_of_year`**: Week of year (1-52)
- **`day_sin`**: Sin-encoded day of year (cyclical)
- **`day_cos`**: Cos-encoded day of year (cyclical)
- **`days_from_species_mean`**: Days from species-specific mean bloom day

### Environmental - Current Conditions (3)
- **`ndvi_mean`**: Raw mean NDVI (not smoothed)
- **`ndvi_max`**: Maximum NDVI
- **`ndvi_trend`**: NDVI trend/slope

### Topography (1)
- **`elevation`**: Elevation above sea level (meters)

### Derived Features (3)
- **`growing_degree_days`**: Legacy GDD calculation (kept for compatibility)
- **`moisture_index`**: Precipitation / (Temperature + 20) ratio
- **`vegetation_health`**: NDVI × (1 + NDVI_trend)

### Air Temperature GDD (2)
- **`gdd_current`**: Current day air temperature GDD (base 0°C)
- **`gdd_accumulated_30d`**: 30-day accumulated air temperature GDD

## Complete Feature Count: 44

### Breakdown by Category:
1. **Spatial**: 2 features
2. **Temporal**: 6 features
3. **Air Temperature**: 4 features
4. **Precipitation**: 2 features
5. **NDVI (raw)**: 3 features
6. **NDVI (smoothed)**: 3 features ✨ NEW
7. **Topography**: 1 feature
8. **Spring Phenology**: 4 features
9. **Air Temperature GDD**: 2 features
10. **Soil Temperature & GDD**: 3 features ✨ NEW
11. **Soil Water Availability**: 4 features
12. **Soil Texture**: 4 features ✨ NEW
13. **Evapotranspiration**: 3 features ✨ NEW
14. **Derived Indices**: 3 features

**Total: 44 features**

## New Features Summary (23 new features added)

✨ **Smoothed NDVI** (3): Better vegetation trend detection  
✨ **Soil Temperature & GDD** (3): Root development indicators  
✨ **Soil Texture** (4): Water retention and nutrient availability  
✨ **Evapotranspiration** (3): Water stress analysis  

## Earth Engine Data Collections Used

| Variable | Collection | Resolution | Temporal | Band(s) |
|----------|-----------|------------|----------|---------|
| NDVI | MODIS/061/MOD13Q1 | 250m | 16-day | NDVI |
| Air Temperature | MODIS/061/MOD11A1 | 1km | Daily | LST_Day_1km, LST_Night_1km |
| Soil Temperature | MODIS/061/MOD11A1 | 1km | Daily | LST_Day_1km × 0.7 |
| Precipitation | UCSB-CHG/CHIRPS/DAILY | 5km | Daily | precipitation |
| Soil Moisture | NASA/SMAP/SPL4SMGP/007 | 10km | 3-hourly | sm_surface |
| Soil Texture | OpenLandMap SoilGrids | 250m | Static | Sand, Clay, Silt fractions |
| Evapotranspiration | MODIS/061/MOD16A2GF | 500m | 8-day | ET, PET |
| Elevation | USGS/SRTMGL1_003 | 30m | Static | elevation |

## Feature Importance for Bloom Prediction

Based on ecological significance:

### High Importance:
1. **Spring Start Date** - Critical bloom trigger for many species
2. **Soil GDD** - Root development and bloom initiation
3. **Smoothed NDVI** - Vegetation health and growth stage
4. **Water Stress** - Limits bloom development
5. **Soil Water Days** - Must be adequate for bloom support

### Medium Importance:
6. **Air Temperature GDD** - General development indicator
7. **Evapotranspiration** - Water demand/stress
8. **Precipitation** - Water supply
9. **Soil Texture** - Affects water and nutrient availability
10. **Day of Year** - Seasonal timing

### Supporting Features:
11. Spatial location (lat/lon)
12. Elevation
13. Temperature range
14. NDVI trend
15. Various derived indices

## Usage Example

```python
from app.bloom_predictor_v2 import ImprovedBloomPredictor

# Initialize predictor
predictor = ImprovedBloomPredictor(
    data_path='../backend/data.csv',
    use_earth_engine=True
)

# The model automatically collects all 44 features from Earth Engine
# and calculates derived features

# Make a prediction
prob = predictor.predict_bloom_probability(
    lat=40.7,
    lon=-74.0,
    date=datetime(2025, 6, 15),
    species='Symphyotrichum novae-angliae'
)

print(f"Bloom probability: {prob:.2%}")
```

## Model Performance Impact

With all 44 features:
- **Better seasonal timing**: Spring detection + temporal features
- **Improved water stress detection**: Soil water + ET + texture
- **Enhanced growth modeling**: Soil GDD + smoothed NDVI
- **More robust predictions**: Comprehensive environmental context

Expected improvements:
- ✅ Fewer false positives in drought conditions (ET + soil water)
- ✅ Better early/late spring prediction (spring start date)
- ✅ Improved regional accuracy (soil texture + elevation)
- ✅ More stable predictions (smoothed NDVI vs. noisy raw NDVI)

## Validation Requirements

After adding all new features, the model must be **retrained**:

```bash
cd api
python retrain_and_save_v2.py
```

This will:
1. Load historical bloom data
2. Generate negative examples
3. Calculate all 44 features
4. Train new model
5. Save to `bloom_model_v2.pkl`

## Future Enhancements

Potential additional features:
1. **Photoperiod** - Day length effects on blooming
2. **Frost Events** - Late spring frost impact
3. **Soil Nutrients** - Nitrogen/phosphorus levels
4. **Wind Speed** - Affects ET and pollination
5. **Solar Radiation** - Direct measurement vs. calculated
6. **Humidity** - Direct measurement for better ET
7. **Snow Cover** - Spring melt timing
8. **Land Cover** - Vegetation type classification

## References

- **Spring Phenology**: Moving average smoothing for vegetation indices
- **Growing Degree Days**: Baskerville-Emin (1969) method
- **Soil Water**: Permanent Wilting Point method
- **Evapotranspiration**: Hargreaves equation (simplified FAO-56)
- **Soil Texture**: USDA soil texture triangle classification

## License

Part of the Bloombly project (NASA Hack).

---

**Last Updated**: October 5, 2025  
**Model Version**: v2 with 44 features  
**Feature Count**: 44 (21 original + 23 new)
