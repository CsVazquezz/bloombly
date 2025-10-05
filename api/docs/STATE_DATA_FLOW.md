# How State-Based Searches Pull Environmental Data

## Quick Answer: YES! ✅

When you search for a **state** (like "Texas" or "California"), the system:
1. Converts the state to bounding box coordinates
2. Samples random locations within that state
3. For **each location**, pulls real satellite data at those exact coordinates
4. Calculates all the new features (spring start, GDD, soil water) for each point
5. Returns bloom predictions with full environmental data

---

## Complete Data Flow Example

### Example: Searching for Texas Blooms on May 15, 2024

#### Step 1: User Request
```http
GET /api/predict/blooms?aoi_type=state&aoi_state=Texas&date=2024-05-15&method=v2
```

#### Step 2: System Looks Up State Bounds
```python
# From config.py
STATE_BOUNDS = {
    'Texas': {
        'min_lat': 25.8,  # South Texas (border)
        'max_lat': 36.5,  # Texas Panhandle
        'min_lon': -106.6, # West Texas (El Paso area)
        'max_lon': -93.5   # East Texas (Louisiana border)
    }
}
```

#### Step 3: System Generates Sample Locations
```python
# From bloom_predictor_v2.py -> predict_blooms_for_date()
# Lines ~757-770

# For Texas, generates ~100 random locations like:
Location 1: (29.5°N, -95.3°W)  # Houston area
Location 2: (32.8°N, -96.8°W)  # Dallas area  
Location 3: (30.3°N, -97.7°W)  # Austin area
Location 4: (31.8°N, -106.4°W) # El Paso area
# ... and 96 more random points across Texas
```

#### Step 4: For EACH Location, Fetch Real Satellite Data

**Location 1 Example: Houston Area (29.5°N, -95.3°W)**

```python
# System calls: get_environmental_data(29.5, -95.3, "2024-05-15")

# This triggers 3 Earth Engine queries:

# Query 1: NDVI Time Series (Spring Detection)
ndvi_query = {
    'collection': 'MODIS/061/MOD13Q1',
    'location': Point(-95.3, 29.5),
    'date_range': '2024-02-15 to 2024-05-15',  # 90 days
    'band': 'NDVI'
}
# Returns: [0.32, 0.35, 0.38, 0.42, ..., 0.68]  # 90 days of vegetation data
# Calculation: Spring started on day 72 (mid-March)

# Query 2: Temperature Time Series (GDD)
temp_query = {
    'collection': 'MODIS/061/MOD11A1',
    'location': Point(-95.3, 29.5),
    'date_range': '2024-04-15 to 2024-05-15',  # 30 days
    'bands': ['LST_Day_1km', 'LST_Night_1km']
}
# Returns: 
# Tmax: [28, 29, 27, 30, 31, 29, ...]  # 30 days of max temps
# Tmin: [18, 19, 17, 20, 21, 19, ...]  # 30 days of min temps
# Calculation: Accumulated GDD = 645

# Query 3: Soil Moisture (Water Availability)
soil_query = {
    'collection': 'NASA/SMAP/SPL4SMGP/007',
    'location': Point(-95.3, 29.5),
    'date_range': '2024-04-15 to 2024-05-15',  # 30 days
    'band': 'sm_surface'
}
# Returns: 24% soil moisture (averaged over 30 days)
# Calculation: 
#   - Wilting point = 13.5%
#   - Soil water days = 10.5
#   - Water stress = False
```

#### Step 5: Calculate New Features for This Location

```python
# Using the real satellite data from Step 4:

features_houston = {
    # Spring phenology
    'spring_start_day': 72,        # Mid-March (real Houston spring timing!)
    'days_since_spring_start': 64, # 64 days into spring
    'is_spring_active': True,      # Currently in spring
    'winter_ndvi_baseline': 0.28,  # Houston winter baseline
    
    # Growing Degree Days
    'gdd_current': 23.5,           # Today's heat accumulation
    'gdd_accumulated_30d': 645,    # Total heat over 30 days
    
    # Soil water
    'soil_water_days': 10.5,       # Good water availability
    'wilting_point': 13.5,         # Calculated for local soil
    'water_stress': False,         # No drought stress
    'available_water_ratio': 0.44  # 44% of field capacity
}
```

#### Step 6: Predict Bloom Probability

```python
# Model uses all 31 features (21 original + 10 new)
bloom_probability = model.predict([
    # Location
    29.5, -95.3,
    # Original features
    135, 5, 20, ...,  # day of year, month, week, temps, precip, etc.
    # NEW FEATURES from satellite data
    72, 64, 1, 0.28,  # Spring features
    23.5, 645,        # GDD features
    10.5, 13.5, 0, 0.44  # Soil water features
])

# Result: 0.73 (73% bloom probability)
```

#### Step 7: Repeat for All 100 Locations in Texas

The system does Steps 4-6 for each sampled location:
- Dallas area: Spring day 70, GDD 680, soil 8.2 → 68% probability
- Austin area: Spring day 71, GDD 655, soil 6.5 → 71% probability
- El Paso area: Spring day 65, GDD 710, soil 3.2 → 45% probability (drier)
- San Antonio: Spring day 72, GDD 670, soil 9.1 → 75% probability
- ... 96 more locations

#### Step 8: Filter and Return Top Predictions

```python
# System filters by confidence threshold (default 0.3)
# Sorts by probability
# Returns top 100 predictions

predictions = [
    {
        "type": "Feature",
        "properties": {
            "Site": "Symphyotrichum novae-angliae",
            "bloom_probability": 0.75,  # From model
            "environmental_factors": {
                "spring_start_day": 72,
                "gdd_accumulated": 670,
                "soil_water_days": 9.1,
                "water_stress": False
            }
        },
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[[-98.5, 29.4], ...]]  # San Antonio area
        }
    },
    # ... 99 more predictions across Texas
]
```

---

## Real Data vs State Bounds

### What Happens:

✅ **State defines WHERE to look** (bounding box)
```
Texas = area between (25.8°N to 36.5°N) × (-106.6°W to -93.5°W)
```

✅ **Satellite data pulled for SPECIFIC POINTS** within that area
```
Point 1: 29.5°N, -95.3°W → Get NDVI, temp, soil for THIS exact spot
Point 2: 32.8°N, -96.8°W → Get NDVI, temp, soil for THIS exact spot
Point 3: 30.3°N, -97.7°W → Get NDVI, temp, soil for THIS exact spot
... repeat 100 times
```

✅ **Each point gets REAL satellite data**
- NDVI from MODIS at 250m resolution
- Temperature from MODIS at 1km resolution  
- Soil moisture from SMAP at 10km resolution

---

## Code References

### Where State → Coordinates Conversion Happens:

**File: `api/app/routes/predict.py`**

```python
# Line 45-52
def get_aoi_bounds(aoi_type, aoi_state, aoi_country, bbox):
    """Returns the bounding box for a given AOI."""
    if aoi_type == 'state' and aoi_state in config.STATE_BOUNDS:
        return config.STATE_BOUNDS[aoi_state]  # ← Converts "Texas" to coords
    # ...
```

### Where Locations are Sampled:

**File: `api/app/bloom_predictor_v2.py`**

```python
# Lines ~757-770
for i, (species, bloom_info) in enumerate(self.species_bloom_windows.items()):
    # Sample locations within the AOI bounds
    n_samples = min(100, max(50, num_predictions // len(...)))
    
    # Generate random locations within AOI bounds
    candidate_lats = np.random.uniform(
        aoi_bounds['min_lat'],  # ← Uses Texas min_lat (25.8)
        aoi_bounds['max_lat'],  # ← Uses Texas max_lat (36.5)
        n_samples
    )
    candidate_lons = np.random.uniform(
        aoi_bounds['min_lon'],  # ← Uses Texas min_lon (-106.6)
        aoi_bounds['max_lon'],  # ← Uses Texas max_lon (-93.5)
        n_samples
    )
    
    # For each (lat, lon) pair:
    for lat, lon in zip(candidate_lats, candidate_lons):
        # ← THIS is where satellite data is fetched!
        probability = self.predict_bloom_probability(lat, lon, target_date, species)
```

### Where Satellite Data is Retrieved:

**File: `api/app/bloom_predictor_v2.py`**

```python
# Lines 488-499
def get_environmental_data(self, lat, lon, date):
    """Get environmental data with caching"""
    # ...
    if self.use_earth_engine:
        data = self.get_environmental_data_ee(lat, lon, date)  # ← REAL DATA
    else:
        data = self.get_environmental_data_fallback(lat, lon, date)  # Synthetic
```

**File: `api/app/earth_engine_utils.py`**

```python
# Lines 316-361
def get_comprehensive_environmental_data(lat, lon, date, lookback_days=90):
    """
    Get comprehensive environmental data for advanced bloom feature calculation.
    """
    # ... 
    # Get time series data
    ndvi_ts = get_ndvi_time_series(lat, lon, ...)      # ← Query 1
    temp_ts = get_temperature_time_series(lat, lon, ...) # ← Query 2
    soil_data = get_soil_moisture_data(lat, lon, ...)   # ← Query 3
```

---

## Performance Considerations

### How Many Queries Per State Search?

**For a single state search:**
- Sample locations: ~100 points
- Queries per location: 3 (NDVI, temp, soil)
- **Total queries: ~300 Earth Engine queries**

**Optimization:**
✅ **Caching**: Same location+date reuses data
✅ **Batch processing**: Earth Engine batches queries efficiently
✅ **Fallback mode**: If EE quota exceeded, uses synthetic data

### Approximate Response Time:

- **With Earth Engine**: 10-30 seconds (real satellite data)
- **Fallback mode**: 2-5 seconds (synthetic data)

---

## Try It Yourself

### API Request Examples:

**1. Texas with Real Satellite Data:**
```bash
curl "http://localhost:5001/api/predict/blooms?aoi_type=state&aoi_state=Texas&date=2024-05-15&method=v2"

# This will:
# ✓ Sample ~100 locations across Texas
# ✓ Fetch NDVI, temp, soil for EACH location
# ✓ Calculate spring start, GDD, soil water for EACH location
# ✓ Return predictions with environmental data
```

**2. California:**
```bash
curl "http://localhost:5001/api/predict/blooms?aoi_type=state&aoi_state=California&date=2024-04-15&method=v2"

# Different state → Different satellite data!
# California will have:
# - Earlier spring (day 55-65 vs Texas 70-75)
# - Different GDD (cooler coastal areas)
# - Lower soil moisture (drier climate)
```

**3. Check What Data Source is Used:**
```bash
curl "http://localhost:5001/api/predict/environmental?lat=30.27&lon=-97.74&date=2024-05-15"

# Response includes:
{
  "data_source": "earth_engine",  # ← Or "climate_normals" if fallback
  "environmental_data": {
    "ndvi_time_series": [...],  # Real satellite NDVI
    "tmax_series": [...],       # Real temperatures
    "soil_moisture": 24.5       # Real soil data
  }
}
```

---

## Summary Table

| Search Type | Coordinates Source | Satellite Data Fetched? | New Features Calculated? |
|-------------|-------------------|------------------------|-------------------------|
| **State** | STATE_BOUNDS → Random points | ✅ YES - for each point | ✅ YES - for each point |
| **Country** | COUNTRY_BOUNDS → Random points | ✅ YES - for each point | ✅ YES - for each point |
| **Bbox** | User-provided bounds → Random points | ✅ YES - for each point | ✅ YES - for each point |
| **Global** | Full world → Random points | ✅ YES - for each point | ✅ YES - for each point |

---

## Visual Flow:

```
User searches "Texas"
        ↓
Texas bounds: (25.8-36.5°N) × (-106.6--93.5°W)
        ↓
Sample 100 random locations
        ↓
    Location 1: (29.5°N, -95.3°W) → Query satellite → Calculate features
    Location 2: (32.8°N, -96.8°W) → Query satellite → Calculate features
    Location 3: (30.3°N, -97.7°W) → Query satellite → Calculate features
    ... × 97 more
        ↓
Filter by bloom probability > 0.3
        ↓
Return top 100 predictions with:
  - Real spring timing for each location
  - Real heat accumulation (GDD)
  - Real soil moisture conditions
```

---

## Bottom Line:

**YES! When you search by state, the system:**
1. ✅ Knows the geographic bounds of that state
2. ✅ Samples specific coordinate points within the state
3. ✅ Pulls REAL satellite data for each coordinate
4. ✅ Calculates all new features (spring, GDD, soil water) from that real data
5. ✅ Makes predictions based on actual environmental conditions

**The state is just a convenient way to define the search area - but the actual predictions use precise, location-specific satellite data!** 🛰️📍🌸
