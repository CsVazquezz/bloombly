# Bloom Prediction Model Architecture Comparison

## v1 Model (BROKEN) 🗑️

```
┌─────────────────────────────────────────────────────────────────┐
│                     v1 MODEL - BROKEN APPROACH                  │
└─────────────────────────────────────────────────────────────────┘

INPUT: Location (lat, lon), Date, Environmental Data
                            ↓
        ┌───────────────────────────────────────┐
        │  TRAINING DATA (BROKEN)               │
        │  ✗ 1,179 bloom observations           │
        │  ✗ 0 non-bloom observations           │
        │  ✗ All labeled: bloom = 1             │
        │  ✗ Model never sees "no bloom"        │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  FEATURES (BASIC)                     │
        │  - lat, lon, day_of_year, month       │
        │  - temperature (single point)         │
        │  - precipitation (single point)       │
        │  - ndvi (single point)                │
        │  - elevation                          │
        │  Total: 9 features                    │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  MODEL (WRONG TASK!)                  │
        │  Gradient Boosting Classifier         │
        │  Target: SPECIES (not bloom!)         │
        │  ✗ Learns species classification      │
        │  ✗ NOT bloom prediction               │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  PREDICTION (FAKE)                    │
        │  species_prob = model.predict(X)      │
        │  bloom_prob = species_prob × Math     │
        │  ✗ Arbitrary environmental formulas   │
        │  ✗ NOT learned from data              │
        └───────────────────────────────────────┘
                            ↓
OUTPUT: "Bloom probability" = Species prob × random math
        ✗ NOT a real bloom prediction
        ✗ Confidence score is meaningless


┌─────────────────────────────────────────────────────────────────┐
│                    WHY v1 IS TRASH                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. Predicts species, not blooms                                 │
│ 2. Only trains on positives (can't learn "no bloom")            │
│ 3. "Bloom probability" is made-up math, not ML prediction       │
│ 4. Random location sampling across entire AOI                   │
│ 5. No temporal dynamics learned                                 │
│ 6. Validation metrics are for species classification            │
└─────────────────────────────────────────────────────────────────┘
```

## v2 Model (LEARNS BLOOM DYNAMICS) ✅

```
┌─────────────────────────────────────────────────────────────────┐
│                v2 MODEL - LEARNS BLOOM DYNAMICS                 │
└─────────────────────────────────────────────────────────────────┘

INPUT: Location (lat, lon), Date, Species
                            ↓
        ┌───────────────────────────────────────┐
        │  TRAINING DATA (CORRECT!)             │
        │  ✓ 1,179 bloom observations           │
        │  ✓ ~3,000 non-bloom observations      │
        │                                       │
        │  Negatives generated from:            │
        │  • Same locations ±60-90 days         │
        │  • Random locations, off-season       │
        │                                       │
        │  ✓ Model learns bloom vs no-bloom!    │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  FEATURE ENGINEERING (RICH!)          │
        │                                       │
        │  SPATIAL (2):                         │
        │  • lat, lon                           │
        │                                       │
        │  TEMPORAL (6):                        │
        │  • day_of_year, month, week           │
        │  • day_sin, day_cos (cyclical)        │
        │  • days_from_species_mean             │
        │                                       │
        │  ENVIRONMENTAL (10):                  │
        │  • temp_mean_30d, temp_max, temp_min  │
        │  • temp_range                         │
        │  • precip_total_30d, precip_mean      │
        │  • ndvi_mean_30d, ndvi_max            │
        │  • ndvi_trend ← DYNAMICS!             │
        │  • elevation                          │
        │                                       │
        │  DERIVED (3):                         │
        │  • growing_degree_days                │
        │  • moisture_index                     │
        │  • vegetation_health                  │
        │                                       │
        │  Total: 21 engineered features        │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  MODEL (CORRECT TASK!)                │
        │  Gradient Boosting Classifier         │
        │  Target: BLOOM (binary: 0 or 1)       │
        │  ✓ Learns bloom vs no-bloom           │
        │  ✓ Captures temporal patterns         │
        │  ✓ Learns environmental relationships │
        │                                       │
        │  Hyperparameters:                     │
        │  • 200 estimators                     │
        │  • max_depth=5                        │
        │  • learning_rate=0.05                 │
        │  • subsample=0.8                      │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  VALIDATION (TIME-SERIES CV)          │
        │  ✓ TimeSeriesSplit (5 folds)          │
        │  ✓ Train on past, test on future      │
        │  ✓ Metrics: ROC-AUC, Precision,       │
        │    Recall, F1                         │
        │  ✓ Realistic performance estimates    │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │  PREDICTION (REAL ML!)                │
        │  bloom_prob = model.predict_proba(X)  │
        │  ✓ Calibrated probability from ML     │
        │  ✓ Based on learned patterns          │
        │  ✓ Reflects actual bloom likelihood   │
        └───────────────────────────────────────┘
                            ↓
OUTPUT: Bloom probability (0.0-1.0)
        ✓ Real ML prediction from trained model
        ✓ Confidence score is calibrated
        ✓ Based on learned bloom dynamics


┌─────────────────────────────────────────────────────────────────┐
│                    WHY v2 LEARNS DYNAMICS                       │
├─────────────────────────────────────────────────────────────────┤
│ 1. ✓ Predicts bloom probability directly                        │
│ 2. ✓ Trains on positives AND negatives                          │
│ 3. ✓ Learns favorable vs unfavorable conditions                 │
│ 4. ✓ Captures temporal patterns (seasonality)                   │
│ 5. ✓ Models environmental trends (NDVI slope, temp changes)     │
│ 6. ✓ 30-day aggregations capture accumulation effects           │
│ 7. ✓ Cyclical encoding handles seasonal wraparound              │
│ 8. ✓ Time-series validation ensures future prediction quality   │
│ 9. ✓ Intelligent spatial sampling near historical blooms        │
│ 10. ✓ Validation metrics measure actual bloom prediction        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Differences Visualized

### What Each Model Actually Learns

```
v1 Model Learning:
    "Which species is likely at this location?"
    
    Examples learned:
    ✗ (lat=41.5, lon=-74.0, day=150) → Symphyotrichum novae-angliae (67%)
    ✗ (lat=41.5, lon=-74.0, day=200) → Symphyotrichum ericoides (53%)
    
    Then GUESSES bloom probability by multiplying by arbitrary factors
    ❌ NOT learning bloom dynamics!


v2 Model Learning:
    "Is a bloom likely at this location/date given conditions?"
    
    Examples learned:
    ✓ (lat=41.5, lon=-74.0, day=150, temp=22°C, ndvi↑, precip=85mm)
      → BLOOM (prob=0.85)
      
    ✓ (lat=41.5, lon=-74.0, day=50, temp=8°C, ndvi↓, precip=30mm)
      → NO BLOOM (prob=0.05)
      
    ✓ (lat=41.5, lon=-74.0, day=150, temp=22°C, ndvi↓, precip=10mm)
      → NO BLOOM (prob=0.15)  ← Same timing, but conditions unfavorable
    
    ✅ ACTUALLY learning what causes blooms!
```

### Training Data Comparison

```
v1: Only Positives (BROKEN)
┌────────────────────────────┐
│ Bloom Events: 1,179        │
│ Non-Bloom Events: 0        │
│                            │
│ ✗ Can't learn what         │
│   "no bloom" looks like    │
└────────────────────────────┘

    All observations labeled bloom=1
    No concept of unfavorable conditions
    Model has nothing to contrast against


v2: Positives + Negatives (CORRECT)
┌────────────────────────────┐
│ Bloom Events: 1,179        │
│ Non-Bloom Events: ~3,000   │
│                            │
│ ✓ Learns favorable vs      │
│   unfavorable conditions   │
└────────────────────────────┘

    Positives: Historical bloom observations
    
    Negatives (synthetic):
    • Same location ±60-90 days (temporal offset)
    • Random locations, off-season
    
    Model learns contrast between bloom/no-bloom
```

### Feature Richness Comparison

```
v1: 9 Basic Features
├── Spatial: lat, lon
├── Temporal: day_of_year, month, year
└── Environmental: temp, precip, ndvi, elevation

❌ Single-point values (no aggregation)
❌ No temporal dynamics
❌ No trends or changes


v2: 21 Engineered Features
├── Spatial (2)
│   └── lat, lon
│
├── Temporal (6)
│   ├── day_of_year, month, week_of_year
│   ├── day_sin, day_cos ← Cyclical encoding
│   └── days_from_species_mean ← Species-specific
│
├── Environmental - Temperature (4)
│   ├── temp_mean_30d ← 30-day average
│   ├── temp_max_30d, temp_min_30d
│   └── temp_range ← Variability
│
├── Environmental - Precipitation (2)
│   ├── precip_total_30d ← Accumulation
│   └── precip_mean_30d
│
├── Environmental - Vegetation (3)
│   ├── ndvi_mean_30d
│   ├── ndvi_max_30d
│   └── ndvi_trend ← DYNAMICS! Rate of change
│
├── Environmental - Other (1)
│   └── elevation
│
└── Derived (3)
    ├── growing_degree_days ← Accumulated heat
    ├── moisture_index ← Precip/temp ratio
    └── vegetation_health ← NDVI × trend

✅ 30-day aggregations (not single points)
✅ Temporal dynamics (cyclical, trends)
✅ Accumulation effects (GDD, precip total)
```

## The Bottom Line

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  v1: Predicts species, pretends it's bloom probability       │
│      ❌ Fundamentally broken                                 │
│                                                              │
│  v2: Predicts bloom probability from learned patterns        │
│      ✅ Scientifically sound                                 │
│                                                              │
│  v2 LEARNS BLOOM DYNAMICS:                                   │
│  • When blooms occur (temporal patterns)                     │
│  • Where blooms occur (spatial patterns)                     │
│  • What conditions favor blooms (environmental relationships)│
│  • How trends indicate bloom readiness (NDVI slope, etc)     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```
