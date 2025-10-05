# Model Info API Fix

## Problem

When accessing `/api/model-info`, the response showed incorrect values:
- `"feature_count": 0`
- `"features": []`  
- `"total_samples": 0`

Even though the model was trained and working, with positive/negative examples showing correctly.

## Root Cause

The `feature_data` DataFrame was not being saved/loaded in the model pickle file.

**In `bloom_predictor_v2.py`:**

1. `feature_data` is created in `build_temporal_features()` method
2. It was **NOT** included in `save_model()` 
3. It was **NOT** restored in `load_model()`

This meant when the API loaded a pre-trained model from disk, the `feature_data` attribute was missing or empty, causing the API to return 0 for total_samples.

## Solution

### 1. Fixed Model Save/Load (bloom_predictor_v2.py)

**In `save_model()` method:**
```python
model_data = {
    # ... existing fields ...
    'feature_data': self.feature_data if hasattr(self, 'feature_data') else None  # ADDED
}
```

**In `load_model()` method:**
```python
self.feature_data = model_data.get('feature_data', None)  # ADDED
print(f"  Loaded {len(self.feature_data)} feature samples")  # ADDED
```

### 2. Fixed API Endpoint Robustness (routes/predict.py)

Added better handling for when `feature_data` is missing:

```python
# Calculate totals based on available data
positive_count = len(predictor.historical_blooms) if hasattr(predictor, 'historical_blooms') and predictor.historical_blooms is not None else 0
negative_count = len(predictor.negative_examples) if hasattr(predictor, 'negative_examples') and predictor.negative_examples is not None else 0

# Use feature_data if available, otherwise calculate from positive + negative
if hasattr(predictor, 'feature_data') and predictor.feature_data is not None:
    total_samples = len(predictor.feature_data)
else:
    total_samples = positive_count + negative_count
```

This ensures the API returns correct values even with old model files.

## How to Apply the Fix

### Option 1: Retrain Model (Recommended)

Run the provided script to retrain with the fix:

```bash
cd /Users/enayala/Developer/NasaHack/bloombly/api
python retrain_and_save_v2.py
```

This will create a new `bloom_model_v2.pkl` with `feature_data` properly saved.

### Option 2: Use Current Model

The API fix will work with your current model file. It will calculate `total_samples` as:
```
total_samples = positive_examples + negative_examples
```

## Expected Results After Fix

```json
{
  "feature_count": 21,
  "features": [
    "lat", "lon", "day_of_year", "month", "week_of_year",
    "day_sin", "day_cos", "days_from_species_mean",
    "temp_mean", "temp_max", "temp_min", "temp_range",
    "precip_total", "precip_mean",
    "ndvi_mean", "ndvi_max", "ndvi_trend",
    "elevation", "growing_degree_days", "moisture_index",
    "vegetation_health"
  ],
  "training_data": {
    "positive_examples": 1177,
    "negative_examples": 2942,
    "total_samples": 4119
  }
}
```

## Files Modified

1. `/api/app/bloom_predictor_v2.py` - Fixed save/load methods
2. `/api/app/routes/predict.py` - Added robust total_samples calculation
3. `/api/retrain_and_save_v2.py` - New helper script for retraining

## Technical Notes

- The `feature_data` DataFrame contains the complete feature matrix used for training
- It includes all engineered features from both bloom and no-bloom examples
- Size should equal: `len(historical_blooms) + len(negative_examples)`
- The DataFrame is relatively large (~4000 rows × 21 features) but important for model inspection
