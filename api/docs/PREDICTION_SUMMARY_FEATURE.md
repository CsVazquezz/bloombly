# Prediction Summary Feature

## Overview
The prediction summary feature provides aggregated statistics and insights for bloom predictions, making it easier to understand the results at a glance.

## Backend Implementation

### Summary Calculation Function
Located in: `/api/app/routes/predict.py`

The `calculate_prediction_summary()` function computes the following statistics:

#### 1. **Overview Statistics**
- Total number of predictions
- Unique species count
- Number of plant families represented
- Total bloom area (square meters)

#### 2. **Confidence Levels** (for ML predictions)
- Average bloom probability
- Probability range (min/max)
- Distribution by confidence:
  - **High confidence**: ≥ 70%
  - **Medium confidence**: 40-69%
  - **Low confidence**: < 40%

#### 3. **Seasonal Distribution**
- Bloom count per season (Spring, Summer, Fall, Winter)

#### 4. **Family Distribution**
- Count of predictions per plant family

#### 5. **Environmental Conditions**
- Temperature statistics (average, min, max)
- Precipitation statistics (average, total)
- Vegetation index (NDVI) statistics (average, min, max)

### API Response Structure

```json
{
  "type": "FeatureCollection",
  "features": [...],
  "metadata": {
    "prediction_date": "2024-09-15",
    "aoi_type": "global",
    "model_version": "v2",
    "summary": {
      "total_predictions": 200,
      "species_count": 45,
      "family_count": 12,
      "families": {
        "Rosaceae": 35,
        "Fabaceae": 28,
        ...
      },
      "seasons": {
        "Spring": 120,
        "Summer": 50,
        "Fall": 20,
        "Winter": 10
      },
      "total_area": 1250000.5,
      "average_probability": 0.653,
      "probability_range": {
        "min": 0.301,
        "max": 0.987
      },
      "high_confidence_count": 75,
      "medium_confidence_count": 98,
      "low_confidence_count": 27,
      "environmental_summary": {
        "temperature": {
          "avg": 18.5,
          "min": 12.3,
          "max": 24.7
        },
        "precipitation": {
          "avg": 45.2,
          "total": 9040.0
        },
        "vegetation_index": {
          "avg": 0.652,
          "min": 0.421,
          "max": 0.843
        }
      }
    }
  }
}
```

## Frontend Implementation

### Summary Panel
Located in: `/frontend/index.html`

#### Features:
1. **Collapsible Panel** - Toggle button to show/hide summary
2. **Organized Sections**:
   - 📊 Overview - Total counts and areas
   - 🎯 Confidence Levels - Probability distribution
   - 🌸 Seasonal Distribution - Blooms by season
   - 🌿 Top Plant Families - Most common families (top 5)
   - 🌡️ Environmental Conditions - Climate and vegetation data
   - ℹ️ Prediction Info - Model and date information

3. **Visual Elements**:
   - Color-coded confidence bars (green/orange/gray)
   - Formatted statistics with labels and values
   - Responsive design that adapts to content

### User Interface

#### Show/Hide Summary
- Click the **"Show Summary"** button in the top-left corner
- Button changes to **"Hide Summary"** when panel is visible
- Panel persists across different data loads

#### Auto-Update
The summary automatically updates when:
- Initial data loads
- User applies new AOI filters
- User changes prediction parameters
- User switches between satellite and prediction data

## Usage Examples

### Example 1: Global Prediction Summary
```
📊 Overview
Total Predictions: 200
Unique Species: 45
Plant Families: 12
Total Area: 1,250,000 m²

🎯 Confidence Levels
Average Probability: 65.3%
Range: 30.1% - 98.7%
[High: 75] [Med: 98] [Low: 27]

🌸 Seasonal Distribution
Spring: 120
Summer: 50
Fall: 20
Winter: 10
```

### Example 2: State-Specific Summary
```
📊 Overview
Total Predictions: 50
Unique Species: 12
Plant Families: 5

🎯 Confidence Levels
Average Probability: 72.4%

🌿 Top Plant Families
Rosaceae: 15
Fabaceae: 12
Asteraceae: 8
```

## Technical Details

### Performance Considerations
- Summary calculation is O(n) where n = number of predictions
- Cached in metadata to avoid recalculation
- Minimal impact on API response time

### Data Validation
- Handles missing or incomplete data gracefully
- Defaults to 0 or "N/A" for unavailable metrics
- Validates probability ranges (0-1)

### Browser Compatibility
- Uses standard JavaScript (ES6+)
- Compatible with modern browsers
- Responsive design for mobile/desktop

## Future Enhancements

Potential improvements:
1. Export summary as CSV/JSON
2. Historical comparison (compare multiple dates)
3. Interactive charts/graphs
4. Custom summary thresholds
5. Summary templates for different use cases
6. Email/notification of summaries

## Testing

To test the summary feature:

1. **Start the API**:
   ```bash
   cd api
   python app/main.py
   ```

2. **Open Frontend**:
   ```bash
   cd frontend
   # Open index.html in browser
   ```

3. **Load Predictions**:
   - Select "ML Predictions (v2)" from Data Source
   - Click "Apply AOI"
   - Click "Show Summary" button

4. **Verify Summary**:
   - Check that all sections display
   - Verify numbers match prediction count
   - Test with different confidence thresholds
   - Test with different AOI settings

## Troubleshooting

### Summary Not Showing
- Check browser console for errors
- Verify API is running and returning metadata
- Check that summary object exists in metadata

### Incorrect Statistics
- Verify predictions have required properties
- Check that environmental_factors exist in predictions
- Review backend calculation logic

### Styling Issues
- Clear browser cache
- Check CSS is properly loaded
- Verify no conflicting styles

## API Endpoints

### Get Predictions with Summary
```
GET /predict/blooms?date=2024-09-15&method=v2&confidence=0.3&num_predictions=200
```

Response includes `metadata.summary` object with all statistics.

### Get Model Info
```
GET /predict/model-info?version=v2
```

Returns model information including feature count and training data stats.
