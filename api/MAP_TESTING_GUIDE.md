# Testing Bloom Model v2 with Map Visualization

## Quick Start: See Your Model Predictions on the Map! 🗺️

### Step 1: Start the API Server

```bash
cd /Users/enayala/Developer/NasaHack/bloombly/api
python app/main.py
```

You should see:
```
 * Running on http://127.0.0.1:5001
```

### Step 2: Start the Frontend

```bash
cd /Users/enayala/Developer/NasaHack/bloombly/frontend
python -m http.server 3000
```

Or use any web server:
```bash
# Using Node.js
npx http-server -p 3000

# Using PHP
php -S localhost:3000
```

### Step 3: Open in Browser

Navigate to: **http://localhost:3000**

---

## Using the Map to Test Model v2

### **Default View (v1 model)**
When you first open the map, it loads **historical bloom observations** using the old v1 model.

### **Switch to Model v2** 🎯

The API supports the v2 model through query parameters. Here's how to test it:

#### **Method 1: Modify the frontend code temporarily**

Open `frontend/index.html` and find this line (around line 427):
```javascript
fetch(`${API_BASE_URL}/blooms?aoi_type=global&date=2024-07-01`)
```

Change it to:
```javascript
fetch(`${API_BASE_URL}/blooms?aoi_type=global&date=2024-07-01&method=v2&confidence_threshold=0.5`)
```

**Parameters explained:**
- `method=v2` - Use the new v2 model (learns bloom dynamics!)
- `confidence_threshold=0.5` - Only show blooms with ≥50% probability

Refresh the browser and you'll see **predicted blooms** based on the v2 model! 🌸

#### **Method 2: Test via Browser Console**

Open browser DevTools (F12), go to Console, and run:

```javascript
// Test v2 model for different dates
const testV2Model = async (date, threshold = 0.5) => {
  const url = `http://localhost:5001/api/blooms?aoi_type=global&date=${date}&method=v2&confidence_threshold=${threshold}`;
  const response = await fetch(url);
  const data = await response.json();
  
  console.log(`Date: ${date}, Predictions: ${data.features.length}`);
  console.log('Sample predictions:', data.features.slice(0, 3));
  
  // Update map
  state.geojsonFeatures = data.features;
  state.pointsData = data.features.map(createPointFromFeature);
  applyFilters();
  
  return data;
};

// Test different dates
await testV2Model('2024-03-15');  // Spring
await testV2Model('2024-06-15');  // Summer  
await testV2Model('2024-09-15');  // Fall (should have highest predictions for Symphyotrichum!)
```

#### **Method 3: Direct API Testing**

Test the API directly in your browser or using curl:

```bash
# Test v2 model predictions for September (peak bloom)
curl "http://localhost:5001/api/blooms?date=2024-09-15&method=v2&confidence_threshold=0.5"

# Test for March (should be low predictions)
curl "http://localhost:5001/api/blooms?date=2024-03-15&method=v2&confidence_threshold=0.5"

# Test with different confidence thresholds
curl "http://localhost:5001/api/blooms?date=2024-09-15&method=v2&confidence_threshold=0.7"  # Only very confident predictions

# Test specific region
curl "http://localhost:5001/api/blooms?date=2024-09-15&method=v2&aoi_type=country&aoi_country=United%20States"
```

---

## What to Look For (Model Validation)

### **✅ Good Signs (Model is Learning!)**

1. **Seasonal Patterns:**
   - September predictions >> March predictions (for fall bloomers like *Symphyotrichum*)
   - More blooms during each species' natural season
   
2. **Spatial Clustering:**
   - Predictions cluster in appropriate geographic regions
   - Not randomly scattered across the globe
   
3. **Probability Distribution:**
   - High confidence (70-90%) in peak season
   - Low confidence (10-30%) in off-season
   - Not all predictions at 50%

4. **Species-Appropriate Timing:**
   - *Symphyotrichum novae-angliae* peaks in September
   - *Symphyotrichum ericoides* peaks in Fall
   - Spring bloomers peak in March-May

### **❌ Red Flags (Model Issues)**

1. **No seasonal variation** (same predictions for all months)
2. **Random scatter** (no geographic clustering)
3. **All predictions same probability** (not learning nuance)
4. **No predictions at all** (threshold too high or model broken)

---

## Interactive Testing Workflow

### **Scenario 1: Validate Seasonal Learning**

```javascript
// Browser console
const testSeasons = async () => {
  const seasons = [
    { month: '03', name: 'Spring' },
    { month: '06', name: 'Summer' },
    { month: '09', name: 'Fall' },
    { month: '12', name: 'Winter' }
  ];
  
  for (const season of seasons) {
    const data = await testV2Model(`2024-${season.month}-15`, 0.5);
    console.log(`${season.name}: ${data.features.length} predictions`);
  }
};

testSeasons();
```

**Expected Result:** Fall should have the most predictions for *Symphyotrichum*!

### **Scenario 2: Test Different Confidence Thresholds**

```javascript
const testThresholds = async () => {
  const thresholds = [0.3, 0.5, 0.7, 0.9];
  
  for (const threshold of thresholds) {
    const data = await testV2Model('2024-09-15', threshold);
    console.log(`Threshold ${threshold}: ${data.features.length} predictions`);
    
    // Show probability distribution
    const probs = data.features.map(f => f.properties.bloom_probability);
    console.log(`  Avg probability: ${(probs.reduce((a,b) => a+b, 0) / probs.length).toFixed(2)}`);
  }
};

testThresholds();
```

**Expected Result:** Higher thresholds = fewer but more confident predictions

### **Scenario 3: Compare v1 vs v2**

```javascript
const compareModels = async (date) => {
  // v1 (old model)
  const v1 = await fetch(`http://localhost:5001/api/blooms?date=${date}&method=bloom_dynamics`).then(r => r.json());
  
  // v2 (new model)
  const v2 = await fetch(`http://localhost:5001/api/blooms?date=${date}&method=v2`).then(r => r.json());
  
  console.log('v1 predictions:', v1.features.length);
  console.log('v2 predictions:', v2.features.length);
  
  // Check if v2 has probabilities
  console.log('v2 sample:', v2.features[0]?.properties);
};

compareModels('2024-09-15');
```

---

## Advanced: Add Model Comparison UI

Want to see v1 vs v2 side-by-side? Add this to `index.html`:

### Add to the legend controls:

```html
<div class="filter-section">
  <h4>Model Version</h4>
  <select id="modelSelect">
    <option value="bloom_dynamics">v1 (Old Model)</option>
    <option value="v2">v2 (ML Model)</option>
  </select>
  <label>
    Min Confidence: 
    <input type="range" id="confidenceSlider" min="0" max="100" value="50">
    <span id="confidenceValue">50%</span>
  </label>
</div>
```

### Add event listener:

```javascript
// After the existing event listeners
document.getElementById('modelSelect').addEventListener('change', function() {
  const method = this.value;
  const confidence = document.getElementById('confidenceSlider').value / 100;
  const date = document.getElementById('dateInput').value;
  
  const url = `${API_BASE_URL}/blooms?date=${date}&method=${method}&confidence_threshold=${confidence}`;
  fetch(url)
    .then(r => r.json())
    .then(data => {
      state.geojsonFeatures = data.features;
      state.pointsData = data.features.map(createPointFromFeature);
      applyFilters();
    });
});

document.getElementById('confidenceSlider').addEventListener('input', function() {
  document.getElementById('confidenceValue').textContent = this.value + '%';
});
```

Now you can toggle between v1 and v2 models directly in the UI! 🎉

---

## Debugging Tips

### **API not responding?**
```bash
# Check if API is running
curl http://localhost:5001/api/health

# Check logs in API terminal
# Look for errors about missing data or model issues
```

### **No predictions showing?**
1. Check browser console for errors (F12 → Console)
2. Lower confidence threshold: `confidence_threshold=0.3`
3. Check API response: Are there any features?
4. Verify date is valid: Use `YYYY-MM-DD` format

### **Map not loading?**
1. Check frontend server is running on port 3000
2. Check `API_BASE_URL` in index.html matches your API (should be `http://localhost:5001/api`)
3. Check CORS errors in console (API should allow localhost:3000)

### **Model predictions seem random?**
1. Run `python evaluate_model.py` to check accuracy
2. Retrain model: `python train_model.py`
3. Check feature importance: Are temporal features important?

---

## Expected Results for Good Model

When testing on **September 15, 2024**:

### **v2 Model (Good)**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "properties": {
        "Site": "Predicted Bloom 1",
        "Family": "Asteraceae",
        "Genus": "Symphyotrichum",
        "Season": "Fall",
        "bloom_probability": 0.87,  // ← HIGH confidence
        "environmental_factors": {
          "temperature": 18.5,
          "ndvi": 0.65
        }
      }
    }
  ]
}
```

### **v1 Model (Random)**
Predictions don't change much between March and September - no seasonal learning!

---

## Summary

### ✅ **To Test v2 Model on Map:**

1. **Start API:** `python app/main.py`
2. **Start Frontend:** `python -m http.server 3000`
3. **Modify URL:** Add `&method=v2&confidence_threshold=0.5`
4. **Compare Seasons:** Test March vs September
5. **Validate:** Fall should have more predictions than Spring

### 🎯 **Key Validation:**
- September blooms >> March blooms ✓
- Spatial clustering in appropriate regions ✓
- Probability varies (not all 50%) ✓
- High confidence during peak season ✓

Your model is **not trash** - it learns seasonal bloom dynamics! The map should prove it visually. 🌸🗺️