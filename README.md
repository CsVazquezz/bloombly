# 🌸 Bloombly - Wildflower Bloom Prediction Platform

[![NASA Space Apps Challenge](https://img.shields.io/badge/NASA-Space%20Apps%20Challenge-blue.svg)](https://www.spaceappschallenge.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)

> **Predicting wildflower blooms using machine learning, satellite imagery, and ecological modeling to understand the impact of climate change on plant phenology.**

<p align="center">
  <img src="https://img.shields.io/badge/🌍-Global%20Coverage-brightgreen" alt="Global Coverage">
  <img src="https://img.shields.io/badge/🛰️-Satellite%20Data-orange" alt="Satellite Data">
  <img src="https://img.shields.io/badge/🤖-ML%20Powered-red" alt="ML Powered">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
- [API Documentation](#-api-documentation)
- [Machine Learning Models](#-machine-learning-models)
- [Data Sources](#-data-sources)
- [Project Structure](#-project-structure)
- [Advanced Features](#-advanced-features)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🌍 Overview

**Bloombly** is an advanced wildflower bloom prediction platform developed for the **NASA Space Apps Challenge**. The project combines satellite remote sensing, machine learning, and ecological modeling to predict when and where wildflowers will bloom across the globe.

### The Challenge

Climate change is altering the timing of seasonal events (phenology), affecting plant blooming patterns worldwide. Understanding these changes is crucial for:

- 🦋 **Biodiversity Conservation** - Tracking pollinator-plant synchronization
- 🌾 **Agriculture** - Optimizing crop management and pollination
- 🔬 **Climate Science** - Monitoring ecosystem responses to climate change
- 🌺 **Tourism & Recreation** - Planning wildflower viewing opportunities

### Our Solution

Bloombly uses a multi-faceted approach:

1. **Machine Learning Model (v2)** - Binary classification predicting bloom probability with 31+ ecological features
2. **Satellite Data Integration** - Real-time environmental data from NASA Earth Engine
3. **Interactive 3D Globe Visualization** - Explore bloom predictions worldwide
4. **Species-Specific Models** - Specialized predictors including cherry blossom (sakura) forecasting
5. **Climate Analysis** - Track bloom timing shifts over 73 years of historical data

---

## ✨ Features

### 🎯 Core Capabilities

- **🔮 Bloom Prediction**
  - Predict bloom dates up to 90 days in advance
  - Confidence scores and probability estimates
  - Support for multiple species (Symphyotrichum, Prunus, and more)
  - Global and regional predictions (USA, Mexico, Japan)

- **🛰️ Real-Time Environmental Analysis**
  - NDVI vegetation indices from MODIS satellites
  - Land surface temperature data
  - Soil moisture from NASA SMAP
  - Precipitation and climate patterns

- **🌸 Advanced Ecological Features**
  - **Spring Start Detection** - Identifies vegetation green-up using NDVI time series
  - **Growing Degree Days (GDD)** - Heat accumulation modeling (Baskerville-Emin method)
  - **Soil Water Availability** - Plant water stress analysis using permanent wilting point
  - **Temporal Dynamics** - Seasonal trends, cyclical patterns, and multi-year analysis

- **🗺️ Interactive Visualization**
  - 3D globe interface with real-time data
  - Timeline controls for temporal exploration
  - Filter by species, location, and bloom probability
  - GeoJSON export for integration with other tools

- **🌏 Special Models**
  - **Sakura Predictor** - Cherry blossom forecasting for Japan (73 years of historical data)
  - **Regional Models** - Optimized for US states and Mexican regions
  - **Multi-species Support** - Extensible framework for different plant families

### 🎨 User Interface

- Interactive 3D globe powered by Globe.gl
- Timeline scrubbing for date selection
- Advanced filtering and search
- Real-time prediction updates
- Downloadable GeoJSON data

---

## 🔧 Technology Stack

### Backend

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core language | 3.11+ |
| **Flask** | REST API framework | 2.3.3 |
| **scikit-learn** | Machine learning | 1.7.1 |
| **Google Earth Engine** | Satellite data | 1.6.10 |
| **pandas/numpy** | Data processing | Latest |
| **joblib** | Model serialization | Latest |

### Frontend

| Technology | Purpose |
|------------|---------|
| **Globe.gl** | 3D globe visualization |
| **Three.js** | 3D rendering |
| **Vanilla JavaScript** | Interactive UI |
| **Express.js** | Static file serving |

### Data Sources

- **NASA MODIS** - NDVI and Land Surface Temperature (250m-1km resolution)
- **NASA SMAP** - Soil Moisture (10km resolution)
- **GLOBE Observer** - Historical bloom observations
- **Japanese Meteorological Agency** - 73 years of sakura bloom dates
- **Kaggle Datasets** - Cherry blossom forecasts and phenology data

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Globe UI)                       │
│  - 3D Globe Visualization  - Timeline Controls               │
│  - Real-time Updates       - GeoJSON Export                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ↓ REST API
┌─────────────────────────────────────────────────────────────┐
│                     Flask API Server                         │
│  Routes:                                                     │
│    /api/predict/blooms  - ML predictions                     │
│    /api/sakura/predict  - Cherry blossom forecasts           │
│    /api/data/blooms     - Historical data                    │
└─────────────────┬───────────────────────┬───────────────────┘
                  │                       │
       ┌──────────▼──────────┐  ┌────────▼──────────┐
       │   ML Model v2        │  │  Earth Engine API │
       │  (Bloom Dynamics)    │  │  (Satellite Data) │
       │  - Binary classifier │  │  - MODIS NDVI     │
       │  - 31 features       │  │  - MODIS LST      │
       │  - GradientBoosting  │  │  - NASA SMAP      │
       └──────────────────────┘  └───────────────────┘
```

### Data Flow

1. **User Request** → Frontend sends date/location to API
2. **Environmental Data** → API queries Earth Engine for satellite data
3. **Feature Engineering** → Calculate 31 ecological features (spring onset, GDD, soil water, etc.)
4. **ML Prediction** → Model predicts bloom probability
5. **Spatial Sampling** → Generate predictions across geographic area
6. **Response** → GeoJSON with bloom predictions and confidence scores
7. **Visualization** → Display on interactive globe

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 16+ (for frontend)
- Google Cloud Platform account (optional, for Earth Engine)
- Anaconda (recommended for dependency management)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/KIKW12/bloombly.git
cd bloombly
```

#### 2. Set Up Backend (API)

```bash
# Create and activate conda environment
conda create -n bloombly python=3.11 -y
conda activate bloombly

# Install dependencies
cd api
conda install -c conda-forge earthengine-api geemap pandas numpy flask flask-cors google-auth python-dotenv -y

# Or use pip
pip install -r requirements.txt
```

#### 3. Configure Environment Variables

Create a `.env` file in the `api` directory:

```bash
# Optional: For real satellite data (otherwise uses synthetic fallback)
EE_PROJECT=your-google-cloud-project-id
GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
```

> **Note:** The system works without Earth Engine credentials using synthetic climate data, but real satellite data improves accuracy.

#### 4. Run the API Server

```bash
cd api
python app/main.py
```

The API will start on `http://localhost:5001`

#### 5. Set Up Frontend

```bash
cd frontend
npm install
npm start
```

The frontend will be available at `http://localhost:3000`

### Quick Test

```bash
# Test bloom prediction API
curl "http://localhost:5001/api/predict/blooms?date=2025-06-15&method=v2&confidence=0.3&aoi_type=state&aoi_state=Texas"

# Test sakura prediction
curl -X POST http://localhost:5001/api/sakura/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 35.68, "longitude": 139.65, "year": 2025}'
```

---

## 📡 API Documentation

### Bloom Predictions

#### `GET /api/predict/blooms`

Predict wildflower blooms for a specific date or date range.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `date` | string | No* | Target date (YYYY-MM-DD) |
| `start_date` | string | No* | Start of date range |
| `end_date` | string | No* | End of date range |
| `method` | string | No | `v2` (default), `enhanced`, `statistical` |
| `confidence` | float | No | Minimum probability (0.0-1.0, default 0.3) |
| `aoi_type` | string | No | `global`, `country`, `state` |
| `aoi_state` | string | No | State name (e.g., "Texas", "Queretaro") |
| `aoi_country` | string | No | Country name |
| `num_predictions` | int | No | Max predictions to return (default 100) |

*Either `date` or `start_date`/`end_date` required

**Example Response:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "Site": "Symphyotrichum novae-angliae",
        "Family": "Asteraceae",
        "Genus": "Symphyotrichum",
        "Season": "Summer",
        "bloom_probability": 0.847,
        "predicted_date": "2025-06-15",
        "is_prediction": true,
        "model_version": "v2_bloom_dynamics",
        "environmental_factors": {
          "temperature": 22.3,
          "precipitation": 85.2,
          "ndvi": 0.682,
          "spring_start_day": 75,
          "gdd_accumulated": 520,
          "soil_water_days": 8.5
        }
      },
      "geometry": {
        "type": "Point",
        "coordinates": [-97.5, 30.3]
      }
    }
  ],
  "metadata": {
    "prediction_date": "2025-06-15",
    "model_version": "v2",
    "confidence_threshold": 0.3,
    "num_predictions": 100
  }
}
```

### Sakura (Cherry Blossom) Predictions

#### `POST /api/sakura/predict`

Predict cherry blossom bloom dates for Japan.

**Request Body:**

```json
{
  "latitude": 35.68,
  "longitude": 139.65,
  "year": 2025,
  "species": "Prunus × yedoensis",
  "include_window": true
}
```

**Response:**

```json
{
  "bloom_day_of_year": 95,
  "bloom_date": "2025-04-05",
  "confidence": 0.92,
  "model_used": "japan_specialized",
  "bloom_window": {
    "early": "2025-03-28",
    "peak": "2025-04-05",
    "late": "2025-04-12"
  }
}
```

### Model Information

#### `GET /api/predict/model-info?version=v2`

Get detailed information about the ML model.

### Health Check

#### `GET /api/health`

Check API status and Earth Engine availability.

---

## 🤖 Machine Learning Models

### Model v2: Bloom Dynamics Predictor

**The current production model** - A complete redesign that learns actual bloom dynamics.

#### Model Architecture

```
Input: 31 Features
├── Spatial (2): latitude, longitude
├── Temporal (6): day_of_year, month, week, cyclical encoding, days_from_species_mean
├── Temperature (4): mean, max, min, range (30-day aggregates)
├── Precipitation (2): total, mean (30-day)
├── Vegetation (3): NDVI mean, max, trend (30-day)
├── Derived (3): growing_degree_days, moisture_index, vegetation_health
├── Spring Features (4): spring_start_day, days_since_spring, is_spring_active, winter_baseline
├── GDD Features (2): current_gdd, accumulated_gdd_30d
└── Soil Water (4): soil_water_days, wilting_point, water_stress, available_water_ratio

Algorithm: Gradient Boosting Classifier
├── 200 estimators
├── Max depth: 5
├── Learning rate: 0.05
└── StandardScaler normalization

Validation: 5-fold Time-Series Cross-Validation
Output: Bloom probability (0.0-1.0)
```

#### Key Features Explained

**1. Spring Start Detection**
- Uses MODIS NDVI time series to detect when vegetation "wakes up"
- 5-day moving average smoothing
- Identifies sustained growth above winter baseline
- Critical for predicting bloom timing

**2. Growing Degree Days (GDD)**
```python
GDD = [(Tmax + Tmin) / 2] - Tbase
# Accumulated over 30 days
```
- Measures heat energy for plant development
- Based on Baskerville-Emin (1969) method
- Predicts earlier blooming in warm years

**3. Soil Water Availability**
```python
PWP = (Field_Capacity × 0.74) - 5  # Permanent Wilting Point
Soil_Water_Days = max(0, Soil_Moisture - PWP)
```
- Estimates plant-available water
- Water stress indicator
- Reduces false positives in drought conditions

#### Training Data

- **Positive Examples**: 1,179 historical bloom observations
- **Negative Examples**: 3,000 synthetic no-bloom events
  - Temporal offset: Same location, ±60-90 days
  - Spatial random: Random locations during off-season
- **Ratio**: ~2.5:1 negative:positive (balanced training)
- **Species**: Symphyotrichum novae-angliae, S. ericoides, and others
- **Time Range**: 2014-2017 (extendable with more data)
- **Geographic Coverage**: Northeastern US (expandable)

#### Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **ROC-AUC** | 0.72-0.85 | Probability calibration quality |
| **Precision** | 0.68-0.78 | Bloom prediction accuracy |
| **Recall** | 0.64-0.74 | Bloom detection rate |
| **F1-Score** | 0.66-0.76 | Overall performance balance |

### Sakura (Cherry Blossom) Model

Specialized model for Japanese cherry blossoms using **73 years** of historical data (1953-2025).

- **Data Sources**: Japanese Meteorological Agency, Phenobase, historical records
- **Species**: Prunus × yedoensis (Somei Yoshino), P. jamasakura, P. sargentii
- **Features**: Temperature accumulation, chilling hours, photoperiod, elevation
- **Accuracy**: ±3 days average error
- **Coverage**: 58 cities across Japan

---

## 📊 Data Sources

### NASA Earth Engine Collections

| Dataset | Collection ID | Resolution | Temporal | Purpose |
|---------|--------------|------------|----------|---------|
| **MODIS NDVI** | `MODIS/061/MOD13Q1` | 250m | 16-day | Spring detection, vegetation health |
| **MODIS LST** | `MODIS/061/MOD11A1` | 1km | Daily | Temperature (GDD calculation) |
| **NASA SMAP** | `NASA/SMAP/SPL4SMGP/007` | 10km | 3-hourly | Soil moisture |

### Historical Bloom Data

- **GLOBE Observer**: Citizen science wildflower observations
- **Japanese Meteorological Agency**: Sakura bloom dates (1953-2025)
- **Phenobase**: Swiss cherry phenology (1978-2015)
- **Kaggle Datasets**: Additional cherry blossom forecasts

### Fallback Data

When Earth Engine is unavailable, the system uses:
- Climate normals based on latitude/longitude
- Seasonal sine wave models for temperature/NDVI
- Precipitation estimates
- Less accurate but functional for demonstrations

---

## 📁 Project Structure

```
bloombly/
├── api/                          # Backend API
│   ├── app/
│   │   ├── main.py              # Flask application entry point
│   │   ├── config.py            # Configuration and AOI definitions
│   │   ├── bloom_predictor_v2.py # ML model v2 (production)
│   │   ├── bloom_predictor.py   # Legacy model v1
│   │   ├── sakura_predictor.py  # Cherry blossom specialized model
│   │   ├── bloom_features.py    # Advanced feature calculations
│   │   ├── earth_engine_utils.py # Satellite data retrieval
│   │   ├── routes/
│   │   │   ├── predict.py       # Bloom prediction endpoints
│   │   │   ├── sakura.py        # Sakura prediction endpoints
│   │   │   └── data.py          # Historical data endpoints
│   │   └── models/
│   │       └── bloom_model_v2.pkl # Trained ML model
│   ├── docs/                     # API documentation
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                     # Frontend application
│   ├── public/
│   │   ├── index.html
│   │   ├── js/
│   │   │   ├── app.js           # Main application
│   │   │   ├── globe.js         # 3D globe visualization
│   │   │   ├── state.js         # State management
│   │   │   └── components/      # UI components
│   │   └── styles/              # CSS stylesheets
│   ├── package.json
│   └── server.js
│
├── ml/                           # Machine learning utilities
│   ├── src/
│   │   ├── clean_data.py        # Data preprocessing
│   │   └── features.py          # Feature engineering
│   └── requirements.txt
│
├── data/                         # Data files
│   ├── raw/                     # Original datasets
│   ├── processed/               # Cleaned datasets
│   └── geojson/                 # GeoJSON outputs
│
├── backend/                      # Data processing scripts
│   ├── clean_blooms_ml.py
│   ├── clean_phenobase.py
│   └── csv_to_geojson.py
│
└── README.md                     # This file
```

---

## 🧬 Advanced Features

### Spring Phenology Detection

Detects the start of spring using NDVI time series:

1. **5-day moving average** smoothing to eliminate daily noise
2. Calculate **winter NDVI baseline** (Dec 1 - Mar 1)
3. Detect when NDVI **exceeds baseline** with sustained growth
4. Identify **longest consecutive growth period** as spring onset

```python
# Example: Spring starts on day 75 (March 16)
spring_start_day = 75
days_since_spring_start = current_day - spring_start_day
is_spring_active = days_since_spring_start > 0
```

### Growing Degree Days (GDD)

Heat accumulation model based on Baskerville-Emin (1969):

```python
daily_gdd = max(0, (T_max + T_min) / 2 - T_base)
accumulated_gdd = sum(daily_gdd for past 30 days)

# T_base = 0°C for general vegetation
# Higher GDD = more growth potential
```

### Soil Water Stress Analysis

Permanent wilting point method:

```python
# Field capacity from SMAP soil moisture
field_capacity = soil_moisture_max

# Wilting point (plants can't extract water below this)
wilting_point = (field_capacity × 0.74) - 5

# Available water
if soil_moisture < wilting_point:
    water_stress = True
    available_water = 0
else:
    water_stress = False
    available_water = soil_moisture - wilting_point
```

### Temporal Feature Engineering

- **Cyclical encoding**: `sin(2π × day/365)`, `cos(2π × day/365)` for seasonal patterns
- **Species-specific timing**: Days from species mean bloom day
- **Rolling averages**: 30-day environmental aggregations
- **Trend detection**: NDVI slopes, temperature changes

---

## 🔬 Use Cases

### 1. Climate Change Research
- Track shifts in bloom timing over decades
- Identify regions most affected by warming
- Study phenological mismatches (plants vs pollinators)

### 2. Biodiversity Conservation
- Monitor critical bloom periods for endangered pollinators
- Predict food availability for migratory species
- Assess ecosystem health through bloom patterns

### 3. Agriculture & Horticulture
- Optimize planting schedules
- Predict pollination windows
- Manage orchard bloom timing

### 4. Tourism & Recreation
- Plan wildflower viewing trips
- Cherry blossom festival timing
- Nature photography expeditions

### 5. Citizen Science
- Validate model predictions with observations
- Crowdsource bloom data collection
- Educational outreach

---

## 🛠️ Development

### Training a New Model

```bash
cd api

# Basic training (uses existing processed data)
python train_model.py

# With custom parameters
python train_model.py \
  --n_estimators 300 \
  --max_depth 7 \
  --learning_rate 0.03 \
  --use_earth_engine

# Quick retrain (maintains existing negative examples)
python quick_retrain.py
```

### Model Evaluation

```bash
# Run full evaluation suite
python evaluate_model.py

# Compare multiple models
python model_comparison.py

# Validate against Kaggle test set
python validate_predictions.py
```

### Testing Features

```bash
# Test new ecological features
python test_new_features.py

# Debug predictions
python test_prediction_debug.py

# Test Sakura models
python test_sakura_models.py
```

### Adding New Species

1. **Prepare training data** (CSV with lat, lon, date, species)
2. **Add to data processing** (`ml/src/clean_data.py`)
3. **Retrain model** with expanded dataset
4. **Update species bloom windows** in predictor

---

## 🌐 Deployment

### Docker Deployment

```bash
# Build container
docker build -t bloombly-api ./api

# Run container
docker run -p 5001:5001 \
  -e EE_PROJECT=your-project \
  -e GOOGLE_APPLICATION_CREDENTIALS_JSON='...' \
  bloombly-api
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EE_PROJECT` | No | Google Cloud project ID for Earth Engine |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | No | Service account JSON for Earth Engine |
| `PORT` | No | API port (default: 5001) |
| `DEBUG` | No | Debug mode (default: True) |

### Production Considerations

- Use **Gunicorn** for production WSGI server
- Enable **CORS** for frontend domain only
- Set up **rate limiting** for API endpoints
- Implement **caching** for Earth Engine queries
- Use **CDN** for static frontend assets
- Enable **HTTPS** for all endpoints

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **Report Bugs** - Open an issue with detailed reproduction steps
2. **Suggest Features** - Propose new capabilities or improvements
3. **Submit Data** - Share bloom observations or datasets
4. **Improve Docs** - Fix typos, add examples, clarify explanations
5. **Write Code** - Submit pull requests with new features or fixes

### Development Setup

```bash
# Fork the repository
git clone https://github.com/YOUR_USERNAME/bloombly.git
cd bloombly

# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and commit
git commit -m "Add amazing feature"

# Push to your fork
git push origin feature/amazing-feature

# Open a Pull Request
```

### Code Style

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: ES6+, consistent formatting
- **Documentation**: Clear comments, update README
- **Testing**: Add tests for new features

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Bloombly Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

### NASA & Space Apps Challenge

- **NASA Earth Science Division** - For satellite data and Earth Engine access
- **NASA Space Apps Challenge** - For inspiring this project
- **GLOBE Observer** - For citizen science bloom observations

### Data Providers

- **MODIS Science Team** - NDVI and LST datasets
- **NASA SMAP Team** - Soil moisture data
- **Japanese Meteorological Agency** - 73 years of sakura data
- **Phenobase** - European phenology observations
- **Kaggle Community** - Cherry blossom datasets

### Open Source Projects

- **Flask** - Web framework
- **scikit-learn** - Machine learning library
- **Google Earth Engine** - Satellite data platform
- **Globe.gl** - 3D globe visualization
- **pandas/numpy** - Data processing

### Scientific References

1. **Baskerville, G.L., and Emin, P. (1969)** - Growing Degree Days methodology
2. **White et al. (2009)** - Spring phenology detection methods
3. **Richardson et al. (2013)** - Climate impacts on plant phenology
4. **Primack et al. (2009)** - Long-term phenological observations

### Team

- Enrique Ayala
- Paola Hernandez
- Ian Hernandez
- Carlos Vazquez
- Lucca Traslosheros

---
### Citation

If you use this project in your research, please cite:

```bibtex
@software{bloombly2025,
  title = {Bloombly: Machine Learning-Based Wildflower Bloom Prediction},
  author = {Bloombly Team},
  year = {2025},
  url = {https://github.com/KIKW12/bloombly},
  note = {NASA Space Apps Challenge Project}
}
```

---

<p align="center">
  <b>Built with ❤️ for the NASA Space Apps Challenge 2025</b><br>
  🌸 Predicting blooms, preserving biodiversity, understanding climate change 🌍
</p>

<p align="center">
  <a href="#-overview">Back to Top ↑</a>
</p>
