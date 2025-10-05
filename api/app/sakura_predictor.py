"""
Sakura Bloom Predictor

Specialized predictor for sakura (cherry blossom) bloom dates using comprehensive
environmental and temporal features.

Features two models:
1. Global Model - For Prunus species worldwide
2. Japan Model - Specialized for Prunus × yedoensis in Japan
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from typing import Dict, List, Optional, Tuple
import json


class SakuraBloomPredictor:
    """
    Predictor for sakura bloom dates using trained machine learning models
    
    This class handles:
    - Loading pre-trained global and Japan-specific models
    - Making predictions for bloom dates
    - Feature preparation and scaling
    - Model selection based on location/species
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize Sakura Bloom Predictor
        
        Args:
            models_dir: Directory containing trained model files
        """
        self.models_dir = models_dir
        self.global_model = None
        self.japan_model = None
        self.global_scaler = None
        self.japan_scaler = None
        self.feature_columns = []
        self.metadata = {}
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models and scalers from disk"""
        
        # Load global model
        global_path = os.path.join(self.models_dir, 'sakura_global_model.pkl')
        if os.path.exists(global_path):
            try:
                data = joblib.load(global_path)
                self.global_model = data['model']
                self.global_scaler = data['scaler']
                self.feature_columns = data['feature_columns']
                self.metadata['global'] = data.get('metadata', {})
                print(f"✓ Loaded global sakura model from {global_path}")
            except Exception as e:
                print(f"⚠ Failed to load global model: {e}")
        else:
            print(f"⚠ Global model not found at {global_path}")
        
        # Load Japan model
        japan_path = os.path.join(self.models_dir, 'sakura_japan_model.pkl')
        if os.path.exists(japan_path):
            try:
                data = joblib.load(japan_path)
                self.japan_model = data['model']
                self.japan_scaler = data['scaler']
                # Feature columns should be the same
                if not self.feature_columns:
                    self.feature_columns = data['feature_columns']
                self.metadata['japan'] = data.get('metadata', {})
                print(f"✓ Loaded Japan sakura model from {japan_path}")
            except Exception as e:
                print(f"⚠ Failed to load Japan model: {e}")
        else:
            print(f"⚠ Japan model not found at {japan_path}")
        
        if not self.global_model and not self.japan_model:
            raise RuntimeError(
                f"No models found in {self.models_dir}. "
                f"Please train models first using train_sakura_model.py"
            )
    
    def predict_bloom_date(
        self,
        latitude: float,
        longitude: float,
        year: int,
        species: str = "Prunus × yedoensis",
        environmental_data: Optional[Dict] = None,
        use_japan_model: bool = None
    ) -> Dict:
        """
        Predict bloom date for sakura trees
        
        Args:
            latitude: Latitude of location
            longitude: Longitude of location  
            year: Year to predict for
            species: Scientific name of species (default: Prunus × yedoensis)
            environmental_data: Dictionary with environmental features
            use_japan_model: Force use of Japan model (auto-detected if None)
            
        Returns:
            Dictionary with prediction results including:
            - bloom_day_of_year: Predicted day of year
            - bloom_date: Predicted date
            - confidence: Model confidence
            - model_used: Which model was used
        """
        
        # Determine which model to use
        is_japan = self._is_japan_location(latitude, longitude, species)
        
        if use_japan_model is None:
            use_japan_model = is_japan
        
        # Select model and scaler
        if use_japan_model and self.japan_model:
            model = self.japan_model
            scaler = self.japan_scaler
            model_name = "Japan-specific"
        elif self.global_model:
            model = self.global_model
            scaler = self.global_scaler
            model_name = "Global"
        else:
            raise RuntimeError("No suitable model available for prediction")
        
        # Prepare features
        features = self._prepare_prediction_features(
            latitude=latitude,
            longitude=longitude,
            year=year,
            species=species,
            environmental_data=environmental_data
        )
        
        # Ensure all required features are present
        feature_df = pd.DataFrame([features])
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0  # Default value for missing features
        
        # Reorder columns to match training
        feature_df = feature_df[self.feature_columns]
        
        # Scale features
        X_scaled = scaler.transform(feature_df)
        
        # Make prediction
        bloom_day = model.predict(X_scaled)[0]
        
        # Convert to date
        bloom_date = self._day_of_year_to_date(int(round(bloom_day)), year)
        
        # Calculate confidence (based on model's prediction consistency)
        # For ensemble models, we can use the variance across trees
        confidence = self._calculate_confidence(model, X_scaled)
        
        return {
            'bloom_day_of_year': int(round(bloom_day)),
            'bloom_date': bloom_date.strftime('%Y-%m-%d'),
            'bloom_month': bloom_date.month,
            'bloom_day': bloom_date.day,
            'year': year,
            'model_used': model_name,
            'confidence': confidence,
            'location': {
                'latitude': latitude,
                'longitude': longitude
            },
            'species': species,
            'is_japan_location': is_japan
        }
    
    def predict_bloom_window(
        self,
        latitude: float,
        longitude: float,
        year: int,
        species: str = "Prunus × yedoensis",
        environmental_data: Optional[Dict] = None,
        window_days: int = 7
    ) -> Dict:
        """
        Predict bloom date with uncertainty window
        
        Args:
            latitude: Latitude of location
            longitude: Longitude of location
            year: Year to predict for
            species: Scientific name of species
            environmental_data: Environmental features
            window_days: Size of uncertainty window in days
            
        Returns:
            Dictionary with prediction and date range
        """
        prediction = self.predict_bloom_date(
            latitude=latitude,
            longitude=longitude,
            year=year,
            species=species,
            environmental_data=environmental_data
        )
        
        bloom_date = datetime.strptime(prediction['bloom_date'], '%Y-%m-%d')
        
        # Adjust window based on confidence
        adjusted_window = int(window_days * (1 - prediction['confidence']))
        adjusted_window = max(3, min(adjusted_window, 14))  # Between 3 and 14 days
        
        early_date = bloom_date - timedelta(days=adjusted_window)
        late_date = bloom_date + timedelta(days=adjusted_window)
        
        prediction['bloom_window'] = {
            'early_date': early_date.strftime('%Y-%m-%d'),
            'peak_date': bloom_date.strftime('%Y-%m-%d'),
            'late_date': late_date.strftime('%Y-%m-%d'),
            'window_days': adjusted_window * 2
        }
        
        return prediction
    
    def batch_predict(
        self,
        locations: List[Dict],
        year: int,
        species: str = "Prunus × yedoensis"
    ) -> List[Dict]:
        """
        Make predictions for multiple locations
        
        Args:
            locations: List of location dicts with 'latitude' and 'longitude'
            year: Year to predict for
            species: Scientific name of species
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for loc in locations:
            try:
                pred = self.predict_bloom_date(
                    latitude=loc['latitude'],
                    longitude=loc['longitude'],
                    year=year,
                    species=species,
                    environmental_data=loc.get('environmental_data')
                )
                pred['location_name'] = loc.get('name', 'Unknown')
                predictions.append(pred)
            except Exception as e:
                print(f"⚠ Failed to predict for {loc}: {e}")
                predictions.append({
                    'location_name': loc.get('name', 'Unknown'),
                    'error': str(e)
                })
        
        return predictions
    
    def _prepare_prediction_features(
        self,
        latitude: float,
        longitude: float,
        year: int,
        species: str,
        environmental_data: Optional[Dict]
    ) -> Dict:
        """
        Prepare feature dictionary for prediction
        
        Args:
            latitude: Latitude
            longitude: Longitude
            year: Year
            species: Species name
            environmental_data: Optional environmental data
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic location and time features
        features['latitude'] = latitude
        features['longitude'] = longitude
        features['year'] = year
        features['year_normalized'] = (year - 1970) / (2024 - 1970)  # Normalize to 0-1
        
        # If environmental data is provided, use it
        if environmental_data:
            features.update(environmental_data)
        else:
            # Use default/estimated values based on location and time
            # In production, you'd fetch this from Earth Engine or weather APIs
            
            # Estimate typical values for Japan (if Japan location)
            if self._is_japan_location(latitude, longitude, species):
                # Typical Japanese spring conditions
                features.update({
                    'temp_avg_30d': 10.0,  # °C
                    'temp_max_30d': 15.0,
                    'temp_min_30d': 5.0,
                    'precip_total_30d': 100.0,  # mm
                    'humidity_avg_30d': 70.0,  # %
                    'solar_avg_30d': 15.0,  # MJ/m²
                    'gdd_30d': 150.0,  # Growing degree days
                    'photoperiod_at_bloom': 12.0,  # hours
                    'elevation_m': 50.0,  # meters (typical for urban areas)
                })
            else:
                # Generic temperate climate values
                features.update({
                    'temp_avg_30d': 8.0,
                    'temp_max_30d': 13.0,
                    'temp_min_30d': 3.0,
                    'precip_total_30d': 80.0,
                    'humidity_avg_30d': 65.0,
                    'solar_avg_30d': 12.0,
                    'gdd_30d': 120.0,
                    'photoperiod_at_bloom': 12.0,
                    'elevation_m': 100.0,
                })
        
        return features
    
    def _is_japan_location(self, latitude: float, longitude: float, species: str) -> bool:
        """
        Determine if location is in Japan
        
        Args:
            latitude: Latitude
            longitude: Longitude
            species: Species name
            
        Returns:
            True if location is in Japan
        """
        # Japan approximate bounds
        japan_lat_min, japan_lat_max = 24.0, 46.0
        japan_lon_min, japan_lon_max = 122.0, 154.0
        
        is_japan_coords = (
            japan_lat_min <= latitude <= japan_lat_max and
            japan_lon_min <= longitude <= japan_lon_max
        )
        
        is_sakura = 'yedoensis' in species.lower() or 'somei' in species.lower()
        
        return is_japan_coords or is_sakura
    
    def _day_of_year_to_date(self, day_of_year: int, year: int) -> datetime:
        """
        Convert day of year to actual date
        
        Args:
            day_of_year: Day of year (1-366)
            year: Year
            
        Returns:
            datetime object
        """
        return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    
    def _calculate_confidence(self, model, X_scaled: np.ndarray) -> float:
        """
        Calculate prediction confidence
        
        For Gradient Boosting, we can use the variance across staged predictions
        
        Args:
            model: Trained model
            X_scaled: Scaled features
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # For GradientBoostingRegressor, check prediction variance across stages
            if hasattr(model, 'estimators_'):
                # Sample predictions from different stages
                n_estimators = len(model.estimators_)
                sample_stages = [
                    int(n_estimators * 0.5),
                    int(n_estimators * 0.75),
                    int(n_estimators * 1.0) - 1
                ]
                
                predictions = []
                for stage in sample_stages:
                    pred = model.predict(X_scaled, n_iter=stage + 1)[0]
                    predictions.append(pred)
                
                # Lower variance = higher confidence
                variance = np.var(predictions)
                
                # Map variance to confidence (inverse relationship)
                # Variance of 10 days = low confidence (0.3)
                # Variance of 1 day = high confidence (0.9)
                confidence = 1.0 / (1.0 + variance / 5.0)
                
                return min(0.95, max(0.3, confidence))
        except:
            pass
        
        # Default moderate confidence
        return 0.7
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models
        
        Returns:
            Dictionary with model information
        """
        info = {
            'models_loaded': [],
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns[:10],  # First 10 features
            'metadata': self.metadata
        }
        
        if self.global_model:
            info['models_loaded'].append('global')
        
        if self.japan_model:
            info['models_loaded'].append('japan')
        
        return info
    
    def get_feature_importance(self, model_type: str = 'global') -> pd.DataFrame:
        """
        Get feature importance from model
        
        Args:
            model_type: 'global' or 'japan'
            
        Returns:
            DataFrame with feature importances
        """
        model = self.japan_model if model_type == 'japan' else self.global_model
        
        if not model:
            raise ValueError(f"Model '{model_type}' not loaded")
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not support feature importances")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


# Global predictor instance (singleton pattern)
_predictor_instance = None


def get_sakura_predictor(models_dir: str = 'models') -> SakuraBloomPredictor:
    """
    Get or create global sakura predictor instance
    
    Args:
        models_dir: Directory containing models
        
    Returns:
        SakuraBloomPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = SakuraBloomPredictor(models_dir=models_dir)
    
    return _predictor_instance
