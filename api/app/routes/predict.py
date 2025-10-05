from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import logging
from bloom_predictor import EnhancedBloomPredictor
from bloom_predictor_v2 import ImprovedBloomPredictor
import config
from models.schemas import BloomsPredictionQuery, EnvironmentalDataQuery
from pydantic import ValidationError

predict_bp = Blueprint('predict', __name__)

predictor = None
predictor_v2 = None

def get_predictor(version='v1'):
    """Initializes and returns the bloom predictor instance."""
    global predictor, predictor_v2
    
    if version == 'v2':
        if predictor_v2 is None:
            logging.info("Initializing Improved Bloom Predictor v2 (Learning Bloom Dynamics)...")
            try:
                # Try to load pre-trained model first for fast startup
                import os
                model_path = os.path.join(os.path.dirname(__file__), '..', 'bloom_model_v2.pkl')
                if os.path.exists(model_path):
                    logging.info(f"  Loading pre-trained model from {model_path}...")
                    predictor_v2 = ImprovedBloomPredictor(load_pretrained=model_path)
                    logging.info("✓ Bloom Predictor v2 loaded from pre-trained model!")
                else:
                    logging.info("  No pre-trained model found, training new model...")
                    predictor_v2 = ImprovedBloomPredictor()
                    logging.info("✓ Bloom Predictor v2 ready!")
                return predictor_v2
            except Exception as e:
                logging.error(f"⚠ Failed to initialize v2: {e}")
                logging.info("  Falling back to v1...")
                version = 'v1'
        else:
            return predictor_v2
    
    # v1 (default)
    if predictor is None:
        logging.info("Initializing Enhanced Bloom Predictor v1...")
        predictor = EnhancedBloomPredictor()
        logging.info("✓ Bloom Predictor v1 ready!")
    return predictor

def get_aoi_bounds(aoi_type, aoi_state, aoi_country, bbox):
    """Returns the bounding box for a given AOI."""
    if aoi_type == 'state' and aoi_state in config.STATE_BOUNDS:
        return config.STATE_BOUNDS[aoi_state]
    if aoi_type == 'country' and aoi_country.lower() in config.COUNTRY_BOUNDS:
        return config.COUNTRY_BOUNDS[aoi_country.lower()]
    if aoi_type == 'bbox' and bbox and len(bbox) == 4:
        return {'min_lon': bbox[0], 'min_lat': bbox[1], 'max_lon': bbox[2], 'max_lat': bbox[3]}
    return None

def create_geojson_response(features, metadata):
    """Creates a GeoJSON FeatureCollection response."""
    return jsonify({
        "type": "FeatureCollection",
        "features": features,
        "metadata": metadata
    })

@predict_bp.route('/blooms', methods=['GET'])
def predict_blooms():
    """Predicts blooms for a given date or date range."""
    try:
        query_params = BloomsPredictionQuery(**request.args)
    except ValidationError as e:
        logging.error(f"Validation error in /blooms endpoint: {e.errors()}")
        return jsonify(error=e.errors()), 400

    try:
        # Determine which model version to use
        version = 'v2' if query_params.method in ['v2', 'bloom_dynamics'] else 'v1'
        predictor = get_predictor(version)
        aoi_bounds = get_aoi_bounds(query_params.aoi_type, query_params.aoi_state, query_params.aoi_country, query_params.bbox)
        
        confidence_threshold = float(request.args.get('confidence', '0.3'))

        if query_params.date:
            target_date = query_params.date
            
            if version == 'v2':
                # Use improved bloom dynamics model
                predictions = predictor.predict_blooms_for_date(
                    target_date, 
                    aoi_bounds, 
                    num_predictions=100,
                    confidence_threshold=confidence_threshold
                )
                model_info = "ML model trained on bloom dynamics with temporal features and environmental factors (v2)"
            elif query_params.method == 'statistical':
                predictions = predictor.predict_blooms_statistical(target_date, aoi_bounds, config.NUM_PREDICTIONS)
                model_info = "Statistical sampling from historical data"
            else:
                predictions = predictor.predict_blooms_enhanced(target_date, aoi_bounds, config.NUM_PREDICTIONS)
                model_info = "Enhanced ML model with environmental factors (v1)"
            
            metadata = {
                "prediction_date": query_params.date.isoformat(),
                "aoi_type": query_params.aoi_type,
                "prediction_type": "single_date",
                "method": query_params.method,
                "model_version": version,
                "confidence_threshold": confidence_threshold if version == 'v2' else None,
                "model_info": model_info
            }
            return create_geojson_response(predictions, metadata)

        elif query_params.start_date and query_params.end_date:
            start_date = query_params.start_date
            end_date = query_params.end_date

            if (end_date - start_date).days > config.MAX_TIME_SERIES_DAYS:
                end_date = start_date + timedelta(days=config.MAX_TIME_SERIES_DAYS)

            predictions = predictor.predict_blooms_time_series(
                start_date, end_date, aoi_bounds, config.TIME_SERIES_INTERVAL_DAYS
            )
            
            all_features = [feature for date in predictions for feature in predictions[date]]
            
            metadata = {
                "start_date": query_params.start_date.isoformat(),
                "end_date": query_params.end_date.isoformat(),
                "aoi_type": query_params.aoi_type,
                "prediction_type": "time_series",
                "interval_days": config.TIME_SERIES_INTERVAL_DAYS,
                "method": query_params.method,
                "model_info": "Enhanced ML model with environmental factors" if query_params.method == 'enhanced' else "Statistical sampling from historical data"
            }
            return create_geojson_response(all_features, metadata)

        else:
            return jsonify(error="Must provide either 'date' or both 'start_date' and 'end_date'"), 400

    except Exception as e:
        import traceback
        logging.error(f"API Error in /blooms endpoint: {e}")
        logging.error(traceback.format_exc())
        return jsonify(error=str(e), details="Check server logs for traceback"), 500

@predict_bp.route('/environmental', methods=['GET'])
def get_environmental_data():
    """Get environmental data for a specific location and date"""
    try:
        query_params = EnvironmentalDataQuery(**request.args)
    except ValidationError as e:
        logging.error(f"Validation error in /environmental endpoint: {e.errors()}")
        return jsonify(error=e.errors()), 400

    try:
        predictor = get_predictor()
        env_data = predictor.get_environmental_data(query_params.lat, query_params.lon, query_params.date)

        return jsonify({
            "location": {"lat": query_params.lat, "lon": query_params.lon},
            "date": query_params.date.isoformat(),
            "environmental_data": env_data,
            "data_source": "earth_engine" if predictor.use_earth_engine else "climate_normals"
        })

    except Exception as e:
        logging.error(f"API Error in /environmental endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@predict_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the prediction model"""
    try:
        version = request.args.get('version', 'v2')
        predictor = get_predictor(version)
        
        if version == 'v2':
            # Get bloom windows info
            bloom_windows = {}
            for species, info in predictor.species_bloom_windows.items():
                bloom_windows[species] = {
                    'peak_day': int(info['mean_day']),
                    'range_days': [int(info['min_day']), int(info['max_day'])],
                    'observation_count': info['count']
                }
            
            # Calculate totals based on available data
            positive_count = len(predictor.historical_blooms) if hasattr(predictor, 'historical_blooms') and predictor.historical_blooms is not None else 0
            negative_count = len(predictor.negative_examples) if hasattr(predictor, 'negative_examples') and predictor.negative_examples is not None else 0
            
            # Use feature_data if available, otherwise calculate from positive + negative
            if hasattr(predictor, 'feature_data') and predictor.feature_data is not None:
                total_samples = len(predictor.feature_data)
            else:
                total_samples = positive_count + negative_count
            
            model_info = {
                "model_version": "v2",
                "model_type": "Gradient Boosting Classifier (Bloom Dynamics)",
                "description": "ML model trained on bloom vs no-bloom classification with temporal features",
                "is_training": predictor.is_training if hasattr(predictor, 'is_training') else False,
                "features": predictor.feature_columns if hasattr(predictor, 'feature_columns') else [],
                "feature_count": len(predictor.feature_columns) if hasattr(predictor, 'feature_columns') else 0,
                "training_data": {
                    "positive_examples": positive_count,
                    "negative_examples": negative_count,
                    "total_samples": total_samples
                },
                "species_bloom_windows": bloom_windows,
                "species_count": len(predictor.species_bloom_windows),
                "environmental_data_source": "Google Earth Engine (30-day averages)" if predictor.use_earth_engine else "Climate Normals",
                "environmental_factors": [
                    "temperature (mean, max, min, range)",
                    "precipitation (total, mean)",
                    "ndvi (mean, max, trend)",
                    "elevation",
                    "growing_degree_days",
                    "moisture_index",
                    "vegetation_health"
                ],
                "prediction_methods": ["v2 (bloom_dynamics)", "enhanced (v1)", "statistical (v1)"],
                "validation": "Time-series cross-validation with ROC-AUC metric",
                "improvements_over_v1": [
                    "Predicts bloom probability directly (not species as proxy)",
                    "Trained on both bloom and no-bloom examples",
                    "Temporal feature engineering (lag, trends, seasonal encoding)",
                    "Environmental data aggregated over 30-day windows",
                    "Time-series aware validation"
                ]
            }
        else:
            # v1 model info
            model_info = {
                "model_version": "v1",
                "model_type": "Enhanced Gradient Boosting Classifier",
                "is_training": predictor.is_training if hasattr(predictor, 'is_training') else False,
                "features": predictor.feature_columns if hasattr(predictor, 'feature_columns') else [],
                "training_data_size": len(predictor.historical_data) if hasattr(predictor, 'historical_data') and not predictor.historical_data.empty else 0,
                "species_count": len(predictor.species_patterns),
                "environmental_data_source": "Google Earth Engine" if predictor.use_earth_engine else "Climate Normals",
                "environmental_factors": [
                    "temperature", "precipitation", "ndvi", "elevation"
                ],
                "prediction_methods": ["enhanced", "statistical"],
                "supported_states": ["Texas", "California", "Florida", "New York"]
            }

        return jsonify(model_info)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500