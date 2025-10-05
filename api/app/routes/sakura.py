"""
Sakura (Cherry Blossom) Prediction Routes (Flask)

API endpoints for predicting sakura bloom dates using specialized models
"""

from flask import Blueprint, request, jsonify
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from app.sakura_predictor import get_sakura_predictor
except ImportError:
    from sakura_predictor import get_sakura_predictor

sakura_bp = Blueprint('sakura', __name__)


@sakura_bp.route('/', methods=['GET'])
def sakura_info():
    """
    Get information about sakura prediction models
    """
    try:
        predictor = get_sakura_predictor(models_dir='app/models')
        info = predictor.get_model_info()
        
        return jsonify({
            "service": "Sakura Bloom Prediction",
            "description": "Specialized prediction for cherry blossom bloom dates",
            "models": info['models_loaded'],
            "features": {
                "count": info['feature_count'],
                "sample": info['features']
            },
            "metadata": info['metadata'],
            "endpoints": {
                "predict": "/api/sakura/predict",
                "batch": "/api/sakura/predict/batch",
                "japan": "/api/sakura/predict/japan",
                "features": "/api/sakura/features/<model_type>",
                "compare": "/api/sakura/compare/models"
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error loading model info: {str(e)}"}), 500


@sakura_bp.route('/predict', methods=['POST'])
def predict_sakura_bloom():
    """
    Predict sakura bloom date for a specific location and year
    
    Request JSON:
    {
        "latitude": 35.68,
        "longitude": 139.65,
        "year": 2025,
        "species": "Prunus × yedoensis",
        "include_window": true,
        "environmental_data": {...},  // optional
        "use_japan_model": true  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'year']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        year = int(data['year'])
        species = data.get('species', 'Prunus × yedoensis')
        include_window = data.get('include_window', False)
        environmental_data = data.get('environmental_data')
        use_japan_model = data.get('use_japan_model')
        
        # Validate ranges
        if not (-90 <= latitude <= 90):
            return jsonify({"error": "Latitude must be between -90 and 90"}), 400
        if not (-180 <= longitude <= 180):
            return jsonify({"error": "Longitude must be between -180 and 180"}), 400
        if not (1900 <= year <= 2100):
            return jsonify({"error": "Year must be between 1900 and 2100"}), 400
        
        predictor = get_sakura_predictor(models_dir='app/models')
        
        if include_window:
            prediction = predictor.predict_bloom_window(
                latitude=latitude,
                longitude=longitude,
                year=year,
                species=species,
                environmental_data=environmental_data
            )
        else:
            prediction = predictor.predict_bloom_date(
                latitude=latitude,
                longitude=longitude,
                year=year,
                species=species,
                environmental_data=environmental_data,
                use_japan_model=use_japan_model
            )
        
        return jsonify(prediction)
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@sakura_bp.route('/predict/japan', methods=['GET'])
def predict_japan_sakura():
    """
    Predict sakura bloom for Japanese prefectures using the Japan-specific model
    
    Query parameters:
    - prefecture: Japanese prefecture name (required)
    - year: Year to predict (required)
    - include_window: Include bloom window (optional, default false)
    
    Example: /api/sakura/predict/japan?prefecture=Tokyo&year=2025&include_window=true
    """
    
    # Prefecture coordinates (approximate centers)
    prefecture_coords = {
        'tokyo': (35.6762, 139.6503),
        'osaka': (34.6937, 135.5023),
        'kyoto': (35.0116, 135.7681),
        'hokkaido': (43.0642, 141.3469),
        'okinawa': (26.2124, 127.6809),
        'fukuoka': (33.5904, 130.4017),
        'nagoya': (35.1815, 136.9066),
        'sendai': (38.2682, 140.8694),
        'hiroshima': (34.3853, 132.4553),
        'sapporo': (43.0642, 141.3469),
    }
    
    prefecture = request.args.get('prefecture')
    year_str = request.args.get('year')
    include_window = request.args.get('include_window', 'false').lower() == 'true'
    
    if not prefecture:
        return jsonify({"error": "Missing parameter: prefecture"}), 400
    if not year_str:
        return jsonify({"error": "Missing parameter: year"}), 400
    
    try:
        year = int(year_str)
    except ValueError:
        return jsonify({"error": "Year must be an integer"}), 400
    
    if not (1900 <= year <= 2100):
        return jsonify({"error": "Year must be between 1900 and 2100"}), 400
    
    pref_lower = prefecture.lower()
    if pref_lower not in prefecture_coords:
        return jsonify({
            "error": f"Prefecture '{prefecture}' not found",
            "available": list(prefecture_coords.keys())
        }), 400
    
    lat, lon = prefecture_coords[pref_lower]
    
    try:
        predictor = get_sakura_predictor(models_dir='app/models')
        
        if include_window:
            prediction = predictor.predict_bloom_window(
                latitude=lat,
                longitude=lon,
                year=year,
                species="Prunus × yedoensis",
                environmental_data=None
            )
        else:
            prediction = predictor.predict_bloom_date(
                latitude=lat,
                longitude=lon,
                year=year,
                species="Prunus × yedoensis",
                environmental_data=None,
                use_japan_model=True
            )
        
        prediction['prefecture'] = prefecture
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@sakura_bp.route('/predict/batch', methods=['POST'])
def batch_predict_sakura():
    """
    Predict sakura bloom dates for multiple locations
    
    Request JSON:
    {
        "year": 2025,
        "species": "Prunus × yedoensis",
        "locations": [
            {"latitude": 35.68, "longitude": 139.65, "name": "Tokyo"},
            {"latitude": 34.69, "longitude": 135.50, "name": "Osaka"}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if 'year' not in data:
            return jsonify({"error": "Missing required field: year"}), 400
        if 'locations' not in data:
            return jsonify({"error": "Missing required field: locations"}), 400
        
        year = int(data['year'])
        species = data.get('species', 'Prunus × yedoensis')
        locations = data['locations']
        
        if not isinstance(locations, list):
            return jsonify({"error": "locations must be a list"}), 400
        
        predictor = get_sakura_predictor(models_dir='app/models')
        
        predictions = predictor.batch_predict(
            locations=locations,
            year=year,
            species=species
        )
        
        return jsonify({
            "year": year,
            "species": species,
            "predictions": predictions,
            "count": len(predictions)
        })
        
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Batch prediction error: {str(e)}"}), 500


@sakura_bp.route('/features/<model_type>', methods=['GET'])
def get_feature_importance(model_type):
    """
    Get feature importance from trained models
    
    Path parameter:
    - model_type: 'global' or 'japan'
    
    Query parameter:
    - top_n: Number of top features to return (default 20, max 100)
    
    Example: /api/sakura/features/japan?top_n=15
    """
    if model_type not in ['global', 'japan']:
        return jsonify({
            "error": "model_type must be 'global' or 'japan'"
        }), 400
    
    top_n = request.args.get('top_n', 20, type=int)
    if not (1 <= top_n <= 100):
        return jsonify({"error": "top_n must be between 1 and 100"}), 400
    
    try:
        predictor = get_sakura_predictor(models_dir='app/models')
        importance_df = predictor.get_feature_importance(model_type=model_type)
        
        # Convert to dict
        top_features = importance_df.head(top_n).to_dict('records')
        
        return jsonify({
            "model_type": model_type,
            "top_features": top_features,
            "total_features": len(importance_df)
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@sakura_bp.route('/compare/models', methods=['GET'])
def compare_models():
    """
    Compare predictions from both global and Japan models
    
    Query parameters:
    - latitude: Latitude (required)
    - longitude: Longitude (required)
    - year: Year (required)
    
    Example: /api/sakura/compare/models?latitude=35.68&longitude=139.65&year=2025
    """
    latitude_str = request.args.get('latitude')
    longitude_str = request.args.get('longitude')
    year_str = request.args.get('year')
    
    if not all([latitude_str, longitude_str, year_str]):
        return jsonify({"error": "Missing required parameters: latitude, longitude, year"}), 400
    
    try:
        latitude = float(latitude_str)
        longitude = float(longitude_str)
        year = int(year_str)
    except ValueError:
        return jsonify({"error": "Invalid parameter types"}), 400
    
    if not (-90 <= latitude <= 90):
        return jsonify({"error": "Latitude must be between -90 and 90"}), 400
    if not (-180 <= longitude <= 180):
        return jsonify({"error": "Longitude must be between -180 and 180"}), 400
    if not (1900 <= year <= 2100):
        return jsonify({"error": "Year must be between 1900 and 2100"}), 400
    
    try:
        predictor = get_sakura_predictor(models_dir='app/models')
        
        results = {}
        
        # Try global model
        if predictor.global_model:
            try:
                global_pred = predictor.predict_bloom_date(
                    latitude=latitude,
                    longitude=longitude,
                    year=year,
                    species="Prunus × yedoensis",
                    use_japan_model=False
                )
                results['global'] = global_pred
            except Exception as e:
                results['global'] = {"error": str(e)}
        
        # Try Japan model
        if predictor.japan_model:
            try:
                japan_pred = predictor.predict_bloom_date(
                    latitude=latitude,
                    longitude=longitude,
                    year=year,
                    species="Prunus × yedoensis",
                    use_japan_model=True
                )
                results['japan'] = japan_pred
            except Exception as e:
                results['japan'] = {"error": str(e)}
        
        # Calculate difference if both succeeded
        if 'global' in results and 'japan' in results:
            if 'bloom_day_of_year' in results['global'] and 'bloom_day_of_year' in results['japan']:
                diff = results['japan']['bloom_day_of_year'] - results['global']['bloom_day_of_year']
                results['difference'] = {
                    'days': diff,
                    'description': f"Japan model predicts {abs(diff)} days {'earlier' if diff < 0 else 'later'}"
                }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Comparison error: {str(e)}"}), 500
