from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from bloom_predictor import EnhancedBloomPredictor

predict_bp = Blueprint('predict', __name__)

# Initialize predictor (lazy loading)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        print("Initializing Enhanced Bloom Predictor...")
        predictor = EnhancedBloomPredictor()
        print("Enhanced Bloom Predictor ready!")
    return predictor

@predict_bp.route('/blooms', methods=['GET'])
def predict_blooms():
    try:
        # Get parameters
        aoi_type = request.args.get('aoi_type', 'global')
        date_str = request.args.get('date')
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        aoi_country = request.args.get('aoi_country', '')
        aoi_state = request.args.get('aoi_state', '')
        prediction_method = request.args.get('method', 'enhanced')  # 'enhanced' or 'statistical'

        predictor = get_predictor()

        # Set AOI bounds based on type
        aoi_bounds = None
        if aoi_type == 'state':
            # Define rough bounds for common states
            state_bounds = {
                'Texas': {'min_lat': 25.8, 'max_lat': 36.5, 'min_lon': -106.6, 'max_lon': -93.5},
                'California': {'min_lat': 32.5, 'max_lat': 42.0, 'min_lon': -124.4, 'max_lon': -114.1},
                'Florida': {'min_lat': 24.5, 'max_lat': 31.0, 'min_lon': -87.6, 'max_lon': -79.8},
                'New York': {'min_lat': 40.5, 'max_lat': 45.0, 'min_lon': -79.8, 'max_lon': -71.8}
            }
            aoi_bounds = state_bounds.get(aoi_state)
        elif aoi_type == 'country':
            # Rough US bounds
            if aoi_country.lower() in ['united states', 'usa', 'us']:
                aoi_bounds = {'min_lat': 24.4, 'max_lat': 49.4, 'min_lon': -125.0, 'max_lon': -66.9}
            elif aoi_country.lower() == 'mexico':
                aoi_bounds = {'min_lat': 14.5, 'max_lat': 32.7, 'min_lon': -118.4, 'max_lon': -86.7}

        if date_str:
            # Single date prediction
            target_date = datetime.strptime(date_str, '%Y-%m-%d')

            if prediction_method == 'statistical':
                predictions = predictor.predict_blooms_statistical(target_date, aoi_bounds, num_predictions=50)
            else:
                predictions = predictor.predict_blooms_enhanced(target_date, aoi_bounds, num_predictions=50)

            return jsonify({
                "type": "FeatureCollection",
                "features": predictions,
                "metadata": {
                    "prediction_date": date_str,
                    "aoi_type": aoi_type,
                    "prediction_type": "single_date",
                    "method": prediction_method,
                    "model_info": "Enhanced ML model with environmental factors" if prediction_method == 'enhanced' else "Statistical sampling from historical data"
                }
            })

        elif start_date_str and end_date_str:
            # Time series prediction
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            # Limit to reasonable range (max 90 days)
            if (end_date - start_date).days > 90:
                end_date = start_date + timedelta(days=90)

            predictions = predictor.predict_blooms_time_series(
                start_date, end_date, aoi_bounds, interval_days=7
            )

            # Convert to GeoJSON
            all_features = []
            for date, features in predictions.items():
                for feature in features:
                    feature['properties']['date'] = date
                    all_features.append(feature)

            return jsonify({
                "type": "FeatureCollection",
                "features": all_features,
                "metadata": {
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "aoi_type": aoi_type,
                    "prediction_type": "time_series",
                    "interval_days": 7,
                    "method": prediction_method,
                    "model_info": "Enhanced ML model with environmental factors" if prediction_method == 'enhanced' else "Statistical sampling from historical data"
                }
            })

        else:
            return jsonify({"error": "Must provide either 'date' for single prediction or 'start_date' and 'end_date' for time series"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@predict_bp.route('/environmental', methods=['GET'])
def get_environmental_data():
    """Get environmental data for a specific location and date"""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        date_str = request.args.get('date')

        if not date_str:
            return jsonify({"error": "Date parameter is required"}), 400

        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        predictor = get_predictor()

        env_data = predictor.get_environmental_data(lat, lon, target_date)

        return jsonify({
            "location": {"lat": lat, "lon": lon},
            "date": date_str,
            "environmental_data": env_data,
            "data_source": "earth_engine" if predictor.use_earth_engine else "climate_normals"
        })

    except ValueError as e:
        return jsonify({"error": "Invalid latitude, longitude, or date format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@predict_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the prediction model"""
    try:
        predictor = get_predictor()

        model_info = {
            "model_type": "Enhanced Gradient Boosting Classifier",
            "features": predictor.feature_columns if hasattr(predictor, 'feature_columns') else [],
            "training_data_size": len(predictor.historical_data) if not predictor.historical_data.empty else 0,
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
        return jsonify({"error": str(e)}), 500