from flask import Blueprint, request, jsonify
import ee
from earth_engine_utils import get_bloom_data, feature_collection_to_geojson

data_bp = Blueprint('data', __name__)

@data_bp.route('/blooms', methods=['GET'])
def get_blooms():
    try:
        aoi_type = request.args.get('aoi_type', 'global')
        aoi_country = request.args.get('aoi_country', '')
        aoi_state = request.args.get('aoi_state', '')
        date = request.args.get('date', '2024-07-01')  # Single date instead of range

        # Get bloom data from Earth Engine for single date
        blooms_fc = get_bloom_data(aoi_type, date=date,
                                   aoi_country=aoi_country, aoi_state=aoi_state)

        # Convert to GeoJSON
        geojson = feature_collection_to_geojson(blooms_fc)

        return jsonify(geojson)

    except Exception as e:
        return jsonify({"error": str(e)}), 500