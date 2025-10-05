from flask import Blueprint, request, jsonify
import ee
from earth_engine_utils import get_bloom_data, feature_collection_to_geojson

data_bp = Blueprint('data', __name__)

def calculate_prediction_summary(features):
    """Calculate summary statistics for bloom data."""
    if not features:
        return {
            "total_predictions": 0,
            "species_count": 0,
            "families": {},
            "seasons": {},
            "total_area": 0
        }
    
    # Extract data from features
    species = set()
    families = {}
    seasons = {}
    total_area = 0
    
    for feature in features:
        props = feature.get('properties', {})
        
        # Count species and families
        site = props.get('Site')
        family = props.get('Family')
        season = props.get('Season')
        
        if site:
            species.add(site)
        
        if family:
            families[family] = families.get(family, 0) + 1
        
        if season:
            seasons[season] = seasons.get(season, 0) + 1
        
        # Area
        area = props.get('Area', 0)
        total_area += area
    
    # Calculate statistics
    summary = {
        "total_predictions": len(features),
        "species_count": len(species),
        "families": families,
        "family_count": len(families),
        "seasons": seasons,
        "total_area": round(total_area, 2)
    }
    
    return summary

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
        
        # Calculate summary
        summary = calculate_prediction_summary(geojson.get('features', []))
        
        # Add metadata with summary
        metadata = {
            "prediction_date": date,
            "aoi_type": aoi_type,
            "data_source": "satellite",
            "summary": summary
        }
        
        # Return structured response
        return jsonify({
            "type": "FeatureCollection",
            "features": geojson.get('features', []),
            "metadata": metadata
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500