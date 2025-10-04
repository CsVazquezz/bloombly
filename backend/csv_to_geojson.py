import csv
import json
from datetime import datetime

def determine_season(date_str):
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Fall"
        else:
            return "Winter"
    except:
        return "Spring"  # default

def csv_to_geojson(csv_file_path):
    features = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                
                # Create a small square around the point (0.01 degrees ~ 1km)
                size = 0.005  # half size in degrees
                square_coords = [
                    [lon - size, lat - size],
                    [lon + size, lat - size],
                    [lon + size, lat + size],
                    [lon - size, lat + size],
                    [lon - size, lat - size]  # close the polygon
                ]
                
                properties = {
                    "id": None,
                    "Site": row['scientificName'],
                    "Family": row['family'],
                    "Genus": row['genus'],
                    "Season": determine_season(row['date']),
                    "Area": 1.0  # Approximate area for the square
                }
                
                geometry = {
                    "type": "MultiPolygon",
                    "coordinates": [[square_coords]]
                }
                
                feature = {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": geometry
                }
                
                features.append(feature)
            except (ValueError, KeyError):
                # Skip rows with invalid data
                continue
    
    geojson = {
        "type": "FeatureCollection",
        "name": "Flowering_sites_US",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": features
    }
    
    return geojson

if __name__ == "__main__":
    csv_path = "data.csv"
    geojson_data = csv_to_geojson(csv_path)
    
    with open("output.geojson", "w") as f:
        json.dump(geojson_data, f, indent=2)
    
    print("GeoJSON file created: output.geojson")