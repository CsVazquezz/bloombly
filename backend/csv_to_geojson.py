import csv
csv.field_size_limit(1000000)  # Increase field size limit
import json
import sys
from datetime import datetime
import math

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

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def approx_area(min_lon, max_lon, min_lat, max_lat):
    # Approximate area in square km
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    avg_lat = (min_lat + max_lat) / 2
    # 1 degree lat ≈ 111 km, lon ≈ 111 * cos(lat) km
    area = lat_diff * 111 * lon_diff * 111 * math.cos(math.radians(avg_lat))
    return area

def csv_to_geojson(csv_file_path):
    points = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                season = determine_season(row['date'])
                points.append((lon, lat, row['scientificName'], row['family'], row['genus'], season))
            except (ValueError, KeyError):
                continue
    
    # Simple clustering: group points within 0.1 degrees
    threshold = 0.1
    clusters = []
    for point in points:
        found = False
        for cluster in clusters:
            if any(distance(point[:2], p[:2]) < threshold for p in cluster):
                cluster.append(point)
                found = True
                break
        if not found:
            clusters.append([point])
    
    features = []
    for cluster in clusters:
        if not cluster:
            continue
        lons = [p[0] for p in cluster]
        lats = [p[1] for p in cluster]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Expand the bounding box based on number of points
        margin = 0.01 * (len(cluster) ** 0.5)
        min_lon -= margin
        max_lon += margin
        min_lat -= margin
        max_lat += margin
        
        # Create a rectangle polygon
        square_coords = [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]  # close
        ]
        
        # Use the first point's properties
        first = cluster[0]
        area = approx_area(min_lon, max_lon, min_lat, max_lat)
        properties = {
            "id": None,
            "Site": first[2],  # scientificName
            "Family": first[3],
            "Genus": first[4],
            "Season": first[5],
            "Area": area  # Approximate area in square km
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