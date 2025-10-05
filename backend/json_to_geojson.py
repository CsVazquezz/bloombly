import os
import json
import pandas as pd
from shapely.geometry import Point, MultiPoint
import math
import numpy as np
from sklearn.cluster import DBSCAN

# Collect all features
features = []

def calculate_area(polygon):
    """Calculate approximate area in km²"""
    coords = list(polygon.exterior.coords)
    area = 0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon_diff = math.radians(lon2 - lon1)
        area += lon_diff * (2 + math.sin(lat1_rad) + math.sin(lat2_rad))
    area = abs(area * 6371**2 / 2)
    return area

def create_polygon(points, num_observations):
    """Create a simple polygon with minimal coordinates"""
    # Much smaller scale factor for tighter polygons
    scale = min(1.0 + (num_observations / 20), 3.0)  # Scale 1.0-3.0x
    
    if len(points) == 1:
        # Single point: create a simple square (4 vertices instead of 16)
        lon, lat = points[0]
        radius = 0.05 * scale  # Base ~5km, up to ~15km
        
        # Square around the point
        coords = [
            [lon - radius, lat - radius],
            [lon + radius, lat - radius],
            [lon + radius, lat + radius],
            [lon - radius, lat + radius],
            [lon - radius, lat - radius]  # Close the polygon
        ]
        from shapely.geometry import Polygon
        return Polygon(coords)
    
    elif len(points) == 2:
        # Two points: create a simple rectangle/diamond (4-5 vertices)
        lon1, lat1 = points[0]
        lon2, lat2 = points[1]
        
        width = 0.03 * scale
        
        # Calculate perpendicular offset
        dx = lon2 - lon1
        dy = lat2 - lat1
        length = math.sqrt(dx**2 + dy**2)
        
        if length < 0.001:  # Points too close, make square
            radius = 0.05 * scale
            coords = [
                [lon1 - radius, lat1 - radius],
                [lon1 + radius, lat1 - radius],
                [lon1 + radius, lat1 + radius],
                [lon1 - radius, lat1 + radius],
                [lon1 - radius, lat1 - radius]
            ]
        else:
            # Perpendicular vector for width
            px = -dy / length * width
            py = dx / length * width
            
            # Rectangle around the line
            coords = [
                [lon1 + px, lat1 + py],
                [lon2 + px, lat2 + py],
                [lon2 - px, lat2 - py],
                [lon1 - px, lat1 - py],
                [lon1 + px, lat1 + py]  # Close
            ]
        
        from shapely.geometry import Polygon
        return Polygon(coords)
    
    else:
        # Multiple points: use simplified convex hull (already minimal vertices)
        lons = [p[0] for p in points]
        lats = [p[1] for p in points]
        spread = math.sqrt((max(lons) - min(lons))**2 + (max(lats) - min(lats))**2)
        
        # Create convex hull (naturally minimal vertices)
        hull = MultiPoint(points).convex_hull
        
        # Small buffer with LOW resolution for fewer vertices
        buffer_size = 0.02 * scale * (1 + spread * 0.2)
        buffered = hull.buffer(buffer_size, resolution=3)  # Only 3 segments per corner
        
        # Simplify to reduce vertices further
        simplified = buffered.simplify(0.01, preserve_topology=True)
        
        return simplified

# Scan family directories
for family in os.listdir('../data/jsonl'):
    family_path = os.path.join('../data/jsonl', family)
    if not os.path.isdir(family_path):
        continue
        
    for file in os.listdir(family_path):
        if not file.endswith('.jsonl'):
            continue
            
        genus = file[:-6]  # remove .jsonl
        filepath = os.path.join(family_path, file)
        
        try:
            df = pd.read_json(filepath, lines=True)
        except:
            continue
            
        if df.empty:
            continue
        
        # Group by year and season
        for (year, season), group_df in df.groupby(['year', 'season']):
            # Collect points
            points = [(row['long'], row['lat']) for _, row in group_df.iterrows() 
                     if pd.notna(row['long']) and pd.notna(row['lat'])]
            
            if not points:
                continue
            
            num_observations = len(points)
            
            # Cluster nearby points
            if len(points) >= 3:
                coords_array = np.array(points)
                # Tighter clustering for smaller, more concentrated polygons
                clustering = DBSCAN(eps=0.15, min_samples=2).fit(coords_array)  # Reduced from 0.3
                
                # Group by cluster
                clusters = {}
                for idx, label in enumerate(clustering.labels_):
                    key = label if label != -1 else f"noise_{idx}"
                    if key not in clusters:
                        clusters[key] = []
                    clusters[key].append(points[idx])
            else:
                # Too few points, treat each separately
                clusters = {i: [points[i]] for i in range(len(points))}
            
            # Create polygon for each cluster
            for cluster_points in clusters.values():
                poly = create_polygon(cluster_points, num_observations)
                
                if poly.geom_type != 'Polygon':
                    continue
                
                # Calculate area
                area = calculate_area(poly)
                
                # Create feature
                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": None,
                        "Family": family,
                        "Genus": genus,
                        "Season": str(season).capitalize(),
                        "Area": round(area, 3),
                        "year": int(year)
                    },
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [[ [[x, y] for x, y in poly.exterior.coords] ]]
                    }
                }
                features.append(feature)

# Create GeoJSON
geojson = {
    "type": "FeatureCollection",
    "name": "Flowering_sites",
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
    },
    "features": features
}

# Write to file
with open("../data/geojson/flowering_sites.geojson", "w") as f:
    json.dump(geojson, f, indent=2)

print(f"GeoJSON file created with {len(features)} features")