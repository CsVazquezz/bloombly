import ee
import json
from datetime import datetime, timedelta
import os

def validate_service_account_scopes(credentials_dict):
    # Quick heuristic: ensure the credentials are for a service account and have an email
    if not credentials_dict.get('client_email'):
        raise ValueError('Service account JSON missing client_email')
    return True

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')
    return image.addBands(ndvi)

def get_aoi(aoi_type, aoi_country='', aoi_state='', aoi_region=None):
    if aoi_type == 'global':
        aoi = ee.Geometry.Rectangle([-180, -85, 180, 85])
        aoi_name = 'Global'
        map_zoom = 2
        map_label = 'Global AOI'
    elif aoi_type == 'country' and aoi_country == 'Mexico':
        aoi = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_na', aoi_country)).geometry()
        aoi_name = aoi_country
        map_zoom = 5
        map_label = aoi_country + ' AOI'
    elif aoi_type == 'country' and aoi_country == 'United States':
        aoi = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_na', aoi_country)).geometry()
        aoi_name = aoi_country
        map_zoom = 4
        map_label = aoi_country + ' AOI'
    elif aoi_type == 'state' and aoi_country == 'Mexico':
        states = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM0_NAME', aoi_country))
        state = states.filter(ee.Filter.eq('ADM1_NAME', aoi_state))
        aoi = state.geometry()
        aoi_name = aoi_state + '_' + aoi_country
        map_zoom = 7
        map_label = aoi_state + ' AOI'
    elif aoi_type == 'state' and aoi_country == 'United States':
        states = ee.FeatureCollection('TIGER/2018/States').filter(ee.Filter.eq('NAME', aoi_state))
        aoi = states.geometry()
        aoi_name = aoi_state + '_' + aoi_country
        map_zoom = 6
        map_label = aoi_state + ' AOI'
    elif aoi_type == 'region' and aoi_region is not None:
        aoi = aoi_region
        aoi_name = 'CustomRegion'
        map_zoom = 6
        map_label = 'Custom Region AOI'
    else:
        raise ValueError('Unsupported AOI_TYPE or AOI_NAME')

    return aoi, aoi_name, map_zoom, map_label

def get_new_blooms(date, aoi, time_step_days=15, aoi_type='global'):
    date = ee.Date(date)
    current_period_start = date
    current_period_end = date.advance(time_step_days, 'day')
    previous_period_start = date.advance(-time_step_days, 'day')

    # S2 collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .map(add_ndvi)

    current_collection = s2.filterDate(current_period_start, current_period_end)
    previous_collection = s2.filterDate(previous_period_start, current_period_start)

    current_size = current_collection.size()
    previous_size = previous_collection.size()
    empty_features = ee.FeatureCollection([]).set('system:time_start', date.millis())

    has_data = current_size.gt(0).And(previous_size.gt(0))

    def compute_blooms():
        current_image = current_collection.median()
        previous_image = previous_collection.median()
        ndvi_change = current_image.select('ndvi').subtract(previous_image.select('ndvi'))
        bloom_threshold = 0.1
        ndvi_threshold = 0.4
        new_blooms_raster = ndvi_change.gt(bloom_threshold).And(current_image.select('ndvi').gt(ndvi_threshold))

        # Adjust parameters based on AOI type to prevent memory issues
        if aoi_type == 'state':
            scale = 500  # Coarser resolution for states
            max_pixels = 1e8  # 100 million pixels
            min_area = 10000  # 10,000 sq meters minimum
        elif aoi_type == 'country':
            scale = 1000  # Even coarser for countries
            max_pixels = 5e7  # 50 million pixels
            min_area = 50000  # 50,000 sq meters minimum
        else:  # global
            scale = 2000  # Very coarse for global
            max_pixels = 1e7  # 10 million pixels
            min_area = 100000  # 100,000 sq meters minimum

        bloom_vectors = new_blooms_raster.selfMask().reduceToVectors(
            geometry=aoi,
            scale=scale,
            geometryType='polygon',
            eightConnected=False,
            maxPixels=max_pixels
        )

        # Filter out very small polygons and add properties
        site_name = 'Texas_Cotton'  # You can set this dynamically if needed
        bloom_type = 'Wild'
        season = 'Summer'  # Or derive from date
        family = 'Asteraceae'  # Default family for blooms
        genus = 'Symphyotrichum'  # Default genus for blooms

        bloom_vectors = bloom_vectors.map(lambda f: f.set({
            'Site': site_name,
            'Type': bloom_type,
            'Season': season,
            'Family': family,
            'Genus': genus,
            'Area': f.geometry().area(1)
        }))

        # Filter by minimum area to reduce number of features
        bloom_vectors = bloom_vectors.filter(ee.Filter.gt('Area', min_area))

        total_bloom_area = bloom_vectors.aggregate_sum('Area')
        superbloom_threshold = 1000000  # 1,000,000 sq meters
        is_superbloom = ee.Number(total_bloom_area).gt(superbloom_threshold)

        # Cast to MultiPolygon
        bloom_vectors = bloom_vectors.map(lambda f: f.setGeometry(
            ee.Algorithms.If(
                ee.String(f.geometry().type()).compareTo('Polygon').eq(0),
                ee.Geometry.MultiPolygon([f.geometry().coordinates()]),
                f.geometry()
            )
        ))

        return bloom_vectors.set('system:time_start', date.millis()) \
                           .set('total_bloom_area', total_bloom_area) \
                           .set('isSuperbloom', is_superbloom)

    result = ee.Algorithms.If(has_data, compute_blooms(), empty_features)
    return ee.FeatureCollection(result)

def get_bloom_data(aoi_type='global', date='2024-07-01', aoi_country='', aoi_state='', aoi_region=None):
    aoi, aoi_name, _, _ = get_aoi(aoi_type, aoi_country=aoi_country, aoi_state=aoi_state, aoi_region=aoi_region)

    # Get blooms for single date
    blooms = get_new_blooms(ee.Date(date), aoi, 15, aoi_type)  # time_step_days=15 for comparison

    return blooms

def feature_collection_to_geojson(fc):
    """Convert Earth Engine FeatureCollection to GeoJSON dict"""
    try:
        # Get the features as a list of dictionaries
        features = fc.getInfo()['features']
        return {
            "type": "FeatureCollection",
            "features": features
        }
    except ee.ee_exception.EEException as e:
        if "User memory limit exceeded" in str(e):
            print(f"Memory limit exceeded for FeatureCollection conversion. Try smaller AOI or time range.")
            return {
                "type": "FeatureCollection",
                "features": [],
                "error": "Memory limit exceeded - try smaller area or shorter time range"
            }
        else:
            print(f"Earth Engine error: {e}")
            return {
                "type": "FeatureCollection",
                "features": [],
                "error": str(e)
            }
    except Exception as e:
        print(f"Error converting to GeoJSON: {e}")
        return {
            "type": "FeatureCollection",
            "features": [],
            "error": str(e)
        }


def get_ndvi_time_series(lat, lon, start_date, end_date, scale=250):
    """
    Get NDVI time series for spring detection analysis.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format
    end_date : str or datetime
        End date in 'YYYY-MM-DD' format
    scale : int
        Scale in meters (default 250m for MODIS NDVI)
    
    Returns:
    --------
    dict : {
        'dates': list of date strings,
        'ndvi': list of NDVI values,
        'success': bool
    }
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        
        # Use MODIS NDVI (16-day composite)
        ndvi_collection = ee.ImageCollection('MODIS/061/MOD13Q1') \
            .filterDate(start_date, end_date) \
            .select('NDVI')
        
        # Extract time series
        def extract_ndvi(image):
            value = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=scale
            ).get('NDVI')
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'ndvi': ee.Number(value).divide(10000)  # Scale to 0-1
            })
        
        time_series = ndvi_collection.map(extract_ndvi)
        
        # Get info
        features = time_series.getInfo()['features']
        
        dates = [f['properties']['date'] for f in features]
        ndvi_values = [f['properties']['ndvi'] if f['properties']['ndvi'] is not None else 0 
                       for f in features]
        
        return {
            'dates': dates,
            'ndvi': ndvi_values,
            'success': True
        }
    except Exception as e:
        print(f"Error getting NDVI time series: {e}")
        return {
            'dates': [],
            'ndvi': [],
            'success': False
        }


def get_temperature_time_series(lat, lon, start_date, end_date, scale=1000):
    """
    Get daily temperature time series for GDD calculation.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format
    end_date : str or datetime
        End date in 'YYYY-MM-DD' format
    scale : int
        Scale in meters (default 1000m for MODIS LST)
    
    Returns:
    --------
    dict : {
        'dates': list of date strings,
        'tmax': list of max temps in Celsius,
        'tmin': list of min temps in Celsius,
        'success': bool
    }
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        
        # MODIS Land Surface Temperature
        lst_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate(start_date, end_date) \
            .select(['LST_Day_1km', 'LST_Night_1km'])
        
        def extract_temps(image):
            temps = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=scale
            )
            
            # Convert from Kelvin to Celsius (MODIS LST is scaled by 0.02)
            # Handle null values with default temperatures
            tmax = ee.Algorithms.If(
                temps.get('LST_Day_1km'),
                ee.Number(temps.get('LST_Day_1km')).multiply(0.02).subtract(273.15),
                20  # Default 20°C if no data
            )
            tmin = ee.Algorithms.If(
                temps.get('LST_Night_1km'),
                ee.Number(temps.get('LST_Night_1km')).multiply(0.02).subtract(273.15),
                10  # Default 10°C if no data
            )
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'tmax': tmax,
                'tmin': tmin
            })
        
        time_series = lst_collection.map(extract_temps)
        features = time_series.getInfo()['features']
        
        dates = [f['properties']['date'] for f in features]
        tmax = [f['properties']['tmax'] if f['properties']['tmax'] is not None else 20 
                for f in features]
        tmin = [f['properties']['tmin'] if f['properties']['tmin'] is not None else 10 
                for f in features]
        
        return {
            'dates': dates,
            'tmax': tmax,
            'tmin': tmin,
            'success': True
        }
    except Exception as e:
        print(f"Error getting temperature time series: {e}")
        return {
            'dates': [],
            'tmax': [],
            'tmin': [],
            'success': False
        }


def get_soil_moisture_data(lat, lon, date, days_before=30, scale=10000):
    """
    Get soil moisture data for soil water availability calculation.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    date : str or datetime
        Target date in 'YYYY-MM-DD' format
    days_before : int
        Number of days before the target date to average
    scale : int
        Scale in meters (default 10000m for SMAP)
    
    Returns:
    --------
    dict : {
        'soil_moisture': float (volumetric water content),
        'field_capacity': float (estimated),
        'success': bool
    }
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        end_date = ee.Date(date)
        start_date = end_date.advance(-days_before, 'day')
        
        # NASA SMAP Soil Moisture - Surface soil moisture (0-5cm)
        # Using SPL4SMGP which is a global soil moisture product
        smap = ee.ImageCollection('NASA/SMAP/SPL4SMGP/007') \
            .filterDate(start_date, end_date) \
            .select('sm_surface')
        
        # Average over the period
        soil_moisture_avg = smap.mean()
        
        result = soil_moisture_avg.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=scale
        )
        
        sm_value = result.get('sm_surface')
        
        if sm_value is not None:
            soil_moisture = float(ee.Number(sm_value).getInfo())
            
            # Estimate field capacity based on soil moisture range
            # Typical field capacity is 1.5-2x the average soil moisture
            field_capacity = soil_moisture * 1.7
        else:
            soil_moisture = 20  # Default
            field_capacity = 25  # Default for loam
        
        return {
            'soil_moisture': soil_moisture,
            'field_capacity': field_capacity,
            'success': sm_value is not None
        }
    except Exception as e:
        print(f"Error getting soil moisture data: {e}")
        # Return defaults based on typical loam soil
        return {
            'soil_moisture': 20,
            'field_capacity': 25,
            'success': False
        }


def get_soil_temperature_data(lat, lon, start_date, end_date, scale=10000):
    """
    Get soil temperature time series for soil temperature GDD calculation.
    
    Uses MODIS Land Surface Temperature as a proxy for soil temperature.
    LST correlates well with soil temperature at shallow depths (0-10cm).
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format
    end_date : str or datetime
        End date in 'YYYY-MM-DD' format
    scale : int
        Scale in meters (default 10000m)
    
    Returns:
    --------
    dict : {
        'dates': list of date strings,
        'soil_tmax': list of max soil temps in Celsius,
        'soil_tmin': list of min soil temps in Celsius,
        'soil_temp_mean': float (average over period),
        'success': bool
    }
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        
        # MODIS Land Surface Temperature (used as proxy for soil temperature)
        # LST Day ~ max soil temp, LST Night ~ min soil temp
        lst_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate(start_date, end_date) \
            .select(['LST_Day_1km', 'LST_Night_1km'])
        
        def extract_soil_temps(image):
            temps = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=scale
            )
            
            # Convert from Kelvin to Celsius (MODIS LST is scaled by 0.02)
            # Apply damping factor: soil temp changes are ~70% of air temp changes
            # Handle null values
            soil_tmax = ee.Algorithms.If(
                temps.get('LST_Day_1km'),
                ee.Number(temps.get('LST_Day_1km')).multiply(0.02).subtract(273.15).multiply(0.7),
                15  # Default 15°C if no data
            )
            soil_tmin = ee.Algorithms.If(
                temps.get('LST_Night_1km'),
                ee.Number(temps.get('LST_Night_1km')).multiply(0.02).subtract(273.15).multiply(0.7),
                8  # Default 8°C if no data
            )
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'soil_tmax': soil_tmax,
                'soil_tmin': soil_tmin
            })
        
        time_series = lst_collection.map(extract_soil_temps)
        features = time_series.getInfo()['features']
        
        dates = [f['properties']['date'] for f in features]
        soil_tmax = [f['properties']['soil_tmax'] if f['properties']['soil_tmax'] is not None else 15 
                     for f in features]
        soil_tmin = [f['properties']['soil_tmin'] if f['properties']['soil_tmin'] is not None else 8 
                     for f in features]
        
        # Calculate mean
        soil_temp_mean = sum([(tmax + tmin) / 2 for tmax, tmin in zip(soil_tmax, soil_tmin)]) / len(soil_tmax) if len(soil_tmax) > 0 else 12
        
        return {
            'dates': dates,
            'soil_tmax': soil_tmax,
            'soil_tmin': soil_tmin,
            'soil_temp_mean': soil_temp_mean,
            'success': True
        }
    except Exception as e:
        print(f"Error getting soil temperature data: {e}")
        return {
            'dates': [],
            'soil_tmax': [],
            'soil_tmin': [],
            'soil_temp_mean': 12,
            'success': False
        }


def get_evapotranspiration_data(lat, lon, start_date, end_date, scale=500):
    """
    Get evapotranspiration data from MODIS MOD16A2 (8-day composite).
    
    ET measures water loss from soil and plant surfaces, critical for
    understanding water stress conditions that affect blooming.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format
    end_date : str or datetime
        End date in 'YYYY-MM-DD' format
    scale : int
        Scale in meters (default 500m for MOD16A2)
    
    Returns:
    --------
    dict : {
        'et_mean': float (mean ET in mm/day),
        'et_total': float (total ET over period in mm),
        'pet_mean': float (mean potential ET in mm/day),
        'success': bool
    }
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        
        # MODIS Evapotranspiration MOD16A2 (8-day composite)
        et_collection = ee.ImageCollection('MODIS/061/MOD16A2GF') \
            .filterDate(start_date, end_date) \
            .select(['ET', 'PET'])  # Actual ET and Potential ET
        
        # Calculate mean ET over the period
        et_mean_image = et_collection.mean()
        et_total_image = et_collection.sum()
        
        result_mean = et_mean_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=scale
        )
        
        result_total = et_total_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=scale
        )
        
        # ET values are in 0.1 mm/8-day, convert to mm/day
        et_mean_val = result_mean.get('ET')
        pet_mean_val = result_mean.get('PET')
        et_total_val = result_total.get('ET')
        
        if et_mean_val is not None:
            # Scale from 0.1 mm/8-day to mm/day
            et_mean = float(ee.Number(et_mean_val).multiply(0.1).divide(8).getInfo())
            pet_mean = float(ee.Number(pet_mean_val).multiply(0.1).divide(8).getInfo())
            et_total = float(ee.Number(et_total_val).multiply(0.1).getInfo())
        else:
            et_mean = 3.5  # Default
            pet_mean = 4.5
            et_total = 100
        
        return {
            'et_mean': et_mean,
            'et_total': et_total,
            'pet_mean': pet_mean,
            'success': et_mean_val is not None
        }
    except Exception as e:
        print(f"Error getting evapotranspiration data: {e}")
        return {
            'et_mean': 3.5,
            'et_total': 100,
            'pet_mean': 4.5,
            'success': False
        }


def get_soil_texture_from_soilgrids(lat, lon, scale=250):
    """
    Get soil texture classification from SoilGrids dataset.
    
    SoilGrids provides global soil property maps including sand, silt, and clay content.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    scale : int
        Scale in meters (default 250m)
    
    Returns:
    --------
    dict : {
        'sand_percent': float,
        'clay_percent': float,
        'silt_percent': float,
        'soil_type': str (texture classification),
        'success': bool
    }
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        
        # OpenLandMap SoilGrids - soil texture fractions at 0-5cm depth
        # Using sand and clay content (g/kg, need to convert to %)
        soil_sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02") \
            .select('b0')  # 0-5cm depth
        soil_clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02") \
            .select('b0')
        # Calculate silt as remainder (100 - sand - clay)
        soil_silt = None  # Will calculate from sand and clay
        
        # Combine into single image
        soil_texture = soil_sand.addBands([soil_clay]) \
            .rename(['sand', 'clay'])
        
        result = soil_texture.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=scale
        )
        
        sand = result.get('sand')
        clay = result.get('clay')
        
        if sand is not None and clay is not None:
            # Convert from g/kg to percentage
            sand_pct = float(ee.Number(sand).divide(10).getInfo())
            clay_pct = float(ee.Number(clay).divide(10).getInfo())
            # Calculate silt as remainder
            silt_pct = max(0, 100 - sand_pct - clay_pct)
            
            # Classify soil texture based on USDA texture triangle
            soil_type = classify_soil_texture(sand_pct, clay_pct, silt_pct)
        else:
            sand_pct = 40  # Default: loam
            clay_pct = 20
            silt_pct = 40
            soil_type = 'loam'
        
        return {
            'sand_percent': sand_pct,
            'clay_percent': clay_pct,
            'silt_percent': silt_pct,
            'soil_type': soil_type,
            'success': sand is not None
        }
    except Exception as e:
        print(f"Error getting soil texture: {e}")
        return {
            'sand_percent': 40,
            'clay_percent': 20,
            'silt_percent': 40,
            'soil_type': 'loam',
            'success': False
        }


def classify_soil_texture(sand, clay, silt):
    """
    Classify soil texture based on USDA texture triangle.
    
    Parameters:
    -----------
    sand : float
        Sand percentage (0-100)
    clay : float
        Clay percentage (0-100)
    silt : float
        Silt percentage (0-100)
    
    Returns:
    --------
    str : Soil texture classification
    """
    # Simplified USDA texture triangle classification
    if sand >= 85:
        return 'sand'
    elif sand >= 70 and clay < 15:
        return 'loamy_sand'
    elif (sand >= 50 and clay >= 7 and clay < 20) or (sand >= 43 and clay < 7):
        return 'sandy_loam'
    elif clay >= 40:
        if sand >= 45:
            return 'sandy_clay'
        elif silt >= 40:
            return 'silty_clay'
        else:
            return 'clay'
    elif clay >= 27:
        if sand >= 20 and sand < 45:
            return 'clay_loam'
        else:
            return 'silty_clay_loam'
    elif silt >= 50 and clay >= 12 and clay < 27:
        return 'silt_loam'
    elif silt >= 80 and clay < 12:
        return 'silt'
    else:
        return 'loam'


def get_comprehensive_environmental_data(lat, lon, date, lookback_days=90):
    """
    Get comprehensive environmental data for advanced bloom feature calculation.
    
    This function retrieves:
    - NDVI time series (for spring detection and smoothed NDVI)
    - Temperature time series (for GDD calculation)
    - Soil temperature time series (for soil GDD)
    - Soil moisture data (for water availability)
    - Soil texture (sand, clay, silt percentages)
    - Evapotranspiration (for water stress analysis)
    - Current environmental conditions
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    date : str or datetime
        Target date
    lookback_days : int
        Days of historical data to retrieve (default 90 for seasonal analysis)
    
    Returns:
    --------
    dict : Comprehensive environmental data including time series and all variables
    """
    if isinstance(date, str):
        target_date = datetime.strptime(date, '%Y-%m-%d')
    else:
        target_date = date
    
    start_date = target_date - timedelta(days=lookback_days)
    
    # Get time series data
    ndvi_ts = get_ndvi_time_series(lat, lon, 
                                   start_date.strftime('%Y-%m-%d'),
                                   target_date.strftime('%Y-%m-%d'))
    
    temp_ts = get_temperature_time_series(lat, lon,
                                          start_date.strftime('%Y-%m-%d'),
                                          target_date.strftime('%Y-%m-%d'))
    
    soil_temp_ts = get_soil_temperature_data(lat, lon,
                                             start_date.strftime('%Y-%m-%d'),
                                             target_date.strftime('%Y-%m-%d'))
    
    soil_data = get_soil_moisture_data(lat, lon, target_date.strftime('%Y-%m-%d'))
    
    # Get soil texture (static data)
    soil_texture = get_soil_texture_from_soilgrids(lat, lon)
    
    # Get evapotranspiration data (for last 30 days)
    et_start = target_date - timedelta(days=30)
    et_data = get_evapotranspiration_data(lat, lon,
                                          et_start.strftime('%Y-%m-%d'),
                                          target_date.strftime('%Y-%m-%d'))
    
    # Combine all data
    result = {
        # NDVI data
        'ndvi_time_series': ndvi_ts['ndvi'] if ndvi_ts['success'] else [],
        'ndvi_dates': ndvi_ts['dates'] if ndvi_ts['success'] else [],
        
        # Air temperature data
        'tmax_series': temp_ts['tmax'] if temp_ts['success'] else [],
        'tmin_series': temp_ts['tmin'] if temp_ts['success'] else [],
        'temp_dates': temp_ts['dates'] if temp_ts['success'] else [],
        
        # Soil temperature data (NEW)
        'soil_tmax_series': soil_temp_ts['soil_tmax'] if soil_temp_ts['success'] else [],
        'soil_tmin_series': soil_temp_ts['soil_tmin'] if soil_temp_ts['success'] else [],
        'soil_temp_mean': soil_temp_ts['soil_temp_mean'],
        
        # Soil moisture and water availability
        'soil_moisture': soil_data['soil_moisture'],
        'field_capacity': soil_data['field_capacity'],
        
        # Soil texture (NEW)
        'sand_percent': soil_texture['sand_percent'],
        'clay_percent': soil_texture['clay_percent'],
        'silt_percent': soil_texture['silt_percent'],
        'soil_type': soil_texture['soil_type'],
        
        # Evapotranspiration (NEW)
        'et_mean': et_data['et_mean'],
        'et_total': et_data['et_total'],
        'pet_mean': et_data['pet_mean'],
        
        # Metadata flags
        'has_time_series': ndvi_ts['success'] and temp_ts['success'],
        'has_soil_data': soil_data['success'],
        'has_soil_temp': soil_temp_ts['success'],
        'has_soil_texture': soil_texture['success'],
        'has_et_data': et_data['success']
    }
    
    return result