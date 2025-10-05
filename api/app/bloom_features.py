"""
Advanced bloom prediction features based on ecological and environmental factors.

This module implements specialized feature calculations for bloom prediction:
1. Spring start date detection using NDVI analysis
2. Growing Degree Days (GDD) calculation using Baskerville-Emin method
3. Soil water availability estimation using wilting point method
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def calculate_spring_start_date(ndvi_time_series, dates):
    """
    Calculate the spring start date using 5-day moving average NDVI analysis.
    
    Spring begins when the smoothed NDVI exceeds the winter average and maintains
    a sustained increase for the longest period of the year.
    
    Parameters:
    -----------
    ndvi_time_series : array-like
        Daily NDVI values
    dates : array-like
        Corresponding dates for the NDVI values
    
    Returns:
    --------
    dict : {
        'spring_start_day': int (day of year),
        'days_since_spring_start': int,
        'is_spring_active': bool,
        'winter_ndvi_baseline': float
    }
    """
    if len(ndvi_time_series) < 90:  # Need at least 3 months of data
        return {
            'spring_start_day': 80,  # Default to around March 21
            'days_since_spring_start': 0,
            'is_spring_active': False,
            'winter_ndvi_baseline': 0.2
        }
    
    # Convert to numpy arrays
    ndvi = np.array(ndvi_time_series)
    dates = pd.to_datetime(dates)
    
    # Step 1: Calculate 5-day moving average to smooth the data
    # This removes daily inconsistencies and shows the real growth trend
    window_size = 5
    ndvi_smoothed = pd.Series(ndvi).rolling(window=window_size, center=True).mean().bfill().ffill().values
    
    # Step 2: Calculate winter NDVI baseline (Dec 1 - March 1)
    # Extract day of year for each date
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Winter period: day 335-365 (Dec 1-31) or day 1-60 (Jan 1 - Mar 1)
    winter_mask = (day_of_year >= 335) | (day_of_year <= 60)
    
    if np.sum(winter_mask) > 0:
        winter_ndvi_baseline = np.mean(ndvi_smoothed[winter_mask])
    else:
        # Fallback: use lowest quartile
        winter_ndvi_baseline = np.percentile(ndvi_smoothed, 25)
    
    # Step 3: Find spring start - when smoothed NDVI exceeds winter baseline
    # and maintains increase (longest sustained growth period)
    threshold = winter_ndvi_baseline * 1.1  # 10% above winter baseline
    
    # Find all points above threshold
    above_threshold = ndvi_smoothed > threshold
    
    # Calculate NDVI trend (derivative)
    ndvi_trend = np.gradient(ndvi_smoothed)
    
    # Find sustained growth periods (above threshold AND increasing)
    spring_candidates = above_threshold & (ndvi_trend > 0)
    
    # Find the longest consecutive spring period
    spring_start_idx = None
    max_spring_length = 0
    current_length = 0
    current_start = 0
    
    for i, is_spring in enumerate(spring_candidates):
        if is_spring:
            if current_length == 0:
                current_start = i
            current_length += 1
        else:
            if current_length > max_spring_length:
                max_spring_length = current_length
                spring_start_idx = current_start
            current_length = 0
    
    # Check the last sequence
    if current_length > max_spring_length:
        spring_start_idx = current_start
    
    # Determine spring start day
    if spring_start_idx is not None and spring_start_idx < len(dates):
        spring_start_day = day_of_year[spring_start_idx]
        spring_start_date = dates[spring_start_idx]
    else:
        # Default to typical spring equinox
        spring_start_day = 80  # Around March 21
        spring_start_date = datetime(dates[0].year, 3, 21)
    
    # Calculate days since spring start (for current date - assume last date in series)
    current_date = dates[-1]
    days_since_spring = (current_date - spring_start_date).days
    is_spring_active = days_since_spring >= 0 and days_since_spring <= 90  # Spring lasts ~90 days
    
    return {
        'spring_start_day': int(spring_start_day),
        'days_since_spring_start': max(0, days_since_spring),
        'is_spring_active': bool(is_spring_active),
        'winter_ndvi_baseline': float(winter_ndvi_baseline)
    }


def calculate_growing_degree_days(tmax, tmin, tbase=0):
    """
    Calculate Growing Degree Days using the Baskerville-Emin (1969) method.
    
    GDD measures the accumulation of ambient heat above a base temperature threshold
    for vegetation development.
    
    Formula: GDD = [(Tmax + Tmin) / 2] - Tbase
    
    For this application, Tbase = 0°C (threshold for general plant growth)
    
    Parameters:
    -----------
    tmax : float or array-like
        Maximum temperature(s) in Celsius
    tmin : float or array-like
        Minimum temperature(s) in Celsius
    tbase : float, default=0
        Base temperature threshold in Celsius
    
    Returns:
    --------
    float or array : Growing degree days
        Higher values = more growth potential
    """
    # Convert to numpy arrays for vectorized operations
    tmax = np.array(tmax)
    tmin = np.array(tmin)
    
    # Calculate average temperature
    tavg = (tmax + tmin) / 2
    
    # Calculate GDD (only positive values - can't have negative growth)
    gdd = np.maximum(0, tavg - tbase)
    
    return gdd


def calculate_accumulated_gdd(tmax_series, tmin_series, tbase=0, days=30):
    """
    Calculate accumulated GDD over a period (default 30 days).
    
    Parameters:
    -----------
    tmax_series : array-like
        Daily maximum temperatures in Celsius
    tmin_series : array-like
        Daily minimum temperatures in Celsius
    tbase : float, default=0
        Base temperature threshold
    days : int, default=30
        Number of days to accumulate over
    
    Returns:
    --------
    float : Accumulated growing degree days
    """
    if len(tmax_series) < days:
        # If we don't have enough data, calculate for available days
        days = len(tmax_series)
    
    # Use the last 'days' values
    tmax_recent = tmax_series[-days:]
    tmin_recent = tmin_series[-days:]
    
    # Calculate daily GDD
    daily_gdd = calculate_growing_degree_days(tmax_recent, tmin_recent, tbase)
    
    # Sum over the period
    accumulated_gdd = np.sum(daily_gdd)
    
    return float(accumulated_gdd)


def calculate_wilting_point(field_capacity_percent):
    """
    Calculate the Permanent Wilting Point (PWP) using the empirical formula.
    
    PWP is the soil water content where plants can no longer extract water
    and will permanently wilt.
    
    Formula: PWP% = (CC% * 0.74) - 5
    
    Where:
    - PWP% = Water content percentage at wilting point
    - CC% = Field capacity (water holding capacity of soil)
    
    Note: Wilting point varies between plant species. Needle-leaved plants
    (like cacti) are naturally protected from water evaporation.
    
    Parameters:
    -----------
    field_capacity_percent : float
        Field capacity as a percentage (typical range: 10-40%)
    
    Returns:
    --------
    float : Wilting point as a percentage
    """
    pwp = (field_capacity_percent * 0.74) - 5
    return max(0, pwp)  # Can't be negative


def calculate_soil_water_days(soil_water_content, field_capacity_percent):
    """
    Calculate available soil water days.
    
    Soil water day = 0 if soil_water_content < wilting_point
    Soil water day = soil_water_content - wilting_point, if soil_water_content > wilting_point
    
    The wilting point is calculated using field capacity:
    Field Capacity (CC) = moisture content when soil is saturated and drained
    
    Parameters:
    -----------
    soil_water_content : float
        Current soil water content (mm or percentage)
    field_capacity_percent : float
        Field capacity as percentage (typical: 15-35% depending on soil type)
        - Sandy soil: ~10-15%
        - Loam soil: ~20-30%
        - Clay soil: ~30-40%
    
    Returns:
    --------
    dict : {
        'soil_water_days': float,
        'wilting_point': float,
        'water_stress': bool,
        'available_water_ratio': float
    }
    """
    # Calculate wilting point from field capacity
    wilting_point = calculate_wilting_point(field_capacity_percent)
    
    # Calculate available water days
    if soil_water_content < wilting_point:
        soil_water_days = 0
        water_stress = True
    else:
        soil_water_days = soil_water_content - wilting_point
        water_stress = False
    
    # Calculate ratio of available water to field capacity
    if field_capacity_percent > 0:
        available_water_ratio = soil_water_days / field_capacity_percent
    else:
        available_water_ratio = 0
    
    return {
        'soil_water_days': float(soil_water_days),
        'wilting_point': float(wilting_point),
        'water_stress': bool(water_stress),
        'available_water_ratio': float(min(1.0, available_water_ratio))
    }


def estimate_field_capacity_from_soil_type(soil_type='loam'):
    """
    Estimate field capacity based on soil type.
    
    Parameters:
    -----------
    soil_type : str
        One of: 'sand', 'loamy_sand', 'sandy_loam', 'loam', 'silt_loam', 
                'sandy_clay', 'clay_loam', 'silty_clay', 'clay'
    
    Returns:
    --------
    float : Field capacity percentage
    """
    soil_fc = {
        'sand': 12,
        'loamy_sand': 15,
        'sandy_loam': 20,
        'loam': 25,
        'silt_loam': 30,
        'sandy_clay': 28,
        'clay_loam': 32,
        'silty_clay': 35,
        'clay': 38
    }
    
    return soil_fc.get(soil_type.lower(), 25)  # Default to loam


def get_soil_texture_encoding(soil_type='loam'):
    """
    Get numeric encoding for soil texture classification.
    
    Soil texture affects water retention, nutrient availability, and root penetration.
    Encoding follows typical soil classification from coarse to fine texture.
    
    Parameters:
    -----------
    soil_type : str
        Soil texture classification
    
    Returns:
    --------
    dict : {
        'soil_texture_code': int (0-8),
        'sand_percent': float,
        'clay_percent': float,
        'silt_percent': float
    }
    """
    soil_properties = {
        'sand': {'code': 0, 'sand': 90, 'clay': 5, 'silt': 5},
        'loamy_sand': {'code': 1, 'sand': 80, 'clay': 10, 'silt': 10},
        'sandy_loam': {'code': 2, 'sand': 65, 'clay': 10, 'silt': 25},
        'loam': {'code': 3, 'sand': 40, 'clay': 20, 'silt': 40},
        'silt_loam': {'code': 4, 'sand': 20, 'clay': 15, 'silt': 65},
        'sandy_clay': {'code': 5, 'sand': 50, 'clay': 35, 'silt': 15},
        'clay_loam': {'code': 6, 'sand': 30, 'clay': 35, 'silt': 35},
        'silty_clay': {'code': 7, 'sand': 10, 'clay': 45, 'silt': 45},
        'clay': {'code': 8, 'sand': 20, 'clay': 60, 'silt': 20}
    }
    
    props = soil_properties.get(soil_type.lower(), soil_properties['loam'])
    
    return {
        'soil_texture_code': props['code'],
        'sand_percent': props['sand'],
        'clay_percent': props['clay'],
        'silt_percent': props['silt']
    }


def calculate_smoothed_ndvi(ndvi_time_series, window_size=5):
    """
    Calculate smoothed NDVI using moving average.
    
    Smoothing removes daily inconsistencies and noise from satellite data,
    revealing the true vegetation growth trend.
    
    Parameters:
    -----------
    ndvi_time_series : array-like
        Daily NDVI values
    window_size : int, default=5
        Size of the moving average window (days)
    
    Returns:
    --------
    dict : {
        'ndvi_smoothed_current': float (current smoothed value),
        'ndvi_smoothed_mean': float (mean of smoothed series),
        'ndvi_smoothed_trend': float (trend/slope of smoothed series)
    }
    """
    if len(ndvi_time_series) < window_size:
        # Not enough data for smoothing
        current_val = ndvi_time_series[-1] if len(ndvi_time_series) > 0 else 0.3
        return {
            'ndvi_smoothed_current': current_val,
            'ndvi_smoothed_mean': current_val,
            'ndvi_smoothed_trend': 0.0
        }
    
    # Apply moving average
    ndvi_smoothed = pd.Series(ndvi_time_series).rolling(
        window=window_size, center=True
    ).mean().bfill().ffill().values
    
    # Calculate trend (slope)
    x = np.arange(len(ndvi_smoothed))
    if len(x) > 1:
        trend = np.polyfit(x, ndvi_smoothed, 1)[0]
    else:
        trend = 0.0
    
    return {
        'ndvi_smoothed_current': float(ndvi_smoothed[-1]),
        'ndvi_smoothed_mean': float(np.mean(ndvi_smoothed)),
        'ndvi_smoothed_trend': float(trend)
    }


def calculate_soil_temperature_gdd(soil_temp_max_series, soil_temp_min_series, tbase=10):
    """
    Calculate Growing Degree Days using soil temperature (Baskerville-Emin method).
    
    Soil temperature GDD is often more accurate than air temperature GDD for root
    development and some bloom triggers, as soil temperature is more stable and
    directly affects root activity.
    
    Formula: Soil GDD = [(Tsoil_max + Tsoil_min) / 2] - Tbase
    
    Parameters:
    -----------
    soil_temp_max_series : array-like
        Daily maximum soil temperature (°C)
    soil_temp_min_series : array-like
        Daily minimum soil temperature (°C)
    tbase : float, default=10
        Base temperature threshold (10°C is common for many plants)
    
    Returns:
    --------
    dict : {
        'soil_gdd_current': float (current day GDD),
        'soil_gdd_accumulated_30d': float (30-day accumulation)
    }
    """
    if len(soil_temp_max_series) == 0 or len(soil_temp_min_series) == 0:
        return {
            'soil_gdd_current': 0.0,
            'soil_gdd_accumulated_30d': 0.0
        }
    
    # Convert to numpy arrays
    tmax = np.array(soil_temp_max_series)
    tmin = np.array(soil_temp_min_series)
    
    # Calculate average soil temperature
    tavg = (tmax + tmin) / 2
    
    # Calculate GDD (only positive values)
    gdd = np.maximum(0, tavg - tbase)
    
    # Current day GDD
    current_gdd = float(gdd[-1]) if len(gdd) > 0 else 0.0
    
    # Accumulated GDD over last 30 days (or available days)
    days_to_accumulate = min(30, len(gdd))
    accumulated_gdd = float(np.sum(gdd[-days_to_accumulate:]))
    
    return {
        'soil_gdd_current': current_gdd,
        'soil_gdd_accumulated_30d': accumulated_gdd
    }


def calculate_reference_evapotranspiration(temp_mean, temp_max, temp_min, 
                                          latitude, day_of_year, 
                                          wind_speed=2.0, humidity=50):
    """
    Calculate reference evapotranspiration (ET0) using simplified Hargreaves equation.
    
    ET measures water loss from soil and plant surfaces. High ET can stress plants
    and reduce bloom probability if soil moisture is insufficient.
    
    Simplified Hargreaves formula:
    ET0 = 0.0023 × (Tmean + 17.8) × (Tmax - Tmin)^0.5 × Ra
    
    Where Ra is extraterrestrial radiation (calculated from latitude and day of year)
    
    Parameters:
    -----------
    temp_mean : float
        Mean daily temperature (°C)
    temp_max : float
        Maximum daily temperature (°C)
    temp_min : float
        Minimum daily temperature (°C)
    latitude : float
        Latitude in degrees
    day_of_year : int
        Day of year (1-365)
    wind_speed : float, optional
        Wind speed in m/s (default 2.0)
    humidity : float, optional
        Relative humidity percentage (default 50)
    
    Returns:
    --------
    dict : {
        'et0_hargreaves': float (mm/day),
        'et0_adjusted': float (mm/day, adjusted for humidity),
        'water_deficit_index': float (ET0 / available water ratio)
    }
    """
    # Calculate extraterrestrial radiation (Ra) - simplified
    # Solar declination
    delta = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
    
    # Convert latitude to radians
    lat_rad = latitude * np.pi / 180
    
    # Sunset hour angle
    omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    
    # Relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    
    # Extraterrestrial radiation (MJ/m²/day)
    Ra = (24 * 60 / np.pi) * 0.082 * dr * (
        omega_s * np.sin(lat_rad) * np.sin(delta) +
        np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    )
    
    # Hargreaves ET0 (mm/day)
    temp_range = max(temp_max - temp_min, 0.1)  # Avoid zero
    et0_hargreaves = 0.0023 * (temp_mean + 17.8) * np.sqrt(temp_range) * Ra
    
    # Adjust for humidity (simple correction factor)
    # Higher humidity = lower ET
    humidity_factor = 1.0 - ((humidity - 50) / 200)  # Centered at 50%
    et0_adjusted = et0_hargreaves * humidity_factor
    
    return {
        'et0_hargreaves': float(max(0, et0_hargreaves)),
        'et0_adjusted': float(max(0, et0_adjusted)),
        'extraterrestrial_radiation': float(Ra)
    }


def calculate_comprehensive_bloom_features(environmental_data, location_data=None):
    """
    Calculate all advanced bloom features from environmental and location data.
    
    Parameters:
    -----------
    environmental_data : dict
        Dictionary containing:
        - 'ndvi_time_series': list of NDVI values (optional, for spring detection)
        - 'dates': list of dates corresponding to NDVI (optional)
        - 'tmax': max temperature or list of max temps (air)
        - 'tmin': min temperature or list of min temps (air)
        - 'soil_tmax': max soil temperature or list (optional)
        - 'soil_tmin': min soil temperature or list (optional)
        - 'soil_moisture': soil water content (mm or %)
        - 'field_capacity': field capacity % (optional, will estimate if missing)
        - 'soil_type': soil texture classification (optional)
        - 'latitude': latitude for ET calculation (optional)
        - 'day_of_year': day of year for ET calculation (optional)
        - 'humidity': relative humidity % (optional)
    
    location_data : dict, optional
        Additional location info like soil type for better estimation
    
    Returns:
    --------
    dict : Comprehensive feature set including all calculated features
    """
    features = {}
    
    # 1. Spring start date features (if NDVI time series available)
    if 'ndvi_time_series' in environmental_data and 'dates' in environmental_data:
        spring_features = calculate_spring_start_date(
            environmental_data['ndvi_time_series'],
            environmental_data['dates']
        )
        features.update(spring_features)
    else:
        # Use defaults if no time series
        features.update({
            'spring_start_day': 80,
            'days_since_spring_start': 0,
            'is_spring_active': False,
            'winter_ndvi_baseline': 0.2
        })
    
    # 2. Smoothed NDVI features
    if 'ndvi_time_series' in environmental_data:
        smoothed_ndvi_features = calculate_smoothed_ndvi(
            environmental_data['ndvi_time_series'],
            window_size=5
        )
        features.update(smoothed_ndvi_features)
    else:
        # Use raw NDVI if available
        current_ndvi = environmental_data.get('ndvi_mean', 0.4)
        features.update({
            'ndvi_smoothed_current': current_ndvi,
            'ndvi_smoothed_mean': current_ndvi,
            'ndvi_smoothed_trend': 0.0
        })
    
    # 3. Air Temperature Growing Degree Days
    tmax = environmental_data.get('tmax', 20)
    tmin = environmental_data.get('tmin', 10)
    
    # If we have time series, calculate accumulated GDD
    if isinstance(tmax, (list, np.ndarray)) and isinstance(tmin, (list, np.ndarray)) and len(tmax) > 0:
        features['gdd_accumulated_30d'] = calculate_accumulated_gdd(tmax, tmin, tbase=0, days=30)
        # Also calculate current day GDD
        features['gdd_current'] = float(calculate_growing_degree_days(tmax[-1], tmin[-1], tbase=0))
    else:
        # Single value GDD
        if isinstance(tmax, (list, np.ndarray)) and len(tmax) > 0:
            tmax = tmax[-1]
            tmin = tmin[-1] if isinstance(tmin, (list, np.ndarray)) and len(tmin) > 0 else 10
        features['gdd_current'] = float(calculate_growing_degree_days(tmax, tmin, tbase=0))
        features['gdd_accumulated_30d'] = features['gdd_current'] * 30  # Rough estimate
    
    # 4. Soil Temperature Growing Degree Days
    soil_tmax = environmental_data.get('soil_tmax', [])
    soil_tmin = environmental_data.get('soil_tmin', [])
    
    if len(soil_tmax) > 0 and len(soil_tmin) > 0:
        soil_gdd_features = calculate_soil_temperature_gdd(soil_tmax, soil_tmin, tbase=10)
        features.update(soil_gdd_features)
    else:
        # Estimate from air temperature (soil temp is typically more stable)
        # Rule of thumb: soil temp lags air temp and has smaller range
        if isinstance(tmax, (list, np.ndarray)):
            soil_temp_avg = (np.mean(tmax) + np.mean(tmin)) / 2 * 0.9  # Slightly cooler
        else:
            soil_temp_avg = (tmax + tmin) / 2 * 0.9
        features['soil_gdd_current'] = max(0, soil_temp_avg - 10)
        features['soil_gdd_accumulated_30d'] = features['soil_gdd_current'] * 30
    
    # Average soil temperature for reference
    if isinstance(tmax, (list, np.ndarray)) and len(tmax) > 0:
        features['soil_temp_mean'] = (np.mean(tmax) + np.mean(tmin)) / 2 * 0.9
    else:
        features['soil_temp_mean'] = (tmax + tmin) / 2 * 0.9
    
    # 5. Soil water availability
    soil_moisture = environmental_data.get('soil_moisture', 20)
    
    # Determine field capacity
    if 'field_capacity' in environmental_data:
        field_capacity = environmental_data['field_capacity']
    elif 'soil_type' in environmental_data:
        field_capacity = estimate_field_capacity_from_soil_type(environmental_data['soil_type'])
    elif location_data and 'soil_type' in location_data:
        field_capacity = estimate_field_capacity_from_soil_type(location_data['soil_type'])
    else:
        field_capacity = 25  # Default to loam
    
    soil_water_features = calculate_soil_water_days(soil_moisture, field_capacity)
    features.update(soil_water_features)
    
    # 6. Soil texture encoding
    soil_type = environmental_data.get('soil_type', 
                                       location_data.get('soil_type', 'loam') if location_data else 'loam')
    soil_texture_features = get_soil_texture_encoding(soil_type)
    features.update(soil_texture_features)
    
    # 7. Evapotranspiration
    if 'latitude' in environmental_data and 'day_of_year' in environmental_data:
        # Get temperature values for ET calculation
        if isinstance(tmax, (list, np.ndarray)) and len(tmax) > 0:
            temp_max_val = tmax[-1]
            temp_min_val = tmin[-1] if len(tmin) > 0 else 10
            temp_mean_val = (temp_max_val + temp_min_val) / 2
        else:
            temp_max_val = tmax if not isinstance(tmax, (list, np.ndarray)) else 20
            temp_min_val = tmin if not isinstance(tmin, (list, np.ndarray)) else 10
            temp_mean_val = (temp_max_val + temp_min_val) / 2
        
        et_features = calculate_reference_evapotranspiration(
            temp_mean_val,
            temp_max_val,
            temp_min_val,
            environmental_data['latitude'],
            environmental_data['day_of_year'],
            humidity=environmental_data.get('humidity', 50)
        )
        features.update(et_features)
        
        # Calculate water deficit index (ET relative to available water)
        if features['soil_water_days'] > 0:
            features['water_deficit_index'] = et_features['et0_adjusted'] / features['soil_water_days']
        else:
            features['water_deficit_index'] = 10.0  # High water stress
    else:
        # Use defaults
        features.update({
            'et0_hargreaves': 4.0,  # Typical ET for temperate climate
            'et0_adjusted': 4.0,
            'extraterrestrial_radiation': 25.0,
            'water_deficit_index': 0.2
        })
    
    return features


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Bloom Feature Calculations")
    print("=" * 60)
    
    # Test 1: Spring start date detection
    print("\n1. Spring Start Date Detection")
    print("-" * 40)
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    # Simulate NDVI with low winter, rising spring, high summer
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    ndvi_simulated = 0.25 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    ndvi_simulated = np.maximum(0.1, ndvi_simulated)  # Floor at 0.1
    
    spring_info = calculate_spring_start_date(ndvi_simulated, dates)
    print(f"Spring start day: {spring_info['spring_start_day']}")
    print(f"Days since spring: {spring_info['days_since_spring_start']}")
    print(f"Is spring active: {spring_info['is_spring_active']}")
    print(f"Winter baseline: {spring_info['winter_ndvi_baseline']:.3f}")
    
    # Test 2: Growing Degree Days
    print("\n2. Growing Degree Days (GDD)")
    print("-" * 40)
    tmax_sample = [22, 24, 23, 25, 26, 24, 23]
    tmin_sample = [12, 14, 13, 15, 16, 14, 13]
    gdd = calculate_growing_degree_days(tmax_sample, tmin_sample)
    print(f"Daily GDD values: {gdd}")
    print(f"Average daily GDD: {np.mean(gdd):.2f}")
    
    accumulated = calculate_accumulated_gdd(tmax_sample, tmin_sample, days=7)
    print(f"7-day accumulated GDD: {accumulated:.2f}")
    
    # Test 3: Soil water days
    print("\n3. Soil Water Days")
    print("-" * 40)
    for soil_content in [10, 15, 20, 25, 30]:
        result = calculate_soil_water_days(soil_content, field_capacity_percent=25)
        print(f"Soil content {soil_content}%: {result['soil_water_days']:.1f} days, "
              f"Stress: {result['water_stress']}, Ratio: {result['available_water_ratio']:.2f}")
    
    # Test 4: Comprehensive features
    print("\n4. Comprehensive Feature Calculation")
    print("-" * 40)
    env_data = {
        'ndvi_time_series': ndvi_simulated,
        'dates': dates,
        'tmax': tmax_sample,
        'tmin': tmin_sample,
        'soil_moisture': 22
    }
    
    all_features = calculate_comprehensive_bloom_features(env_data)
    print("All calculated features:")
    for key, value in all_features.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
