import { state } from '../state.js';
import { applyTimelineFilter } from './timeline.js';
import { updateMetrics } from './metricsCard.js';
import { calculatePolygonCenter, setUserLocationMarker, removeUserLocationMarker } from '../globe.js';

let userLocation = null;
let currentDistanceRange = 50; // km
let currentAreaRange = [0, 1000]; // km²
let notificationsEnabled = false;

export function initAdvancedFilters() {
  const container = document.createElement('div');
  container.id = 'advanced-filters-card';
  
  container.innerHTML = `
    <div class="filters-content">
      <!-- Location-Based Filter -->
      <div class="filter-group">
        <div class="filter-group-header">
          <i class="fas fa-location-dot"></i>
          <span>Blooms Near Me</span>
        </div>
        
        <button id="getLocationBtn" class="location-btn">
          <i class="fas fa-crosshairs"></i>
          Get My Location
        </button>
        
        <div id="locationInfo" style="display: none;">
          <div class="filter-control">
            <label class="filter-label">
              <span>Distance Range</span>
              <span class="filter-value" id="distanceValue">50 km</span>
            </label>
            <input type="range" id="distanceRange" min="1" max="500" value="50" step="1">
            <div class="range-labels">
              <span>1 km</span>
              <span>500 km</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Area Size Filter -->
      <div class="filter-group">
        <div class="filter-group-header">
          <i class="fas fa-expand"></i>
          <span>Bloom Area Size</span>
        </div>
        
        <div class="filter-control">
          <label class="filter-label">
            <span>Area Range</span>
            <span class="filter-value" id="areaRangeValue">0 - 1,000+ km²</span>
          </label>
          <div class="dual-range-container">
            <input type="range" id="minAreaRange" class="range-min" min="0" max="1000" value="0" step="10">
            <input type="range" id="maxAreaRange" class="range-max" min="0" max="1000" value="1000" step="10">
          </div>
          <div class="range-labels">
            <span>0 km²</span>
            <span>1000+ km²</span>
          </div>
        </div>
      </div>
      
      <!-- Reset Button -->
      <button id="resetFiltersBtn" class="reset-filters-btn">
        <i class="fas fa-rotate-right"></i>
        Reset All Filters
      </button>
    </div>
  `;
  
  document.body.appendChild(container);
  attachAdvancedFiltersEventListeners();
}

function attachAdvancedFiltersEventListeners() {
  // Get location button
  document.getElementById('getLocationBtn').addEventListener('click', requestUserLocation);
  
  // Distance range slider
  const distanceRange = document.getElementById('distanceRange');
  const distanceValue = document.getElementById('distanceValue');
  
  distanceRange.addEventListener('input', (e) => {
    currentDistanceRange = parseInt(e.target.value);
    distanceValue.textContent = `${currentDistanceRange} km`;
  });
  
  distanceRange.addEventListener('change', () => {
    if (userLocation) {
      applyLocationFilter();
    }
  });
  
  // Area range sliders (dual-handle)
  const minAreaRange = document.getElementById('minAreaRange');
  const maxAreaRange = document.getElementById('maxAreaRange');
  const areaRangeValue = document.getElementById('areaRangeValue');
  
  function updateAreaDisplay() {
    const minVal = parseInt(minAreaRange.value);
    const maxVal = parseInt(maxAreaRange.value);
    
    // Ensure min doesn't exceed max
    if (minVal > maxVal) {
      minAreaRange.value = maxVal;
      currentAreaRange[0] = maxVal;
    } else {
      currentAreaRange[0] = minVal;
    }
    
    currentAreaRange[1] = maxVal;
    
    // Update display - show "1000+" if maxVal is 1000
    if (maxVal === 1000) {
      areaRangeValue.textContent = `${currentAreaRange[0].toLocaleString()} - 1,000+ km²`;
    } else {
      areaRangeValue.textContent = `${currentAreaRange[0].toLocaleString()} - ${currentAreaRange[1].toLocaleString()} km²`;
    }
  }
  
  function applyAreaFilterIfNeeded() {
    // Only apply filter if not default values
    if (currentAreaRange[0] !== 0 || currentAreaRange[1] !== 1000) {
      applyAreaFilter();
    } else {
      // Reset area filter if back to defaults
      state.areaFilteredFeatures = null;
      applyTimelineFilter();
      updateMetrics();
    }
  }
  
  minAreaRange.addEventListener('input', updateAreaDisplay);
  minAreaRange.addEventListener('change', applyAreaFilterIfNeeded);
  
  maxAreaRange.addEventListener('input', () => {
    const minVal = parseInt(minAreaRange.value);
    const maxVal = parseInt(maxAreaRange.value);
    
    // Ensure max doesn't go below min
    if (maxVal < minVal) {
      maxAreaRange.value = minVal;
    }
    updateAreaDisplay();
  });
  maxAreaRange.addEventListener('change', applyAreaFilterIfNeeded);
  
  // Reset filters button
  document.getElementById('resetFiltersBtn').addEventListener('click', resetAllFilters);
}

function requestUserLocation() {
  const btn = document.getElementById('getLocationBtn');
  const locationInfo = document.getElementById('locationInfo');
  const locationText = document.getElementById('locationText');
  
  if (!navigator.geolocation) {
    alert('Geolocation is not supported by your browser');
    return;
  }
  
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Getting location...';
  btn.disabled = true;
  
  navigator.geolocation.getCurrentPosition(
    (position) => {
      userLocation = {
        lat: position.coords.latitude,
        lng: position.coords.longitude
      };
      
      // Show the user location on the globe with a bright blue marker
      setUserLocationMarker(userLocation.lat, userLocation.lng);
      
      locationInfo.style.display = 'block';
      
      btn.innerHTML = '<i class="fas fa-check"></i> Location Acquired';
      btn.classList.add('success');
      
      // Apply location filter immediately
      applyLocationFilter();
      
      setTimeout(() => {
        btn.innerHTML = '<i class="fas fa-crosshairs"></i> Update Location';
        btn.disabled = false;
        btn.classList.remove('success');
      }, 2000);
    },
    (error) => {
      console.error('Error getting location:', error);
      alert('Unable to get your location. Please check your browser permissions.');
      
      btn.innerHTML = '<i class="fas fa-crosshairs"></i> Get My Location';
      btn.disabled = false;
    }
  );
}

function applyLocationFilter() {
  if (!userLocation) return;
  
  // Apply location filter to the base geojson features
  const filteredByLocation = state.geojsonFeatures.filter(feature => {
    const center = calculatePolygonCenter(feature.geometry.coordinates);
    const distance = calculateDistance(
      userLocation.lat,
      userLocation.lng,
      center.lat,
      center.lng
    );
    
    return distance <= currentDistanceRange;
  });
  
  console.log(`[Location Filter] Filtering ${state.geojsonFeatures.length} features to ${filteredByLocation.length} within ${currentDistanceRange} km`);
  
  // Store the location-filtered features in state for timeline to use
  state.locationFilteredFeatures = filteredByLocation;
  
  // Re-apply timeline filter which will use the location-filtered features
  applyTimelineFilter();
  updateMetrics();
  
  console.log(`[Location Filter] Final filtered points: ${state.filteredPoints.length}`);
}

function removeLocationFilter() {
  // Remove the location filter by clearing the location-filtered features
  state.locationFilteredFeatures = null;
  
  // Re-apply other filters
  applyTimelineFilter();
  updateMetrics();
}

function applyAreaFilter() {
  // Apply area filter to the base features
  const filteredByArea = state.geojsonFeatures.filter(feature => {
    const area = feature.properties.Area || 0;
    
    // If max range is 1000, include all areas >= 1000
    if (currentAreaRange[1] === 1000) {
      return area >= currentAreaRange[0];
    } else {
      return area >= currentAreaRange[0] && area <= currentAreaRange[1];
    }
  });
  
  const rangeText = currentAreaRange[1] === 1000 ? `${currentAreaRange[0]}-1000+` : `${currentAreaRange[0]}-${currentAreaRange[1]}`;
  console.log(`[Area Filter] Filtering ${state.geojsonFeatures.length} features to ${filteredByArea.length} with area ${rangeText} km²`);
  
  // Store the area-filtered features in state
  state.areaFilteredFeatures = filteredByArea;
  
  // Re-apply timeline filter
  applyTimelineFilter();
  updateMetrics();
  
  console.log(`[Area Filter] Final filtered points: ${state.filteredPoints.length}`);
}

function checkForNearbyBlooms() {
  if (!userLocation || !notificationsEnabled) return;
  
  const nearbyBlooms = state.geojsonFeatures.filter(feature => {
    const coords = feature.geometry.coordinates;
    const distance = calculateDistance(
      userLocation.lat,
      userLocation.lng,
      coords[1],
      coords[0]
    );
    
    return distance <= currentDistanceRange;
  });
  
  if (nearbyBlooms.length > 0) {
    new Notification('Blooms Near You!', {
      body: `Found ${nearbyBlooms.length} bloom(s) within ${currentDistanceRange} km of your location`,
      icon: '/images/bloom-icon.png'
    });
  }
}

function resetAllFilters() {
  // Reset location filter
  userLocation = null;
  removeUserLocationMarker();
  state.locationFilteredFeatures = null;
  
  // Hide location info
  document.getElementById('locationInfo').style.display = 'none';
  const btn = document.getElementById('getLocationBtn');
  btn.innerHTML = '<i class="fas fa-crosshairs"></i> Get My Location';
  btn.classList.remove('success');
  btn.disabled = false;
  
  // Reset distance range
  currentDistanceRange = 50;
  document.getElementById('distanceRange').value = 50;
  document.getElementById('distanceValue').textContent = '50 km';
  
  // Reset area range
  currentAreaRange = [0, 1000];
  state.areaFilteredFeatures = null;
  document.getElementById('minAreaRange').value = 0;
  document.getElementById('maxAreaRange').value = 1000;
  document.getElementById('areaRangeValue').textContent = '0 - 1,000+ km²';
  
  // Re-apply timeline filter which will now use all features
  applyTimelineFilter();
  updateMetrics();
}

// Haversine formula to calculate distance between two points
function calculateDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // Radius of the Earth in km
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  
  const a = 
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  const distance = R * c;
  
  return distance;
}

function toRad(degrees) {
  return degrees * (Math.PI / 180);
}

export function showAdvancedFilters() {
  const card = document.getElementById('advanced-filters-card');
  if (card) {
    card.style.display = 'flex';
  }
}

export function hideAdvancedFilters() {
  const card = document.getElementById('advanced-filters-card');
  if (card) {
    card.style.display = 'none';
  }
}
