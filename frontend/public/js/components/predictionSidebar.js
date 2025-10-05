import { state } from '../state.js';
import { COLOR_MODE, DISPLAY_MODE } from '../config.js';
import { switchToHexMode, switchToPointsMode, toggleCloudsVisibility, refreshGlobeColors, applyGlobeStyle, setNightSkyBackground, removeNightSkyBackground } from '../globe.js';
import { applyTimelineFilter } from './timeline.js';

export function initPredictionSidebar() {
  const container = document.getElementById('prediction-sidebar-container');
  container.innerHTML = `
    <div id="prediction-legend">
      <!-- ML Prediction Controls -->
      <div class="filter-section">
        <h4>🔮 ML Predictions</h4>
        
        <div style="margin-top: 8px;">
          <label style="font-size: 11px;">
            Confidence: 
            <input type="number" id="predConfidenceInput" min="0" max="1" step="0.1" value="0.3" style="width: 50px; padding: 2px;">
          </label>
          <label style="font-size: 11px; margin-left: 8px;">
            Count: 
            <input type="number" id="predNumPredictionsInput" min="50" max="500" step="50" value="150" style="width: 50px; padding: 2px;">
          </label>
        </div>
        
        <label style="font-size: 12px; margin-top: 8px;">
          Location Type:
          <select id="predAoiTypeSelect" style="width: 100%; padding: 4px;">
            <option value="point">Point (Lat/Lon)</option>
            <option value="state" selected>State</option>
            <option value="country">Country</option>
          </select>
        </label>
        
        <!-- Point coordinates -->
        <div id="predPointCoordinates" style="display:none; margin-top: 8px; font-size: 11px;">
          <label>Lat: 
            <input type="number" id="predLatInput" min="-90" max="90" step="0.01" placeholder="20.5" style="width: 60px; padding: 2px;">
          </label>
          <label style="margin-left: 4px;">Lon: 
            <input type="number" id="predLonInput" min="-180" max="180" step="0.01" placeholder="-100" style="width: 60px; padding: 2px;">
          </label>
        </div>
        
        <!-- Country selection -->
        <div id="predCountryOptions" style="display:none; margin-top: 8px;">
          <label style="font-size: 12px;">
            Country:
            <select id="predAoiCountrySelect" style="width: 100%; padding: 4px;">
              <option value="">Select...</option>
              <option value="Mexico">Mexico</option>
              <option value="United States">United States</option>
            </select>
          </label>
        </div>
        
        <!-- State selection -->
        <div id="predStateOptions" style="display:block; margin-top: 8px;">
          <label style="font-size: 12px;">
            State:
            <select id="predAoiStateSelect" style="width: 100%; padding: 4px;">
              <optgroup label="🇲🇽 Mexican States">
                <option value="Queretaro" selected>Querétaro</option>
                <option value="Jalisco">Jalisco</option>
                <option value="Guanajuato">Guanajuato</option>
                <option value="Mexico">México</option>
                <option value="Michoacan">Michoacán</option>
                <option value="Puebla">Puebla</option>
                <option value="Veracruz">Veracruz</option>
                <option value="Oaxaca">Oaxaca</option>
                <option value="Chiapas">Chiapas</option>
                <option value="Yucatan">Yucatán</option>
              </optgroup>
              <optgroup label="🇺🇸 US States">
                <option value="California">California</option>
                <option value="Texas">Texas</option>
                <option value="Florida">Florida</option>
                <option value="New York">New York</option>
              </optgroup>
            </select>
          </label>
        </div>
        
        <label style="font-size: 12px; margin-top: 8px;">
          Date:
          <input type="date" id="predDateInput" value="2025-10-05" style="width: 100%; padding: 4px;">
        </label>
        
        <button id="predFetchDataBtn" style="width: 100%; padding: 8px; margin-top: 8px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 12px;">
          🌸 Predict Blooms
        </button>
      </div>
      
      <!-- Point Style Section -->
      <div class="filter-section">
        <h4>Point Style</h4>
        <div class="switcher-container">
          <button class="switcher-prev" data-target="predPointStyleSwitcher">‹</button>
          <div class="style-switcher" id="predPointStyleSwitcher" data-current="0">
            <span class="switcher-value">Points</span>
          </div>
          <button class="switcher-next" data-target="predPointStyleSwitcher">›</button>
        </div>
      </div>
      
      <!-- Point Color Section -->
      <div class="filter-section">
        <h4>Point Color</h4>
        <div class="switcher-container">
          <button class="switcher-prev" data-target="predPointColorSwitcher">‹</button>
          <div class="style-switcher" id="predPointColorSwitcher" data-current="0">
            <span class="switcher-value">Single Color</span>
          </div>
          <button class="switcher-next" data-target="predPointColorSwitcher">›</button>
        </div>
      </div>
      
      <!-- Globe Style Section -->
      <div class="filter-section">
        <h4>Globe Style</h4>
        <div class="switcher-container">
          <button class="switcher-prev" data-target="predGlobeStyleSwitcher">‹</button>
          <div class="style-switcher" id="predGlobeStyleSwitcher" data-current="0">
            <span class="switcher-value">Normal</span>
          </div>
          <button class="switcher-next" data-target="predGlobeStyleSwitcher">›</button>
        </div>
        <div class="toggle-row">
          <label class="toggle-option">
            <input type="checkbox" id="predToggleNightSky">
            <span>Night Sky</span>
          </label>
          <label class="toggle-option">
            <input type="checkbox" id="predToggleClouds">
            <span>Clouds</span>
          </label>
        </div>
      </div>
      
      <!-- Family Filter -->
      <div class="filter-section">
        <h4>Family</h4>
        <input type="text" id="predFamilySearch" placeholder="Search...">
        <div class="checkbox-group" id="predFamilyCheckboxes"></div>
      </div>
      
      <!-- Genus Filter -->
      <div class="filter-section">
        <h4>Genus</h4>
        <input type="text" id="predSpeciesSearch" placeholder="Search...">
        <div class="checkbox-group" id="predGenusCheckboxes"></div>
      </div>
    </div>
    
    <!-- Colors Legend Column -->
    <div id="predColorsLegend"></div>
  `;

  attachPredictionSidebarEventListeners();
}

function attachPredictionSidebarEventListeners() {
  // AOI Type selection - show/hide relevant options
  document.getElementById('predAoiTypeSelect').addEventListener('change', (e) => {
    const aoiType = e.target.value;
    document.getElementById('predPointCoordinates').style.display = aoiType === 'point' ? 'block' : 'none';
    document.getElementById('predCountryOptions').style.display = aoiType === 'country' ? 'block' : 'none';
    document.getElementById('predStateOptions').style.display = aoiType === 'state' ? 'block' : 'none';
  });
  
  // Fetch Data button
  document.getElementById('predFetchDataBtn').addEventListener('click', fetchPredictionData);
  
  // Point style switcher
  const pointStyleSwitcher = document.getElementById('predPointStyleSwitcher');
  const pointStyleOptions = ['Points', 'Hexagons', 'Points + Rings'];
  
  document.querySelectorAll('[data-target="predPointStyleSwitcher"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = parseInt(pointStyleSwitcher.dataset.current);
      const isPrev = btn.classList.contains('switcher-prev');
      const next = isPrev 
        ? (current - 1 + pointStyleOptions.length) % pointStyleOptions.length
        : (current + 1) % pointStyleOptions.length;
      pointStyleSwitcher.dataset.current = next;
      pointStyleSwitcher.querySelector('.switcher-value').textContent = pointStyleOptions[next];
      handlePredPointStyleChange(next);
    });
  });
  
  // Point color switcher
  const pointColorSwitcher = document.getElementById('predPointColorSwitcher');
  const pointColorOptions = ['Single Color', 'By Family'];
  
  document.querySelectorAll('[data-target="predPointColorSwitcher"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = parseInt(pointColorSwitcher.dataset.current);
      const isPrev = btn.classList.contains('switcher-prev');
      const next = isPrev 
        ? (current - 1 + pointColorOptions.length) % pointColorOptions.length
        : (current + 1) % pointColorOptions.length;
      pointColorSwitcher.dataset.current = next;
      pointColorSwitcher.querySelector('.switcher-value').textContent = pointColorOptions[next];
      handlePredPointColorChange(next);
    });
  });
  
  // Globe style switcher
  const globeStyleSwitcher = document.getElementById('predGlobeStyleSwitcher');
  const globeStyleOptions = ['Normal', 'Detailed', 'Plain'];
  
  document.querySelectorAll('[data-target="predGlobeStyleSwitcher"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = parseInt(globeStyleSwitcher.dataset.current);
      const isPrev = btn.classList.contains('switcher-prev');
      const next = isPrev 
        ? (current - 1 + globeStyleOptions.length) % globeStyleOptions.length
        : (current + 1) % globeStyleOptions.length;
      globeStyleSwitcher.dataset.current = next;
      globeStyleSwitcher.querySelector('.switcher-value').textContent = globeStyleOptions[next];
      handlePredGlobeStyleChange(next);
    });
  });
  
  // Globe toggle checkboxes
  document.getElementById('predToggleNightSky').addEventListener('change', handlePredToggleNightSky);
  document.getElementById('predToggleClouds').addEventListener('change', handlePredToggleClouds);
  
  // Search inputs
  document.getElementById('predFamilySearch').addEventListener('input', (e) => {
    updatePredFamilyCheckboxes(e.target.value);
  });
  
  document.getElementById('predSpeciesSearch').addEventListener('input', (e) => {
    updatePredGenusCheckboxes(e.target.value);
  });
}

function handlePredPointStyleChange(index) {
  // 0: Points, 1: Hexagons, 2: Points + Rings
  if (index === 0) {
    state.currentDisplayMode = DISPLAY_MODE.POINTS;
    state.ringsEnabled = false;
    switchToPointsMode();
  } else if (index === 1) {
    state.currentDisplayMode = DISPLAY_MODE.HEX;
    state.ringsEnabled = false;
    switchToHexMode();
  } else if (index === 2) {
    state.currentDisplayMode = DISPLAY_MODE.POINTS;
    state.ringsEnabled = true;
    switchToPointsMode();
  }
  
  refreshGlobeColors();
}

function handlePredPointColorChange(index) {
  // 0: Single Color, 1: Color by Family
  if (index === 0) {
    state.currentColorMode = COLOR_MODE.DEFAULT;
  } else {
    state.currentColorMode = COLOR_MODE.FAMILY;
  }
  
  updatePredLegend();
  refreshGlobeColors();
}

function handlePredGlobeStyleChange(index) {
  // 0: Normal, 1: Detailed, 2: Plain
  const styles = ['normal', 'detailed', 'plain'];
  applyGlobeStyle(styles[index]);
}

function handlePredToggleNightSky(e) {
  const isChecked = e.target.checked;
  if (isChecked) {
    setNightSkyBackground();
  } else {
    removeNightSkyBackground();
  }
}

function handlePredToggleClouds(e) {
  toggleCloudsVisibility();
}

async function fetchPredictionData() {
  const aoiType = document.getElementById('predAoiTypeSelect').value;
  const date = document.getElementById('predDateInput').value;
  const confidence = document.getElementById('predConfidenceInput').value;
  const numPredictions = document.getElementById('predNumPredictionsInput').value;
  const btn = document.getElementById('predFetchDataBtn');

  // Build params based on AOI type
  const params = {
    aoi_type: aoiType,
    date: date,
    method: 'v2',
    confidence: confidence,
    num_predictions: numPredictions
  };
  
  if (aoiType === 'point') {
    const lat = document.getElementById('predLatInput').value;
    const lon = document.getElementById('predLonInput').value;
    if (!lat || !lon) {
      alert('Please enter latitude and longitude');
      return;
    }
    params.lat = lat;
    params.lon = lon;
  } else if (aoiType === 'state') {
    const aoiState = document.getElementById('predAoiStateSelect').value;
    if (!aoiState) {
      alert('Please select a state');
      return;
    }
    params.aoi_country = 'Mexico';
    params.aoi_state = aoiState;
  } else if (aoiType === 'country') {
    const aoiCountry = document.getElementById('predAoiCountrySelect').value;
    if (!aoiCountry) {
      alert('Please select a country');
      return;
    }
    params.aoi_country = aoiCountry;
  }
  
  const qs = new URLSearchParams(params).toString();
  const API_BASE_URL = 'http://localhost:5001/api';
  const apiUrl = `${API_BASE_URL}/predict/blooms?${qs}`;
  
  console.log('Fetching predictions from:', apiUrl);

  // Show loading
  const originalText = btn.textContent;
  btn.textContent = '⏳ Predicting...';
  btn.disabled = true;

  try {
    const response = await fetch(apiUrl);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    console.log('API Response:', data);
    
    const allFeatures = (data.features || []).filter(feature => {
      const hasProps = feature.properties && feature.properties.Family && feature.properties.Genus && feature.properties.Season;
      return hasProps;
    });

    if (allFeatures.length === 0) {
      alert('No bloom predictions found. Try a different location or date.');
      return;
    }

    // Import needed functions dynamically
    const { createPointFromFeature, switchToPointsMode } = await import('../globe.js');
    const { initializeFamilyColors } = await import('../state.js');
    const { buildTimelineSteps } = await import('./timeline.js');
    
    state.geojsonFeatures = allFeatures;
    state.pointsData = allFeatures.map(createPointFromFeature);
    state.allFamilies = new Set(allFeatures.map(f => f.properties.Family));
    state.allGenus = new Set(allFeatures.map(f => f.properties.Genus));
    state.selectedFamilies = new Set(state.allFamilies);
    state.selectedGenus = new Set(state.allGenus);
    
    // Build genus to family mapping
    state.genusToFamily.clear();
    allFeatures.forEach(f => {
      state.genusToFamily.set(f.properties.Genus, f.properties.Family);
    });
    
    // Initialize colors
    initializeFamilyColors();
    
    // Rebuild timeline (this triggers globe update)
    buildTimelineSteps();
    
    // Update UI
    createPredFilterUI();
    updatePredLegend();
    
    // Switch to points mode to display the data
    switchToPointsMode();
    
    console.log('Globe updated with', state.pointsData.length, 'predictions');
    alert(`✅ Loaded ${allFeatures.length} bloom predictions!`);
  } catch (err) {
    console.error('Fetch failed:', err);
    alert('Error loading predictions: ' + err.message);
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

export function updatePredLegend() {
  const colorsLegend = document.getElementById('predColorsLegend');
  
  if (state.currentColorMode === COLOR_MODE.FAMILY) {
    const familyCounts = {};
    state.geojsonFeatures.forEach(f => {
      const family = f.properties.Family;
      familyCounts[family] = (familyCounts[family] || 0) + 1;
    });
    
    const sortedFamilies = Object.entries(familyCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([family]) => family);
    
    let html = '<h4>Colors</h4><div class="colors-list">';
    sortedFamilies.forEach(family => {
      html += `
        <div class="color-item">
          <div class="color-box" style="background-color: ${state.familyColors[family]};"></div>
          <span>${family}</span>
        </div>
      `;
    });
    html += '</div>';
    
    colorsLegend.innerHTML = html;
    colorsLegend.style.display = 'block';
  } else {
    colorsLegend.innerHTML = '';
    colorsLegend.style.display = 'none';
  }
}

export function createPredFilterUI() {
  updatePredFamilyCheckboxes();
  updatePredGenusCheckboxes();
}

function updatePredFamilyCheckboxes(searchTerm = '') {
  const container = document.getElementById('predFamilyCheckboxes');
  const families = Array.from(state.allFamilies).sort();
  const filteredFamilies = searchTerm 
    ? families.filter(f => f.toLowerCase().includes(searchTerm.toLowerCase()))
    : families;
  
  const shouldBeChecked = (family) => {
    if (state.selectedFamilies.size === 0) {
      return true;
    }
    return state.selectedFamilies.has(family);
  };
  
  container.innerHTML = filteredFamilies.map(family => `
    <label class="checkbox-item">
      <input type="checkbox" value="${family}" 
             ${shouldBeChecked(family) ? 'checked' : ''}>
      ${family}
    </label>
  `).join('');
  
  container.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
    checkbox.addEventListener('change', handlePredFamilyFilterChange);
  });
}

function updatePredGenusCheckboxes(searchTerm = '') {
  const container = document.getElementById('predGenusCheckboxes');
  const genuses = Array.from(state.allGenus).sort();
  const filteredGenuses = searchTerm 
    ? genuses.filter(g => g.toLowerCase().includes(searchTerm.toLowerCase()))
    : genuses;
  
  const shouldBeChecked = (genus) => {
    if (state.selectedGenus.size === 0) {
      return true;
    }
    return state.selectedGenus.has(genus);
  };
  
  container.innerHTML = filteredGenuses.map(genus => `
    <label class="checkbox-item">
      <input type="checkbox" value="${genus}" 
             ${shouldBeChecked(genus) ? 'checked' : ''}>
      ${genus}
    </label>
  `).join('');
  
  container.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
    checkbox.addEventListener('change', handlePredGenusFilterChange);
  });
}

function handlePredFamilyFilterChange(event) {
  const changedFamily = event.target.value;
  const isChecked = event.target.checked;
  const container = document.getElementById('predFamilyCheckboxes');
  const allCheckboxes = container.querySelectorAll('input[type="checkbox"]');
  const checkedCheckboxes = Array.from(allCheckboxes).filter(cb => cb.checked);
  
  state.selectedFamilies.clear();
  checkedCheckboxes.forEach(cb => {
    state.selectedFamilies.add(cb.value);
  });
  
  if (checkedCheckboxes.length === allCheckboxes.length) {
    state.selectedFamilies.clear();
  }
  
  if (!isChecked) {
    const genusInFamily = Array.from(state.genusToFamily.entries())
      .filter(([genus, family]) => family === changedFamily)
      .map(([genus, family]) => genus);
    
    genusInFamily.forEach(genus => {
      state.selectedGenus.delete(genus);
    });
    
    const genusContainer = document.getElementById('predGenusCheckboxes');
    genusContainer.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
      if (genusInFamily.includes(checkbox.value)) {
        checkbox.checked = false;
      }
    });
  }
  
  applyTimelineFilter();
}

function handlePredGenusFilterChange(event) {
  const genus = event.target.value;
  const container = document.getElementById('predGenusCheckboxes');
  const allCheckboxes = container.querySelectorAll('input[type="checkbox"]');
  const checkedCheckboxes = Array.from(allCheckboxes).filter(cb => cb.checked);
  
  state.selectedGenus.clear();
  checkedCheckboxes.forEach(cb => {
    state.selectedGenus.add(cb.value);
  });
  
  if (checkedCheckboxes.length === allCheckboxes.length) {
    state.selectedGenus.clear();
  }
  
  applyTimelineFilter();
}
