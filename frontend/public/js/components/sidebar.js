import { state } from '../state.js';
import { COLOR_MODE, DISPLAY_MODE } from '../config.js';
import { switchToHexMode, switchToPointsMode, toggleCloudsVisibility, refreshGlobeColors, applyGlobeStyle, setNightSkyBackground, removeNightSkyBackground, toggleCountryBorders } from '../globe.js';
import { applyTimelineFilter } from './timeline.js';

export function initSidebar() {
  const container = document.getElementById('sidebar-container');
  container.innerHTML = `
    <div id="legend">
      <!-- API Data Controls -->
      <div class="filter-section">
        <h4>📡 Load Bloom Data</h4>
        
        <label style="font-size: 12px; margin-bottom: 4px;">
          Data Source:
          <select id="dataSourceSelect" style="width: 100%; padding: 4px;">
            <option value="prediction" selected>ML Predictions ⭐</option>
            <option value="satellite">Satellite (Slow ⚠️)</option>
          </select>
        </label>
        
        <div id="predictionOptions" style="display:block; margin-top: 8px;">
          <label style="font-size: 11px;">
            Confidence: 
            <input type="number" id="confidenceInput" min="0" max="1" step="0.1" value="0.3" style="width: 50px; padding: 2px;">
          </label>
          <label style="font-size: 11px; margin-left: 8px;">
            Count: 
            <input type="number" id="numPredictionsInput" min="50" max="500" step="50" value="150" style="width: 50px; padding: 2px;">
          </label>
        </div>
        
        <label style="font-size: 12px; margin-top: 8px;">
          Location Type:
          <select id="aoiTypeSelect" style="width: 100%; padding: 4px;">
            <option value="point">Point (Lat/Lon)</option>
            <option value="state" selected>State</option>
            <option value="country">Country</option>
          </select>
        </label>
        
        <!-- Point coordinates -->
        <div id="pointCoordinates" style="display:none; margin-top: 8px; font-size: 11px;">
          <label>Lat: 
            <input type="number" id="latInput" min="-90" max="90" step="0.01" placeholder="20.5" style="width: 60px; padding: 2px;">
          </label>
          <label style="margin-left: 4px;">Lon: 
            <input type="number" id="lonInput" min="-180" max="180" step="0.01" placeholder="-100" style="width: 60px; padding: 2px;">
          </label>
        </div>
        
        <!-- Country selection -->
        <div id="countryOptions" style="display:none; margin-top: 8px;">
          <label style="font-size: 12px;">
            Country:
            <select id="aoiCountrySelect" style="width: 100%; padding: 4px;">
              <option value="">Select...</option>
              <option value="Mexico">Mexico</option>
              <option value="United States">United States</option>
            </select>
          </label>
        </div>
        
        <!-- State selection -->
        <div id="stateOptions" style="display:block; margin-top: 8px;">
          <label style="font-size: 12px;">
            State:
            <select id="aoiStateSelect" style="width: 100%; padding: 4px;">
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
          <input type="date" id="dateInput" value="2025-10-05" style="width: 100%; padding: 4px;">
        </label>
        
        <button id="fetchDataBtn" style="width: 100%; padding: 8px; margin-top: 8px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 12px;">
          🌸 Fetch Data
        </button>
      </div>
      
      <!-- Point Style Section -->
      <div class="filter-section">
        <h4>Point Style</h4>
        <div class="switcher-container">
          <button class="switcher-prev" data-target="pointStyleSwitcher">‹</button>
          <div class="style-switcher" id="pointStyleSwitcher" data-current="0">
            <span class="switcher-value">Points</span>
          </div>
          <button class="switcher-next" data-target="pointStyleSwitcher">›</button>
        </div>
      </div>
      
      <!-- Point Color Section -->
      <div class="filter-section">
        <h4>Point Color</h4>
        <div class="switcher-container">
          <button class="switcher-prev" data-target="pointColorSwitcher">‹</button>
          <div class="style-switcher" id="pointColorSwitcher" data-current="0">
            <span class="switcher-value">Single Color</span>
          </div>
          <button class="switcher-next" data-target="pointColorSwitcher">›</button>
        </div>
      </div>
      
      <!-- Globe Style Section -->
      <div class="filter-section">
        <h4>Globe Style</h4>
        <div class="switcher-container">
          <button class="switcher-prev" data-target="globeStyleSwitcher">‹</button>
          <div class="style-switcher" id="globeStyleSwitcher" data-current="0">
            <span class="switcher-value">Normal</span>
          </div>
          <button class="switcher-next" data-target="globeStyleSwitcher">›</button>
        </div>
        <div class="toggle-row">
          <label class="toggle-option">
            <input type="checkbox" id="toggleNightSky">
            <span>Night Sky</span>
          </label>
          <label class="toggle-option">
            <input type="checkbox" id="toggleClouds">
            <span>Clouds</span>
          </label>
        </div>
      </div>
      
      <!-- Family Filter -->
      <div class="filter-section">
        <h4>Family</h4>
        <input type="text" id="familySearch" placeholder="Search...">
        <div class="checkbox-group" id="familyCheckboxes"></div>
      </div>
      
      <!-- Genus Filter -->
      <div class="filter-section">
        <h4>Genus</h4>
        <input type="text" id="speciesSearch" placeholder="Search...">
        <div class="checkbox-group" id="genusCheckboxes"></div>
      </div>
    </div>
    
    <!-- Colors Legend Column -->
    <div id="colorsLegend"></div>
  `;

  attachSidebarEventListeners();
}

function attachSidebarEventListeners() {
  // AOI Type selection - show/hide relevant options
  document.getElementById('aoiTypeSelect').addEventListener('change', (e) => {
    const aoiType = e.target.value;
    document.getElementById('pointCoordinates').style.display = aoiType === 'point' ? 'block' : 'none';
    document.getElementById('countryOptions').style.display = aoiType === 'country' ? 'block' : 'none';
    document.getElementById('stateOptions').style.display = aoiType === 'state' ? 'block' : 'none';
  });
  
  // Data source selection
  document.getElementById('dataSourceSelect').addEventListener('change', (e) => {
    const predictionOptions = document.getElementById('predictionOptions');
    predictionOptions.style.display = e.target.value === 'prediction' ? 'block' : 'none';
  });
  
  // Fetch Data button
  document.getElementById('fetchDataBtn').addEventListener('click', fetchBloomData);
  
  // Point style switcher
  const pointStyleSwitcher = document.getElementById('pointStyleSwitcher');
  const pointStyleOptions = ['Points', 'Hexagons', 'Points + Rings'];
  
  document.querySelectorAll('[data-target="pointStyleSwitcher"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = parseInt(pointStyleSwitcher.dataset.current);
      const isPrev = btn.classList.contains('switcher-prev');
      const next = isPrev 
        ? (current - 1 + pointStyleOptions.length) % pointStyleOptions.length
        : (current + 1) % pointStyleOptions.length;
      pointStyleSwitcher.dataset.current = next;
      pointStyleSwitcher.querySelector('.switcher-value').textContent = pointStyleOptions[next];
      handlePointStyleChange(next);
    });
  });
  
  // Point color switcher
  const pointColorSwitcher = document.getElementById('pointColorSwitcher');
  const pointColorOptions = ['Single Color', 'By Family', 'By Genus'];
  
  document.querySelectorAll('[data-target="pointColorSwitcher"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = parseInt(pointColorSwitcher.dataset.current);
      const isPrev = btn.classList.contains('switcher-prev');
      const next = isPrev 
        ? (current - 1 + pointColorOptions.length) % pointColorOptions.length
        : (current + 1) % pointColorOptions.length;
      pointColorSwitcher.dataset.current = next;
      pointColorSwitcher.querySelector('.switcher-value').textContent = pointColorOptions[next];
      handlePointColorChange(next);
    });
  });
  
  // Globe style switcher
  const globeStyleSwitcher = document.getElementById('globeStyleSwitcher');
  const globeStyleOptions = ['Normal', 'Detailed', 'Plain'];
  
  document.querySelectorAll('[data-target="globeStyleSwitcher"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = parseInt(globeStyleSwitcher.dataset.current);
      const isPrev = btn.classList.contains('switcher-prev');
      const next = isPrev 
        ? (current - 1 + globeStyleOptions.length) % globeStyleOptions.length
        : (current + 1) % globeStyleOptions.length;
      globeStyleSwitcher.dataset.current = next;
      globeStyleSwitcher.querySelector('.switcher-value').textContent = globeStyleOptions[next];
      handleGlobeStyleChange(next);
    });
  });
  
  // Globe toggle checkboxes
  document.getElementById('toggleNightSky').addEventListener('change', handleToggleNightSky);
  document.getElementById('toggleClouds').addEventListener('change', handleToggleClouds);
  
  // Search inputs
  document.getElementById('familySearch').addEventListener('input', (e) => {
    updateFamilyCheckboxes(e.target.value);
  });
  
  document.getElementById('speciesSearch').addEventListener('input', (e) => {
    updateGenusCheckboxes(e.target.value);
  });
}

function handlePointStyleChange(index) {
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

function handlePointColorChange(index) {
  // 0: Single Color, 1: Color by Family, 2: Color by Genus
  if (index === 0) {
    state.currentColorMode = COLOR_MODE.DEFAULT;
  } else if (index === 1) {
    state.currentColorMode = COLOR_MODE.FAMILY;
  } else if (index === 2) {
    state.currentColorMode = COLOR_MODE.GENUS;
  }
  
  updateLegend();
  refreshGlobeColors();
}

function handleGlobeStyleChange(index) {
  // 0: Normal, 1: Detailed, 2: Plain
  const styles = ['normal', 'detailed', 'plain'];
  applyGlobeStyle(styles[index]);
}

function handleToggleNightSky(e) {
  const isChecked = e.target.checked;
  if (isChecked) {
    setNightSkyBackground();
  } else {
    removeNightSkyBackground();
  }
}

function handleToggleClouds(e) {
  toggleCloudsVisibility();
}

async function fetchBloomData() {
  const aoiType = document.getElementById('aoiTypeSelect').value;
  const date = document.getElementById('dateInput').value;
  const dataSource = document.getElementById('dataSourceSelect').value;
  const confidence = document.getElementById('confidenceInput').value;
  const numPredictions = document.getElementById('numPredictionsInput').value;
  const btn = document.getElementById('fetchDataBtn');

  // Build params based on AOI type
  const params = {
    aoi_type: aoiType,
    date: date
  };
  
  if (aoiType === 'point') {
    const lat = document.getElementById('latInput').value;
    const lon = document.getElementById('lonInput').value;
    if (!lat || !lon) {
      alert('Please enter latitude and longitude');
      return;
    }
    params.lat = lat;
    params.lon = lon;
  } else if (aoiType === 'state') {
    const aoiState = document.getElementById('aoiStateSelect').value;
    if (!aoiState) {
      alert('Please select a state');
      return;
    }
    params.aoi_country = 'Mexico';
    params.aoi_state = aoiState;
  } else if (aoiType === 'country') {
    const aoiCountry = document.getElementById('aoiCountrySelect').value;
    if (!aoiCountry) {
      alert('Please select a country');
      return;
    }
    params.aoi_country = aoiCountry;
  }
  
  // Add prediction-specific parameters
  if (dataSource === 'prediction') {
    params.method = 'v2';
    params.confidence = confidence;
    params.num_predictions = numPredictions;
  }
  
  const qs = new URLSearchParams(params).toString();
  const endpoint = dataSource === 'prediction' ? 'predict/blooms' : 'data/blooms';
  const API_BASE_URL = 'http://localhost:5001/api';
  const apiUrl = `${API_BASE_URL}/${endpoint}?${qs}`;
  
  console.log('Fetching from:', apiUrl);

  // Show loading
  const originalText = btn.textContent;
  btn.textContent = '⏳ Loading...';
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
      alert('No bloom data found. Try a different location or date.');
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
    createFilterUI();
    updateLegend();
    
    // Switch to points mode to display the data
    switchToPointsMode();
    
    console.log('Globe updated with', state.pointsData.length, 'points');
    alert(`✅ Loaded ${allFeatures.length} bloom predictions!`);
  } catch (err) {
    console.error('Fetch failed:', err);
    alert('Error loading data: ' + err.message);
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

export function updateLegend() {
  const colorsLegend = document.getElementById('colorsLegend');
  
  if (state.currentColorMode === COLOR_MODE.FAMILY) {
    // Show family colors only - all genera in a family use the same color
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
  } else if (state.currentColorMode === COLOR_MODE.GENUS) {
    // Show all genus colors grouped by family with different shades
    const familyCounts = {};
    const generaByFamily = {};
    
    state.geojsonFeatures.forEach(f => {
      const family = f.properties.Family;
      const genus = f.properties.Genus;
      
      familyCounts[family] = (familyCounts[family] || 0) + 1;
      
      if (!generaByFamily[family]) {
        generaByFamily[family] = new Set();
      }
      generaByFamily[family].add(genus);
    });
    
    const sortedFamilies = Object.entries(familyCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([family]) => family);
    
    let html = '<h4>Colors</h4><div class="colors-list" style="gap: 1px;">';
    
    sortedFamilies.forEach(family => {
      // Add family header with main color - compact spacing
      html += `
        <div class="color-item" style="font-weight: bold; margin-top: 3px; padding: 2px 3px; background: rgba(78, 217, 217, 0.05); line-height: 1.2;">
          <div class="color-box" style="background-color: ${state.familyColors[family]}; width: 9px; height: 9px; border: 1.5px solid rgba(78, 217, 217, 0.5); margin-right: 4px;"></div>
          <span style="font-size: 9px;">${family}</span>
        </div>
      `;
      
      // Add all genera in this family with their unique shades - very compact
      const genera = Array.from(generaByFamily[family]).sort();
      genera.forEach(genus => {
        html += `
          <div class="color-item" style="padding: 0.5px 3px 0.5px 10px; line-height: 1.2;">
            <div class="color-box" style="background-color: ${state.genusColors[genus]}; width: 6px; height: 6px; margin-right: 3px;"></div>
            <span style="font-size: 9px;">${genus}</span>
          </div>
        `;
      });
    });
    
    html += '</div>';
    
    colorsLegend.innerHTML = html;
    colorsLegend.style.display = 'block';
  } else {
    colorsLegend.innerHTML = '';
    colorsLegend.style.display = 'none';
  }
}

export function createFilterUI() {
  updateFamilyCheckboxes();
  updateGenusCheckboxes();
}

function updateFamilyCheckboxes(searchTerm = '') {
  const container = document.getElementById('familyCheckboxes');
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
    checkbox.addEventListener('change', handleFamilyFilterChange);
  });
}

function updateGenusCheckboxes(searchTerm = '') {
  const container = document.getElementById('genusCheckboxes');
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
    checkbox.addEventListener('change', handleGenusFilterChange);
  });
}

function handleFamilyFilterChange(event) {
  const changedFamily = event.target.value;
  const isChecked = event.target.checked;
  const container = document.getElementById('familyCheckboxes');
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
    
    const genusContainer = document.getElementById('genusCheckboxes');
    genusContainer.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
      if (genusInFamily.includes(checkbox.value)) {
        checkbox.checked = false;
      }
    });
  }
  
  applyTimelineFilter();
}

function handleGenusFilterChange(event) {
  const genus = event.target.value;
  const container = document.getElementById('genusCheckboxes');
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
