import { state } from '../state.js';
import { COLOR_MODE, DISPLAY_MODE } from '../config.js';
import { switchToHexMode, switchToPointsMode, toggleCloudsVisibility, refreshGlobeColors, applyGlobeStyle, setNightSkyBackground, removeNightSkyBackground } from '../globe.js';
import { applyTimelineFilter } from './timeline.js';

// Sidebar mode state
let currentMode = 'dataset'; // 'dataset' or 'prediction'

export function initUnifiedSidebar() {
  const container = document.getElementById('unified-sidebar-container');
  container.innerHTML = `
    <div id="unified-legend">
      <!-- Scrollable Content Area -->
      <div id="scrollableContent">
        <!-- Mode Switcher -->
        <div class="filter-section">
          <div class="mode-switcher-container">
            <button class="mode-btn active" id="datasetModeBtn" data-mode="dataset">Dataset</button>
            <button class="mode-btn" id="predictionModeBtn" data-mode="prediction">Prediction</button>
          </div>
        </div>
        
        <!-- Dynamic Content Section (changes based on mode) -->
        <div id="dynamicContent"></div>
      </div>
      
      <!-- Fixed Controls at Bottom -->
      <div id="fixedControls">
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
      </div>
    </div>
    
    <!-- Colors Legend Column -->
    <div id="unifiedColorsLegend"></div>
  `;

  attachUnifiedSidebarEventListeners();
  switchMode('dataset'); // Start with dataset mode
}

function attachUnifiedSidebarEventListeners() {
  // Mode switcher buttons
  document.getElementById('datasetModeBtn').addEventListener('click', () => switchMode('dataset'));
  document.getElementById('predictionModeBtn').addEventListener('click', () => switchMode('prediction'));
  
  // Point style switcher
  const pointStyleSwitcher = document.getElementById('pointStyleSwitcher');
  const pointStyleOptions = ['Points', 'Hexagons', 'Rings'];
  
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
}

function switchMode(mode) {
  currentMode = mode;
  
  // Update button states
  const datasetBtn = document.getElementById('datasetModeBtn');
  const predictionBtn = document.getElementById('predictionModeBtn');
  
  if (mode === 'dataset') {
    datasetBtn.classList.add('active');
    predictionBtn.classList.remove('active');
    renderDatasetContent();
  } else {
    predictionBtn.classList.add('active');
    datasetBtn.classList.remove('active');
    renderPredictionContent();
  }
}

function renderDatasetContent() {
  const dynamicContent = document.getElementById('dynamicContent');
  dynamicContent.innerHTML = `
    <!-- Dataset Data Controls -->
    <div class="filter-section">
      <select id="datasetSelect" style="width: 100%; padding: 4px; background: #121418; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 6px; outline: none;">
        <option value="">-- Choose Dataset --</option>
        <option value="blooms">Bloom Observations</option>
        <option value="flowering_sites">Flowering Sites</option>
        <option value="wildflower_aoi">Wildflower AOI</option>
      </select>
      
      <button id="loadDatasetBtn" class="load-dataset-btn">
        Load Data
      </button>
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
  `;
  
  // Attach dataset-specific listeners
  document.getElementById('loadDatasetBtn').addEventListener('click', loadDatasetData);
  document.getElementById('familySearch').addEventListener('input', (e) => {
    updateFamilyCheckboxes(e.target.value);
  });
  document.getElementById('speciesSearch').addEventListener('input', (e) => {
    updateGenusCheckboxes(e.target.value);
  });
}

function renderPredictionContent() {
  const dynamicContent = document.getElementById('dynamicContent');
  dynamicContent.innerHTML = `
    <!-- ML Prediction Controls -->
    <div class="filter-section">
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;">
        <div>
          <label style="font-size: 11px; color: #4ED9D9; font-weight: 200; display: block; margin-bottom: 4px;">
            Confidence
          </label>
          <input type="number" id="predConfidenceInput" min="0" max="1" step="0.1" value="0.3" style="width: 100%; padding: 4px; background: #121418; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 4px; box-sizing: border-box;">
        </div>
        <div>
          <label style="font-size: 11px; color: #4ED9D9; font-weight: 200; display: block; margin-bottom: 4px;">
            Count
          </label>
          <input type="number" id="predNumPredictionsInput" min="50" max="500" step="50" value="150" style="width: 100%; padding: 4px; background: #121418; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 4px; box-sizing: border-box;">
        </div>
      </div>
      
      <label style="font-size: 11px; margin-top: 8px; display: block; color: #4ED9D9; font-weight: 200;">
        Location Type:
        <select id="predAoiTypeSelect" style="width: 100%; padding: 4px; margin-top: 4px; background: #121418; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 6px; outline: none; box-sizing: border-box;">
          <option value="point">Point (Lat/Lon)</option>
          <option value="state" selected>State</option>
          <option value="country">Country</option>
        </select>
      </label>
      
      <!-- Point coordinates -->
      <div id="predPointCoordinates" style="display:none; margin-top: 8px; font-size: 11px;">
        <label>Lat: 
          <input type="number" id="predLatInput" min="-90" max="90" step="0.01" placeholder="20.5" style="width: 60px; padding: 2px; background: #121418; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 4px;">
        </label>
        <label style="margin-left: 4px;">Lon: 
          <input type="number" id="predLonInput" min="-180" max="180" step="0.01" placeholder="-100" style="width: 60px; padding: 2px; background: #121418; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 4px;">
        </label>
      </div>
      
      <!-- Country selection -->
      <div id="predCountryOptions" style="display:none; margin-top: 8px;">
        <label style="font-size: 11px; display: block; color: #4ED9D9; font-weight: 200;">
          Country:
          <select id="predAoiCountrySelect" style="width: 100%; padding: 4px; margin-top: 4px; background: #121418; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 6px; outline: none; box-sizing: border-box;">
            <option value="">Select...</option>
            <option value="Mexico">Mexico</option>
            <option value="United States">United States</option>
          </select>
        </label>
      </div>
      
      <!-- State selection -->
      <div id="predStateOptions" style="display:block; margin-top: 8px;">
        <label style="font-size: 11px; display: block; color: #4ED9D9; font-weight: 200;">
          State:
          <select id="predAoiStateSelect" style="width: 100%; padding: 4px; margin-top: 4px; background: #121418; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 6px; outline: none; box-sizing: border-box;">
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
      
      <label style="font-size: 11px; margin-top: 8px; display: block; color: #4ED9D9; font-weight: 200;">
        Date:
        <input type="date" id="predDateInput" value="2025-10-05" style="width: 100%; padding: 4px; margin-top: 4px; background: #121418; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 6px; outline: none; box-sizing: border-box;">
      </label>
      
      <button id="predFetchDataBtn" class="load-dataset-btn">
        Predict Blooms
      </button>
    </div>
  `;
  
  // Attach prediction-specific listeners
  document.getElementById('predAoiTypeSelect').addEventListener('change', (e) => {
    const aoiType = e.target.value;
    document.getElementById('predPointCoordinates').style.display = aoiType === 'point' ? 'block' : 'none';
    document.getElementById('predCountryOptions').style.display = aoiType === 'country' ? 'block' : 'none';
    document.getElementById('predStateOptions').style.display = aoiType === 'state' ? 'block' : 'none';
  });
  
  document.getElementById('predFetchDataBtn').addEventListener('click', fetchPredictionData);
}

// Common control handlers
function handlePointStyleChange(index) {
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
  const styles = ['normal', 'detailed', 'plain'];
  applyGlobeStyle(styles[index]);
}

function handleToggleNightSky(e) {
  if (e.target.checked) {
    setNightSkyBackground();
  } else {
    removeNightSkyBackground();
  }
}

function handleToggleClouds(e) {
  toggleCloudsVisibility();
}

// Dataset loading function
async function loadDatasetData() {
  const datasetName = document.getElementById('datasetSelect').value;
  const btn = document.getElementById('loadDatasetBtn');
  
  if (!datasetName) {
    alert('Please select a dataset');
    return;
  }
  
  const datasetPaths = {
    'blooms': '../data/processed/blooms.geojson',
    'flowering_sites': '../data/geojson/flowering_sites.geojson',
    'wildflower_aoi': '../data/geojson/WildflowerBlooms_AreaOfInterest.geojson'
  };
  
  const filePath = datasetPaths[datasetName];
  
  if (!filePath) {
    alert('Dataset path not found');
    return;
  }
  
  const originalText = btn.textContent;
  btn.textContent = 'Loading...';
  btn.disabled = true;
  btn.classList.add('loading');
  
  try {
    const response = await fetch(filePath);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('Dataset loaded:', data);
    
    const allFeatures = (data.features || []).filter(feature => {
      const hasProps = feature.properties && feature.properties.Family && feature.properties.Genus && feature.properties.Season;
      return hasProps;
    });

    if (allFeatures.length === 0) {
      throw new Error('No valid features found in dataset');
    }

    const { createPointFromFeature, switchToPointsMode } = await import('../globe.js');
    const { initializeFamilyColors } = await import('../state.js');
    const { buildTimelineSteps } = await import('./timeline.js');
    
    state.geojsonFeatures = allFeatures;
    state.pointsData = allFeatures.map(createPointFromFeature);
    state.allFamilies = new Set(allFeatures.map(f => f.properties.Family));
    state.allGenus = new Set(allFeatures.map(f => f.properties.Genus));
    state.selectedFamilies = new Set(state.allFamilies);
    state.selectedGenus = new Set(state.allGenus);
    
    state.genusToFamily.clear();
    allFeatures.forEach(f => {
      state.genusToFamily.set(f.properties.Genus, f.properties.Family);
    });
    
    initializeFamilyColors();
    buildTimelineSteps();
    createFilterUI();
    updateLegend();
    switchToPointsMode();
    
    btn.textContent = 'Loaded ✓';
    btn.classList.remove('loading');
    btn.classList.add('success');
    
    setTimeout(() => {
      btn.textContent = originalText;
      btn.classList.remove('success');
      btn.disabled = false;
    }, 1500);
    
    console.log('Globe updated with', state.pointsData.length, 'points from dataset');
  } catch (err) {
    console.error('Dataset load failed:', err);
    
    btn.textContent = 'Error ✗';
    btn.classList.remove('loading');
    btn.classList.add('error');
    
    setTimeout(() => {
      btn.textContent = originalText;
      btn.classList.remove('error');
      btn.disabled = false;
    }, 2000);
    
    alert('Error loading dataset: ' + err.message);
  }
}

// Prediction fetching function
async function fetchPredictionData() {
  const aoiType = document.getElementById('predAoiTypeSelect').value;
  const date = document.getElementById('predDateInput').value;
  const confidence = document.getElementById('predConfidenceInput').value;
  const numPredictions = document.getElementById('predNumPredictionsInput').value;
  const btn = document.getElementById('predFetchDataBtn');

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

  const originalText = btn.textContent;
  btn.textContent = 'Loading...';
  btn.disabled = true;
  btn.classList.add('loading');

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

    const { createPointFromFeature, switchToPointsMode } = await import('../globe.js');
    const { initializeFamilyColors } = await import('../state.js');
    const { buildTimelineSteps } = await import('./timeline.js');
    
    state.geojsonFeatures = allFeatures;
    state.pointsData = allFeatures.map(createPointFromFeature);
    state.allFamilies = new Set(allFeatures.map(f => f.properties.Family));
    state.allGenus = new Set(allFeatures.map(f => f.properties.Genus));
    state.selectedFamilies = new Set(state.allFamilies);
    state.selectedGenus = new Set(state.allGenus);
    
    state.genusToFamily.clear();
    allFeatures.forEach(f => {
      state.genusToFamily.set(f.properties.Genus, f.properties.Family);
    });
    
    initializeFamilyColors();
    buildTimelineSteps();
    createFilterUI();
    updateLegend();
    switchToPointsMode();
    
    btn.textContent = 'Loaded ✓';
    btn.classList.remove('loading');
    btn.classList.add('success');
    
    setTimeout(() => {
      btn.textContent = originalText;
      btn.classList.remove('success');
      btn.disabled = false;
    }, 1500);
    
    console.log('Globe updated with', state.pointsData.length, 'predictions');
  } catch (err) {
    console.error('Fetch failed:', err);
    
    btn.textContent = 'Error ✗';
    btn.classList.remove('loading');
    btn.classList.add('error');
    
    setTimeout(() => {
      btn.textContent = originalText;
      btn.classList.remove('error');
      btn.disabled = false;
    }, 2000);
    
    alert('Error loading predictions: ' + err.message);
  }
}

// Filter UI functions
export function createFilterUI() {
  // Check which mode we're in and populate the correct containers
  if (currentMode === 'prediction') {
    updatePredictionFamilyCheckboxes();
    updatePredictionGenusCheckboxes();
  } else {
    updateFamilyCheckboxes();
    updateGenusCheckboxes();
  }
}

function updateFamilyCheckboxes(searchTerm = '') {
  const container = document.getElementById('familyCheckboxes');
  if (!container) return;
  
  const families = Array.from(state.allFamilies).sort();
  const filteredFamilies = searchTerm 
    ? families.filter(f => f.toLowerCase().includes(searchTerm.toLowerCase()))
    : families;
  
  const shouldBeChecked = (family) => {
    if (state.selectedFamilies.size === 0) return true;
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
  if (!container) return;
  
  const genuses = Array.from(state.allGenus).sort();
  const filteredGenuses = searchTerm 
    ? genuses.filter(g => g.toLowerCase().includes(searchTerm.toLowerCase()))
    : genuses;
  
  const shouldBeChecked = (genus) => {
    if (state.selectedGenus.size === 0) return true;
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

// Prediction-specific filter functions
function updatePredictionFamilyCheckboxes(searchTerm = '') {
  const container = document.getElementById('predFamilyCheckboxes');
  if (!container) return;
  
  const families = Array.from(state.allFamilies).sort();
  const filteredFamilies = searchTerm 
    ? families.filter(f => f.toLowerCase().includes(searchTerm.toLowerCase()))
    : families;
  
  const shouldBeChecked = (family) => {
    if (state.selectedFamilies.size === 0) return true;
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
    checkbox.addEventListener('change', handlePredictionFamilyFilterChange);
  });
}

function updatePredictionGenusCheckboxes(searchTerm = '') {
  const container = document.getElementById('predGenusCheckboxes');
  if (!container) return;
  
  const genuses = Array.from(state.allGenus).sort();
  const filteredGenuses = searchTerm 
    ? genuses.filter(g => g.toLowerCase().includes(searchTerm.toLowerCase()))
    : genuses;
  
  const shouldBeChecked = (genus) => {
    if (state.selectedGenus.size === 0) return true;
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
    checkbox.addEventListener('change', handlePredictionGenusFilterChange);
  });
}

function handlePredictionFamilyFilterChange(event) {
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

function handlePredictionGenusFilterChange(event) {
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

// Legend update function
export function updateLegend() {
  const colorsLegend = document.getElementById('unifiedColorsLegend');
  
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
  } else if (state.currentColorMode === COLOR_MODE.GENUS) {
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
      html += `
        <div class="color-item" style="font-weight: bold; margin-top: 3px; padding: 2px 3px; background: rgba(78, 217, 217, 0.05); line-height: 1.2;">
          <div class="color-box" style="background-color: ${state.familyColors[family]}; width: 9px; height: 9px; border: 1.5px solid rgba(78, 217, 217, 0.5); margin-right: 4px;"></div>
          <span style="font-size: 9px;">${family}</span>
        </div>
      `;
      
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
