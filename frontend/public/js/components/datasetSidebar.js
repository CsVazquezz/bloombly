import { state } from '../state.js';
import { COLOR_MODE, DISPLAY_MODE } from '../config.js';
import { switchToHexMode, switchToPointsMode, toggleCloudsVisibility, refreshGlobeColors, applyGlobeStyle, setNightSkyBackground, removeNightSkyBackground } from '../globe.js';
import { applyTimelineFilter } from './timeline.js';

export function initDatasetSidebar() {
  const container = document.getElementById('dataset-sidebar-container');
  container.innerHTML = `
    <div id="dataset-legend">
      <!-- Dataset Data Controls -->
      <div class="filter-section">
        <h4>Load Dataset</h4>
        
        <label style="font-size: 12px; margin-bottom: 8px;">
          Select Dataset:
          <select id="datasetSelect" style="width: 100%; padding: 4px; margin-top: 4px; background: #121418; color: #FFFFFF; border: 2px solid rgba(78, 217, 217, 0.3); border-radius: 6px; outline: none;">
            <option value="">-- Choose Dataset --</option>
            <option value="blooms">Bloom Observations</option>
            <option value="flowering_sites">Flowering Sites</option>
            <option value="wildflower_aoi">Wildflower AOI</option>
          </select>
        </label>
        
        <button id="loadDatasetBtn" class="load-dataset-btn">
          Load Data
        </button>
      </div>
      
      <!-- Point Style Section -->
      <div class="filter-section">
        <h4>Point Style</h4>
        <div class="switcher-container">
          <button class="switcher-prev" data-target="datasetPointStyleSwitcher">‹</button>
          <div class="style-switcher" id="datasetPointStyleSwitcher" data-current="0">
            <span class="switcher-value">Points</span>
          </div>
          <button class="switcher-next" data-target="datasetPointStyleSwitcher">›</button>
        </div>
      </div>
      
      <!-- Point Color Section -->
      <div class="filter-section">
        <h4>Point Color</h4>
        <div class="switcher-container">
          <button class="switcher-prev" data-target="datasetPointColorSwitcher">‹</button>
          <div class="style-switcher" id="datasetPointColorSwitcher" data-current="0">
            <span class="switcher-value">Single Color</span>
          </div>
          <button class="switcher-next" data-target="datasetPointColorSwitcher">›</button>
        </div>
      </div>
      
      <!-- Globe Style Section -->
      <div class="filter-section">
        <h4>Globe Style</h4>
        <div class="switcher-container">
          <button class="switcher-prev" data-target="datasetGlobeStyleSwitcher">‹</button>
          <div class="style-switcher" id="datasetGlobeStyleSwitcher" data-current="0">
            <span class="switcher-value">Normal</span>
          </div>
          <button class="switcher-next" data-target="datasetGlobeStyleSwitcher">›</button>
        </div>
        <div class="toggle-row">
          <label class="toggle-option">
            <input type="checkbox" id="datasetToggleNightSky">
            <span>Night Sky</span>
          </label>
          <label class="toggle-option">
            <input type="checkbox" id="datasetToggleClouds">
            <span>Clouds</span>
          </label>
        </div>
      </div>
      
      <!-- Family Filter -->
      <div class="filter-section">
        <h4>Family</h4>
        <input type="text" id="datasetFamilySearch" placeholder="Search...">
        <div class="checkbox-group" id="datasetFamilyCheckboxes"></div>
      </div>
      
      <!-- Genus Filter -->
      <div class="filter-section">
        <h4>Genus</h4>
        <input type="text" id="datasetSpeciesSearch" placeholder="Search...">
        <div class="checkbox-group" id="datasetGenusCheckboxes"></div>
      </div>
    </div>
    
    <!-- Colors Legend Column -->
    <div id="datasetColorsLegend"></div>
  `;

  attachDatasetSidebarEventListeners();
}

function attachDatasetSidebarEventListeners() {
  // Load Dataset button
  document.getElementById('loadDatasetBtn').addEventListener('click', loadDatasetData);
  
  // Point style switcher
  const pointStyleSwitcher = document.getElementById('datasetPointStyleSwitcher');
  const pointStyleOptions = ['Points', 'Hexagons', 'Points + Rings'];
  
  document.querySelectorAll('[data-target="datasetPointStyleSwitcher"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = parseInt(pointStyleSwitcher.dataset.current);
      const isPrev = btn.classList.contains('switcher-prev');
      const next = isPrev 
        ? (current - 1 + pointStyleOptions.length) % pointStyleOptions.length
        : (current + 1) % pointStyleOptions.length;
      pointStyleSwitcher.dataset.current = next;
      pointStyleSwitcher.querySelector('.switcher-value').textContent = pointStyleOptions[next];
      handleDatasetPointStyleChange(next);
    });
  });
  
  // Point color switcher
  const pointColorSwitcher = document.getElementById('datasetPointColorSwitcher');
  const pointColorOptions = ['Single Color', 'By Family', 'By Genus'];
  
  document.querySelectorAll('[data-target="datasetPointColorSwitcher"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = parseInt(pointColorSwitcher.dataset.current);
      const isPrev = btn.classList.contains('switcher-prev');
      const next = isPrev 
        ? (current - 1 + pointColorOptions.length) % pointColorOptions.length
        : (current + 1) % pointColorOptions.length;
      pointColorSwitcher.dataset.current = next;
      pointColorSwitcher.querySelector('.switcher-value').textContent = pointColorOptions[next];
      handleDatasetPointColorChange(next);
    });
  });
  
  // Globe style switcher
  const globeStyleSwitcher = document.getElementById('datasetGlobeStyleSwitcher');
  const globeStyleOptions = ['Normal', 'Detailed', 'Plain'];
  
  document.querySelectorAll('[data-target="datasetGlobeStyleSwitcher"]').forEach(btn => {
    btn.addEventListener('click', () => {
      const current = parseInt(globeStyleSwitcher.dataset.current);
      const isPrev = btn.classList.contains('switcher-prev');
      const next = isPrev 
        ? (current - 1 + globeStyleOptions.length) % globeStyleOptions.length
        : (current + 1) % globeStyleOptions.length;
      globeStyleSwitcher.dataset.current = next;
      globeStyleSwitcher.querySelector('.switcher-value').textContent = globeStyleOptions[next];
      handleDatasetGlobeStyleChange(next);
    });
  });
  
  // Globe toggle checkboxes
  document.getElementById('datasetToggleNightSky').addEventListener('change', handleDatasetToggleNightSky);
  document.getElementById('datasetToggleClouds').addEventListener('change', handleDatasetToggleClouds);
  
  // Search inputs
  document.getElementById('datasetFamilySearch').addEventListener('input', (e) => {
    updateDatasetFamilyCheckboxes(e.target.value);
  });
  
  document.getElementById('datasetSpeciesSearch').addEventListener('input', (e) => {
    updateDatasetGenusCheckboxes(e.target.value);
  });
}

function handleDatasetPointStyleChange(index) {
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

function handleDatasetPointColorChange(index) {
  // 0: Single Color, 1: Color by Family, 2: Color by Genus
  if (index === 0) {
    state.currentColorMode = COLOR_MODE.DEFAULT;
  } else if (index === 1) {
    state.currentColorMode = COLOR_MODE.FAMILY;
  } else if (index === 2) {
    state.currentColorMode = COLOR_MODE.GENUS;
  }
  
  updateDatasetLegend();
  refreshGlobeColors();
}

function handleDatasetGlobeStyleChange(index) {
  // 0: Normal, 1: Detailed, 2: Plain
  const styles = ['normal', 'detailed', 'plain'];
  applyGlobeStyle(styles[index]);
}

function handleDatasetToggleNightSky(e) {
  const isChecked = e.target.checked;
  if (isChecked) {
    setNightSkyBackground();
  } else {
    removeNightSkyBackground();
  }
}

function handleDatasetToggleClouds(e) {
  toggleCloudsVisibility();
}

async function loadDatasetData() {
  const datasetName = document.getElementById('datasetSelect').value;
  const btn = document.getElementById('loadDatasetBtn');
  
  if (!datasetName) {
    alert('Please select a dataset');
    return;
  }
  
  // Map dataset names to file paths
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
  
  // Show loading state
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
    createDatasetFilterUI();
    updateDatasetLegend();
    
    // Switch to points mode to display the data
    switchToPointsMode();
    
    // Show success state briefly
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
    
    // Show error state
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

export function updateDatasetLegend() {
  const colorsLegend = document.getElementById('datasetColorsLegend');
  
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

export function createDatasetFilterUI() {
  updateDatasetFamilyCheckboxes();
  updateDatasetGenusCheckboxes();
}

function updateDatasetFamilyCheckboxes(searchTerm = '') {
  const container = document.getElementById('datasetFamilyCheckboxes');
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
    checkbox.addEventListener('change', handleDatasetFamilyFilterChange);
  });
}

function updateDatasetGenusCheckboxes(searchTerm = '') {
  const container = document.getElementById('datasetGenusCheckboxes');
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
    checkbox.addEventListener('change', handleDatasetGenusFilterChange);
  });
}

function handleDatasetFamilyFilterChange(event) {
  const changedFamily = event.target.value;
  const isChecked = event.target.checked;
  const container = document.getElementById('datasetFamilyCheckboxes');
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
    
    const genusContainer = document.getElementById('datasetGenusCheckboxes');
    genusContainer.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
      if (genusInFamily.includes(checkbox.value)) {
        checkbox.checked = false;
      }
    });
  }
  
  applyTimelineFilter();
}

function handleDatasetGenusFilterChange(event) {
  const genus = event.target.value;
  const container = document.getElementById('datasetGenusCheckboxes');
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
