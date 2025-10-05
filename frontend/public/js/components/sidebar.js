import { state } from '../state.js';
import { COLOR_MODE, DISPLAY_MODE } from '../config.js';
import { switchToHexMode, switchToPointsMode, toggleCloudsVisibility, refreshGlobeColors, applyGlobeStyle, setNightSkyBackground, removeNightSkyBackground } from '../globe.js';
import { applyTimelineFilter } from './timeline.js';

export function initSidebar() {
  const container = document.getElementById('sidebar-container');
  container.innerHTML = `
    <div id="legend">
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
  const pointColorOptions = ['Single Color', 'Color by Family'];
  
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
  // 0: Single Color, 1: Color by Family
  if (index === 0) {
    state.currentColorMode = COLOR_MODE.DEFAULT;
  } else {
    state.currentColorMode = COLOR_MODE.FAMILY;
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

export function updateLegend() {
  const colorsLegend = document.getElementById('colorsLegend');
  
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
