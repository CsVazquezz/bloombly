import { CONFIG } from './config.js';
import { state, initializeFamilyColors } from './state.js';
import { initGlobe, createPointFromFeature, initializeClouds, switchToPointsMode, loadCountryBorders } from './globe.js';
import { initTopSelector } from './components/topSelector.js';
import { initSidebar, createFilterUI, updateLegend } from './components/sidebar.js';
import { initTimeline, buildTimelineSteps } from './components/timeline.js';

// Initialize the application
async function init() {
  // Initialize components
  initTopSelector();
  initSidebar();
  initTimeline();
  
  // Initialize globe
  initGlobe();
  initializeClouds();
  
  // Load country borders
  await loadCountryBorders();
  
  // Load data
  await loadGeoJSONData();
}

async function loadGeoJSONData() {
  try {
    const datas = await Promise.all(
      CONFIG.GEOJSON_FILES.map(file => fetch(file).then(response => response.json()))
    );
    
    const allFeatures = datas.flatMap(data => data.features).filter(feature => 
      feature.properties.Family && feature.properties.Genus && feature.properties.Season
    );
    
    state.geojsonFeatures = allFeatures;
    state.pointsData = allFeatures.map(createPointFromFeature);
    
    // Populate all families and genus
    state.allFamilies = new Set(allFeatures.map(f => f.properties.Family));
    state.allGenus = new Set(allFeatures.map(f => f.properties.Genus));
    
    // Build genus to family mapping
    state.genusToFamily.clear();
    allFeatures.forEach(f => {
      state.genusToFamily.set(f.properties.Genus, f.properties.Family);
    });
    
    // Initialize family colors
    initializeFamilyColors();
    
    // Build timeline
    buildTimelineSteps();
    
    // Create filter UI
    createFilterUI();
    
    updateLegend();
    switchToPointsMode();
  } catch (error) {
    console.error('Error loading GeoJSON data:', error);
    alert('Error loading data: ' + error.message);
  }
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
