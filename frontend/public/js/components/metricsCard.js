import { state } from '../state.js';

let updateInterval = null;

export function initMetricsCard() {
  const container = document.createElement('div');
  container.id = 'metrics-card';
  
  // Build the static HTML structure once
  container.innerHTML = `
    <div class="metrics-content">
      <div class="metric-item">
        <div>
          <div class="metric-label">Bloom Count</div>
          <div class="metric-sublabel">vs year before</div>
        </div>
        <div class="metric-value-container">
          <div class="metric-count" id="bloom-count">0</div>
          <div class="metric-percentage-wrapper">
            <span class="metric-percentage" id="bloom-percentage">0%</span>
          </div>
        </div>
      </div>
      
      <div class="metric-item">
        <div>
          <div class="metric-label">Coverage Area</div>
          <div class="metric-sublabel">vs year before</div>
        </div>
        <div class="metric-value-container">
          <div class="metric-count" id="coverage-count">0 km²</div>
          <div class="metric-percentage-wrapper">
            <span class="metric-percentage" id="coverage-percentage">0%</span>
          </div>
        </div>
      </div>
    </div>
  `;
  
  document.body.appendChild(container);
}

export function showMetricsCard() {
  const card = document.getElementById('metrics-card');
  
  if (!card) {
    console.error('Metrics card not initialized');
    return;
  }
  
  // Show the card
  card.style.display = 'flex';
  
  // Update metrics immediately
  updateMetrics();
  
  // Update metrics every time filters change
  // This will be called from the filter functions
}

export function hideMetricsCard() {
  const card = document.getElementById('metrics-card');
  if (!card) return;
  
  card.style.display = 'none';
}

export function updateMetrics() {
  const card = document.getElementById('metrics-card');
  if (!card || card.style.display === 'none') return;
  
  // Calculate metrics based on currently filtered/visible data
  const metrics = calculateGlobalMetrics();
  
  // Get the elements to update (only the values, not the whole structure)
  const bloomCount = document.getElementById('bloom-count');
  const bloomPercentage = document.getElementById('bloom-percentage');
  const coverageCount = document.getElementById('coverage-count');
  const coveragePercentage = document.getElementById('coverage-percentage');
  
  if (!bloomCount || !bloomPercentage || !coverageCount || !coveragePercentage) return;
  
  // Add update animation class
  bloomCount.classList.add('updating');
  bloomPercentage.classList.add('updating');
  coverageCount.classList.add('updating');
  coveragePercentage.classList.add('updating');
  
  // Update values after brief delay for animation
  setTimeout(() => {
    // Update bloom count
    bloomCount.textContent = metrics.totalBlooms.toLocaleString();
    
    // Update bloom percentage
    if (metrics.yearOverYearChange !== null) {
      const bloomChange = metrics.yearOverYearChange;
      const bloomIcon = bloomChange > 0 ? '↑' : bloomChange < 0 ? '↓' : '=';
      bloomPercentage.innerHTML = `${bloomIcon} ${Math.abs(bloomChange).toFixed(1)}%`;
      bloomPercentage.className = `metric-percentage ${bloomChange > 0 ? 'positive' : bloomChange < 0 ? 'negative' : 'neutral'}`;
    } else {
      bloomPercentage.innerHTML = '—';
      bloomPercentage.className = 'metric-percentage neutral';
    }
    
    // Update coverage count
    coverageCount.textContent = `${metrics.totalArea.toFixed(2)} km²`;
    
    // Update coverage percentage
    if (metrics.areaChange !== null) {
      const areaChange = metrics.areaChange;
      const areaIcon = areaChange > 0 ? '↑' : areaChange < 0 ? '↓' : '=';
      coveragePercentage.innerHTML = `${areaIcon} ${Math.abs(areaChange).toFixed(1)}%`;
      coveragePercentage.className = `metric-percentage ${areaChange > 0 ? 'positive' : areaChange < 0 ? 'negative' : 'neutral'}`;
    } else {
      coveragePercentage.innerHTML = '—';
      coveragePercentage.className = 'metric-percentage neutral';
    }
    
    // Remove animation class
    bloomCount.classList.remove('updating');
    bloomPercentage.classList.remove('updating');
    coverageCount.classList.remove('updating');
    coveragePercentage.classList.remove('updating');
  }, 150);
}

function calculateGlobalMetrics() {
  const metrics = {
    totalBlooms: 0,
    totalArea: 0,
    yearOverYearChange: null,
    areaChange: null
  };
  
  // Get currently filtered features (what's actually visible on the globe)
  const currentFeatures = state.filteredFeatures || state.geojsonFeatures || [];
  
  if (currentFeatures.length === 0) {
    return metrics;
  }
  
  // Calculate total blooms and area for current filtered data
  metrics.totalBlooms = currentFeatures.length;
  metrics.totalArea = currentFeatures.reduce((sum, feature) => {
    return sum + (feature.properties.Area || 0);
  }, 0);
  
  // Find the most recent year and season in the filtered data
  const years = currentFeatures.map(f => f.properties.year).filter(y => y);
  const seasons = currentFeatures.map(f => f.properties.Season).filter(s => s);
  
  if (years.length > 0) {
    const currentYear = Math.max(...years);
    const currentSeason = seasons[0]; // Use the first available season
    
    // Get all features for current year/season from all data (not just filtered)
    const currentYearFeatures = state.geojsonFeatures.filter(f => 
      f.properties.year === currentYear && 
      f.properties.Season === currentSeason
    );
    
    // Get all features for previous year/same season from all data
    const previousYearFeatures = state.geojsonFeatures.filter(f => 
      f.properties.year === currentYear - 1 && 
      f.properties.Season === currentSeason
    );
    
    if (previousYearFeatures.length > 0 && currentYearFeatures.length > 0) {
      // Calculate year-over-year change in number of blooms
      const bloomChange = ((currentYearFeatures.length - previousYearFeatures.length) / previousYearFeatures.length) * 100;
      metrics.yearOverYearChange = bloomChange;
      
      // Calculate area change
      const prevTotalArea = previousYearFeatures.reduce((sum, f) => sum + (f.properties.Area || 0), 0);
      const currTotalArea = currentYearFeatures.reduce((sum, f) => sum + (f.properties.Area || 0), 0);
      
      if (prevTotalArea > 0) {
        metrics.areaChange = ((currTotalArea - prevTotalArea) / prevTotalArea) * 100;
      }
    }
  }
  
  return metrics;
}
