import { state } from '../state.js';
import { DISPLAY_MODE } from '../config.js';
import { switchToHexMode, switchToPointsMode } from '../globe.js';
import { updateMetrics } from './metricsCard.js';

export function initTimeline() {
  const container = document.getElementById('timeline-container');
  container.innerHTML = `
    <div id="timeline">
      <div id="timelineLabel">
        <span id="currentSeason">Spring</span>
        <span id="currentYear">2000</span>
      </div>
      <div id="timelineControls">
        <button class="timeline-btn step-btn" id="prevStepBtn" title="Previous Season">
          <i class="fa-solid fa-chevron-left"></i>
        </button>
        <button class="timeline-btn play-btn" id="playBackwardBtn" title="Play Backward">
          <i class="fa-solid fa-backward"></i>
        </button>
        <button class="timeline-btn pause-btn" id="pauseBtn" title="Pause">
          <i class="fa-solid fa-pause"></i>
        </button>
        <button class="timeline-btn play-btn" id="playForwardBtn" title="Play Forward">
          <i class="fa-solid fa-forward"></i>
        </button>
        <button class="timeline-btn step-btn" id="nextStepBtn" title="Next Season">
          <i class="fa-solid fa-chevron-right"></i>
        </button>
      </div>
    </div>
  `;

  attachTimelineEventListeners();
}

function attachTimelineEventListeners() {
  document.getElementById('prevStepBtn').addEventListener('click', goToPreviousStep);
  document.getElementById('nextStepBtn').addEventListener('click', goToNextStep);
  document.getElementById('playForwardBtn').addEventListener('click', () => togglePlay('forward'));
  document.getElementById('playBackwardBtn').addEventListener('click', () => togglePlay('backward'));
  document.getElementById('pauseBtn').addEventListener('click', pause);
}

export function buildTimelineSteps() {
  const yearsSet = new Set(state.geojsonFeatures.map(f => f.properties.year));
  state.allYears = Array.from(yearsSet).sort((a, b) => a - b);
  
  const seasons = ['Spring', 'Summer', 'Fall', 'Winter'];
  state.timelineSteps = [];
  
  state.allYears.forEach(year => {
    seasons.forEach(season => {
      state.timelineSteps.push({ year, season });
    });
  });
  
  const spring2000Index = state.timelineSteps.findIndex(
    step => step.year === 2000 && step.season === 'Spring'
  );
  
  state.currentTimelineIndex = spring2000Index !== -1 ? spring2000Index : 0;
  
  updateTimelineDisplay();
}

function updateTimelineDisplay() {
  if (state.timelineSteps.length === 0) return;
  
  const currentStep = state.timelineSteps[state.currentTimelineIndex];
  document.getElementById('currentSeason').textContent = currentStep.season;
  document.getElementById('currentYear').textContent = currentStep.year;
  
  // Update step button states
  document.getElementById('prevStepBtn').disabled = state.currentTimelineIndex === 0;
  document.getElementById('nextStepBtn').disabled = state.currentTimelineIndex === state.timelineSteps.length - 1;
  
  // Update play button states based on position and play state
  const playForwardBtn = document.getElementById('playForwardBtn');
  const playBackwardBtn = document.getElementById('playBackwardBtn');
  
  // Disable forward if at end
  playForwardBtn.disabled = state.currentTimelineIndex === state.timelineSteps.length - 1;
  // Disable backward if at start
  playBackwardBtn.disabled = state.currentTimelineIndex === 0;
  
  // Visual feedback for active play direction
  if (state.isPlaying) {
    if (state.playDirection === 'forward') {
      playForwardBtn.classList.add('active');
      playBackwardBtn.classList.remove('active');
    } else {
      playBackwardBtn.classList.add('active');
      playForwardBtn.classList.remove('active');
    }
  } else {
    playForwardBtn.classList.remove('active');
    playBackwardBtn.classList.remove('active');
  }
  
  applyTimelineFilter();
}

export function applyTimelineFilter() {
  if (state.timelineSteps.length === 0) return;
  
  const currentStep = state.timelineSteps[state.currentTimelineIndex];
  
  // Start with all features
  let baseFeatures = state.geojsonFeatures;
  
  // Apply location filter if active
  if (state.locationFilteredFeatures !== null && state.locationFilteredFeatures.length >= 0) {
    baseFeatures = state.locationFilteredFeatures;
    console.log(`[Timeline] Using location-filtered features: ${baseFeatures.length}`);
  }
  
  // Apply area filter if active (intersect with location filter if both exist)
  if (state.areaFilteredFeatures !== null && state.areaFilteredFeatures.length >= 0) {
    if (state.locationFilteredFeatures !== null && state.locationFilteredFeatures.length >= 0) {
      // Both filters active - intersect them
      // Create unique IDs for features to match
      const locationIds = new Set(state.locationFilteredFeatures.map(f => 
        `${f.properties.Genus}_${f.properties.Family}_${f.properties.year}_${f.properties.Season}_${f.properties.Area}`
      ));
      baseFeatures = state.areaFilteredFeatures.filter(f => 
        locationIds.has(`${f.properties.Genus}_${f.properties.Family}_${f.properties.year}_${f.properties.Season}_${f.properties.Area}`)
      );
      console.log(`[Timeline] Intersecting location and area filters: ${baseFeatures.length}`);
    } else {
      baseFeatures = state.areaFilteredFeatures;
      console.log(`[Timeline] Using area-filtered features: ${baseFeatures.length}`);
    }
  }
  
  // Now apply timeline, family, and genus filters on top
  state.filteredFeatures = baseFeatures.filter(feature => {
    const matchesTimeline = feature.properties.year === currentStep.year && 
                            feature.properties.Season === currentStep.season;
    
    const matchesFamily = state.selectedFamilies.size === 0 || 
                         state.selectedFamilies.has(feature.properties.Family);
    const matchesGenus = state.selectedGenus.size === 0 || 
                        state.selectedGenus.has(feature.properties.Genus);
    
    return matchesTimeline && matchesFamily && matchesGenus;
  });
  
  console.log(`[Timeline] After timeline/family/genus filters: ${state.filteredFeatures.length} features`);
  
  // Create a set of IDs from filtered features to filter points
  const featureIds = new Set(state.filteredFeatures.map(f => 
    `${f.properties.Genus}_${f.properties.Family}_${f.properties.year}_${f.properties.Season}_${f.properties.Area}`
  ));
  
  // Filter points to match the filtered features, but always include user location
  state.filteredPoints = state.pointsData.filter(point => 
    point.isUserLocation || featureIds.has(`${point.Genus}_${point.Family}_${point.year}_${point.Season}_${point.Area}`)
  );
  
  console.log(`[Timeline] Final filtered points: ${state.filteredPoints.length}`);
  
  if (state.currentDisplayMode === DISPLAY_MODE.HEX) {
    switchToHexMode();
  } else {
    switchToPointsMode();
  }
  
  // Update metrics after filter changes
  updateMetrics();
}

function goToPreviousStep() {
  if (state.currentTimelineIndex > 0) {
    state.currentTimelineIndex--;
    updateTimelineDisplay();
  }
}

function goToNextStep() {
  if (state.currentTimelineIndex < state.timelineSteps.length - 1) {
    state.currentTimelineIndex++;
    updateTimelineDisplay();
  } else if (state.isPlaying && state.playDirection === 'forward') {
    // Stop playing if at end
    pause();
  }
}

function togglePlay(direction) {
  // If already playing in this direction, do nothing (pause button handles stopping)
  // If playing in opposite direction, switch direction
  // If paused, start playing in this direction
  
  if (state.isPlaying && state.playDirection === direction) {
    // Already playing in this direction, ignore (use pause button)
    return;
  }
  
  // Stop current playback if any
  if (state.playInterval) {
    clearInterval(state.playInterval);
    state.playInterval = null;
  }
  
  state.isPlaying = true;
  state.playDirection = direction;
  
  state.playInterval = setInterval(() => {
    if (direction === 'forward') {
      if (state.currentTimelineIndex < state.timelineSteps.length - 1) {
        goToNextStep();
      } else {
        pause();
      }
    } else {
      if (state.currentTimelineIndex > 0) {
        goToPreviousStep();
      } else {
        pause();
      }
    }
  }, 500);
  
  updateTimelineDisplay();
}

function pause() {
  state.isPlaying = false;
  state.playDirection = null;
  
  if (state.playInterval) {
    clearInterval(state.playInterval);
    state.playInterval = null;
  }
  
  updateTimelineDisplay();
}
