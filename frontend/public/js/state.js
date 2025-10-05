import { COLOR_MODE, DISPLAY_MODE } from './config.js';

// Application state
export const state = {
  geojsonFeatures: [],
  pointsData: [],
  currentColorMode: COLOR_MODE.DEFAULT,
  currentDisplayMode: DISPLAY_MODE.POINTS,
  cloudsMesh: null,
  ringsEnabled: false,
  allFamilies: new Set(),
  allGenus: new Set(),
  genusToFamily: new Map(),
  selectedFamilies: new Set(),
  selectedGenus: new Set(),
  allYears: [],
  timelineSteps: [],
  currentTimelineIndex: 0,
  isPlaying: false,
  playDirection: null, // 'forward' or 'backward'
  playInterval: null,
  filteredFeatures: [],
  filteredPoints: [],
  familyColors: {}
};

// Color generation functions
export function generateColorForFamily(family, index, total) {
  const hue = (index * 360 / total) % 360;
  const saturation = 70 + (index % 3) * 10;
  const lightness = 50 + (index % 2) * 10;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

export function initializeFamilyColors() {
  const allFamiliesArray = Array.from(state.allFamilies).sort();
  allFamiliesArray.forEach((family, index) => {
    state.familyColors[family] = generateColorForFamily(family, index, allFamiliesArray.length);
  });
}
