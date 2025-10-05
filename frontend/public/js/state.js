import { COLOR_MODE, DISPLAY_MODE } from './config.js';

// Color palette for family colors
const BLOOM_COLOR_PALETTE = [
  '#dd001dff', // 🌺 Crimson Flame
  '#ff469fff', // 🌸 Vivid Magenta
  '#ffc20cff', // 🌼 Solar Marigold
  '#f87b21ff', // 🍊 Tangerine Bloom
  '#9cc300ff', // 💙 Azure Petal
  '#a25cfeff', // 💜 Royal Violet
  '#0e973eff', // 🌿 Spring Leaf
  '#ff8eb9ff', // 🌹 Wine Dahlia
  '#fdfdfdff'  // 🌤 Alpine White
];

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
  // Use the color palette, cycling through if there are more families than colors
  return BLOOM_COLOR_PALETTE[index % BLOOM_COLOR_PALETTE.length];
}

export function initializeFamilyColors() {
  const allFamiliesArray = Array.from(state.allFamilies).sort();
  allFamiliesArray.forEach((family, index) => {
    state.familyColors[family] = generateColorForFamily(family, index, allFamiliesArray.length);
  });
}
