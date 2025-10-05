import { COLOR_MODE, DISPLAY_MODE } from './config.js';

// Color palette for family colors
const BLOOM_COLOR_PALETTE = [
  '#E43A7B', // 🌸 Cerise Bloom
  '#FF5C5C', // 🌺 Flame Rose
  '#FF8A3C', // 🍊 Tangerine Glow
  '#F5C400', // 🌼 Golden Pollen
  '#DFA52B', // 🍯 Honey Amber
  '#9C2B2B', // 🌹 Crimson Dust
  '#A94ADB', // 💜 Orchid Beam
  '#4C6DDA', // 🌌 Iris Blue
  '#C457B5', // 🌺 Mauve Flame
  '#F3E2D9'  // 🌤 Pearl White
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
