import { COLOR_MODE, DISPLAY_MODE, CONFIG } from './config.js';

// Tailwind-inspired color scales for each family color
// Each family gets a main color, and genera get different shades
const COLOR_SCALES = {
  red: [
    '#991b1b', // 900
    '#b91c1c', // 800
    '#dc2626', // 700
    '#ef4444', // 600
    '#f87171', // 500 - main
    '#fca5a5', // 400
    '#fecaca', // 300
    '#fee2e2', // 200
    '#fef2f2', // 100
  ],
  yellow: [
    '#854d0e', // 900
    '#a16207', // 800
    '#ca8a04', // 700
    '#eab308', // 600
    '#facc15', // 500 - main
    '#fde047', // 400
    '#fef08a', // 300
    '#fef9c3', // 200
    '#fefce8', // 100
  ],
  orange: [
    '#9a3412', // 900
    '#c2410c', // 800
    '#ea580c', // 700
    '#f97316', // 600
    '#fb923c', // 500 - main
    '#fdba74', // 400
    '#fed7aa', // 300
    '#ffedd5', // 200
    '#fff7ed', // 100
  ],
  green: [
    '#14532d', // 900
    '#166534', // 800
    '#15803d', // 700
    '#16a34a', // 600
    '#22c55e', // 500 - main
    '#4ade80', // 400
    '#86efac', // 300
    '#bbf7d0', // 200
    '#dcfce7', // 100
  ],
  purple: [
    '#581c87', // 900
    '#6b21a8', // 800
    '#7e22ce', // 700
    '#9333ea', // 600
    '#a855f7', // 500 - main
    '#c084fc', // 400
    '#d8b4fe', // 300
    '#e9d5ff', // 200
    '#f3e8ff', // 100
  ],
  pink: [
    '#831843', // 900
    '#9f1239', // 800
    '#be185d', // 700
    '#db2777', // 600
    '#ec4899', // 500 - main
    '#f472b6', // 400
    '#f9a8d4', // 300
    '#fbcfe8', // 200
    '#fce7f3', // 100
  ],
    blue: [
    '#1e3a8a', // 900 - darkest
    '#1e40af', // 800
    '#1d4ed8', // 700
    '#2563eb', // 600
    '#3b82f6', // 500 - main
    '#60a5fa', // 400
    '#93c5fd', // 300
    '#bfdbfe', // 200
    '#dbeafe', // 100 - lightest
  ],
  white: [
    '#f3f4f6', // 100
    '#e5e7eb', // 200
    '#d1d5db', // 300
    '#9ca3af', // 400
    '#6b7280', // 500
    '#4b5563', // 600
    '#374151', // 700
    '#1f2937', // 800
    '#111827', // 900 - gray scale (dark to light)
  ],
};

// Main family colors (using the 500 shade - middle of the scale)
const FAMILY_BASE_COLORS = ['blue', 'red', 'yellow', 'orange', 'green', 'purple', 'pink', 'white'];

// Custom color overrides for specific families
// This allows manual assignment of colors to families, overriding the alphabetical assignment
const FAMILY_COLOR_OVERRIDES = {
  'rosaceae': 'green',
  'asteraceae': 'pink',
  'fabaceae': 'blue',
  'ranunculaceae': 'white',
  // Add more custom mappings here as needed, e.g.:
  // 'rosaceae': 'red',
  // 'fabaceae': 'green',
};

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
  familyColors: {},
  genusColors: {},
  familyToColorScale: {} // Maps family name to color scale name
};

// Color generation functions
export function generateColorForFamily(family, index, total) {
  // Check if there's a custom color override for this family
  const lowerFamily = family.toLowerCase();
  const colorScaleName = FAMILY_COLOR_OVERRIDES[lowerFamily] || FAMILY_BASE_COLORS[index % FAMILY_BASE_COLORS.length];
  
  state.familyToColorScale[family] = colorScaleName;
  
  // Return the 800 shade (index 1) for the family
  return COLOR_SCALES[colorScaleName][1];
}

export function generateColorForGenus(genus, family) {
  // Get the color scale for this family
  const colorScaleName = state.familyToColorScale[family];
  if (!colorScaleName) {
    console.warn(`No color scale found for family: ${family}`);
    return CONFIG.DEFAULT_COLOR;
  }
  
  const colorScale = COLOR_SCALES[colorScaleName];
  
  // Get all genera for this family, sorted alphabetically for consistency
  const generaInFamily = Array.from(state.genusToFamily.entries())
    .filter(([g, f]) => f === family)
    .map(([g]) => g)
    .sort();
  
  // Find the index of this genus within the family
  const genusIndex = generaInFamily.indexOf(genus);
  
  if (genusIndex === -1) {
    console.warn(`Genus ${genus} not found in family ${family}`);
    return colorScale[4]; // Return main family color as fallback
  }
  
  // Distribute genera across the color scale
  // Use all shades except index 4 (which is the main family color)
  // Order: darkest to lightest, skipping the middle
  const shadeIndices = [0, 1, 2, 3, 5, 6, 7, 8];
  
  // If we have more genera than shades, cycle through them
  const shadeIndex = shadeIndices[genusIndex % shadeIndices.length];
  
  return colorScale[shadeIndex];
}

export function initializeFamilyColors() {
  const allFamiliesArray = Array.from(state.allFamilies).sort();
  
  console.log('=== INITIALIZING FAMILY COLORS ===');
  console.log('Total families:', allFamiliesArray.length);
  console.log('Families:', allFamiliesArray);
  
  // Assign colors to families
  allFamiliesArray.forEach((family, index) => {
    state.familyColors[family] = generateColorForFamily(family, index, allFamiliesArray.length);
  });
  
  console.log('Family colors assigned:', state.familyColors);
  console.log('Family to color scale mapping:', state.familyToColorScale);
  
  // Assign colors to all genera
  const allGeneraArray = Array.from(state.allGenus).sort();
  console.log('\n=== INITIALIZING GENUS COLORS ===');
  console.log('Total genera:', allGeneraArray.length);
  
  // Group genera by family for better debugging
  const generaByFamily = {};
  allGeneraArray.forEach((genus) => {
    const family = state.genusToFamily.get(genus);
    if (family) {
      if (!generaByFamily[family]) {
        generaByFamily[family] = [];
      }
      generaByFamily[family].push(genus);
      state.genusColors[genus] = generateColorForGenus(genus, family);
    } else {
      console.warn(`No family found for genus: ${genus}`);
    }
  });
  
  console.log('\n=== GENERA BY FAMILY ===');
  Object.entries(generaByFamily).forEach(([family, genera]) => {
    console.log(`\n${family} (${state.familyToColorScale[family]}):`);
    genera.forEach((genus, idx) => {
      console.log(`  ${idx + 1}. ${genus}: ${state.genusColors[genus]}`);
    });
  });
  
  console.log('\n=== SUMMARY ===');
  console.log('Total genus colors initialized:', Object.keys(state.genusColors).length);
  console.log('Sample genus colors:', Object.entries(state.genusColors).slice(0, 5));
}

