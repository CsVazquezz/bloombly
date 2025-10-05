import { CONFIG, COLOR_MODE, DISPLAY_MODE } from './config.js';
import { state } from './state.js';

let globe = null;

export function initGlobe() {
  globe = new Globe(document.getElementById('globeViz'), { animateIn: false })
    .globeImageUrl(CONFIG.EARTH_IMAGE_URL)
    .bumpImageUrl(CONFIG.EARTH_TOPOLOGY_URL)
    .backgroundColor('rgba(15,17,21,1)')
    .atmosphereColor('rgba(255, 255, 255, 1)')
    .atmosphereAltitude(0.15)
    .pointOfView(CONFIG.INITIAL_VIEW)
    .hexPolygonsData([])
    .hexPolygonResolution(CONFIG.HEX_POLYGON_RESOLUTION)
    .hexPolygonMargin(CONFIG.HEX_POLYGON_MARGIN)
    .hexPolygonUseDots(true)
    .hexPolygonColor(polygon => getColorForFeature(polygon.properties))
    .hexPolygonLabel(createHexPolygonLabel)
    .pointsData([])
    .pointColor(getColorForFeature)
    .pointRadius(calculatePointRadius)
    .pointAltitude(0)
    .ringsData([])
    .ringColor(getColorForFeature)
    .ringMaxRadius('maxR')
    .ringPropagationSpeed('propagationSpeed')
    .ringRepeatPeriod('repeatPeriod')
    .polygonsData([])
    .polygonCapColor(() => 'rgba(0, 0, 0, 0)')
    .polygonSideColor(() => 'rgba(0, 0, 0, 0)')
    .polygonStrokeColor(() => 'rgba(255, 255, 255, 0.25)')
    .polygonAltitude(0.001);

  globe.controls().autoRotate = false;
  globe.controls().autoRotateSpeed = 0.35;

  return globe;
}

export function getGlobe() {
  return globe;
}

export function getColorForFeature(feature) {
  if (state.currentColorMode === COLOR_MODE.FAMILY) {
    // In Family mode, all genera in the same family show the same family color
    const color = state.familyColors[feature.Family] || CONFIG.DEFAULT_COLOR;
    return color;
  } else if (state.currentColorMode === COLOR_MODE.GENUS) {
    // In Genus mode, each genus gets its unique shade of the family color
    const color = state.genusColors[feature.Genus] || state.familyColors[feature.Family] || CONFIG.DEFAULT_COLOR;
    if (!state.genusColors[feature.Genus]) {
      console.warn(`No color found for genus: ${feature.Genus}, Family: ${feature.Family}`);
    }
    return color;
  }
  return CONFIG.DEFAULT_COLOR;
}

export function calculatePointRadius(point) {
  return point.Area * CONFIG.POINT_AREA_SCALE + CONFIG.POINT_BASE_RADIUS;
}

export function calculatePolygonCenter(coordinates) {
  const ring = coordinates[0][0];
  let latSum = 0;
  let lngSum = 0;
  
  ring.forEach(coord => {
    lngSum += coord[0];
    latSum += coord[1];
  });
  
  const count = ring.length;
  return {
    lat: latSum / count,
    lng: lngSum / count
  };
}

export function createPointFromFeature(feature) {
  const center = calculatePolygonCenter(feature.geometry.coordinates);
  const properties = feature.properties;
  
  return {
    lat: center.lat,
    lng: center.lng,
    Site: `${properties.Family} - ${properties.Genus}`,
    Family: properties.Family,
    Genus: properties.Genus,
    Season: properties.Season,
    Area: properties.Area,
    year: properties.year,
    maxR: properties.Area * CONFIG.RING_AREA_SCALE + CONFIG.RING_BASE_RADIUS,
    propagationSpeed: CONFIG.RING_PROPAGATION_SPEED,
    repeatPeriod: CONFIG.RING_REPEAT_PERIOD
  };
}

export function createHexPolygonLabel({ properties: data }) {
  return `
    <b>${data.Family} - ${data.Genus}</b><br/>
    Season: <i>${data.Season}</i><br/>
    Area: <i>${data.Area.toFixed(2)} km²</i><br/>
    Year: <i>${data.year}</i>
  `;
}

export function switchToHexMode() {
  if (!globe) return;
  globe.hexPolygonsData(state.filteredFeatures);
  globe.pointsData([]);
  globe.ringsData([]);
}

export function switchToPointsMode() {
  if (!globe) return;
  globe.pointsData(state.filteredPoints);
  globe.hexPolygonsData([]);
  globe.ringsData(state.ringsEnabled ? state.filteredPoints : []);
}

export function refreshGlobeColors() {
  if (!globe) return;
  globe.hexPolygonsData(globe.hexPolygonsData());
  globe.pointsData(globe.pointsData());
  globe.ringsData(globe.ringsData());
}

export function initializeClouds() {
  if (!globe) return;
  
  new THREE.TextureLoader().load(CONFIG.CLOUDS_IMAGE_URL, cloudsTexture => {
    const cloudsGeometry = new THREE.SphereGeometry(
      globe.getGlobeRadius() * (1 + CONFIG.CLOUDS_ALTITUDE), 
      75, 
      75
    );
    const cloudsMaterial = new THREE.MeshPhongMaterial({ 
      map: cloudsTexture, 
      transparent: true 
    });
    
    state.cloudsMesh = new THREE.Mesh(cloudsGeometry, cloudsMaterial);
    animateClouds();
  });
}

function animateClouds() {
  if (state.cloudsMesh) {
    state.cloudsMesh.rotation.y += CONFIG.CLOUDS_ROTATION_SPEED * Math.PI / 180;
  }
  requestAnimationFrame(animateClouds);
}

export function toggleCloudsVisibility() {
  if (!state.cloudsMesh || !globe) return;
  
  const scene = globe.scene();
  if (scene.children.includes(state.cloudsMesh)) {
    scene.remove(state.cloudsMesh);
  } else {
    scene.add(state.cloudsMesh);
  }
}

export function toggleRings() {
  state.ringsEnabled = !state.ringsEnabled;
  if (state.currentDisplayMode === DISPLAY_MODE.POINTS) {
    globe.ringsData(state.ringsEnabled ? state.filteredPoints : []);
    refreshGlobeColors();
  }
}

export function applyGlobeStyle(style) {
  if (!globe) return;
  
  switch(style) {
    case 'detailed':
      applyDetailedStyle();
      break;
    case 'plain':
      applyPlainStyle();
      break;
    case 'normal':
    default:
      applyNormalStyle();
      break;
  }
}

function applyNormalStyle() {
  globe
    .globeImageUrl(CONFIG.EARTH_IMAGE_URL)
    .bumpImageUrl(CONFIG.EARTH_TOPOLOGY_URL)
    .showGlobe(true)
    .showAtmosphere(true);
  
  // Wait for textures to load, then reset globe material
  setTimeout(() => {
    const globeMaterial = globe.globeMaterial();
    globeMaterial.color = new THREE.Color(0xffffff); // Reset to white so texture shows properly
    globeMaterial.bumpScale = 1;
    globeMaterial.specularMap = null;
    globeMaterial.specular = new THREE.Color(0x111111);
    globeMaterial.shininess = 0;
    globeMaterial.transparent = false;
    globeMaterial.opacity = 1.0;
    globeMaterial.emissive = new THREE.Color(0x000000);
    globeMaterial.emissiveIntensity = 0;
    globeMaterial.needsUpdate = true;
    
    // Remove plain mode rim lights if they exist
    const scene = globe.scene();
    if (scene.userData.plainModeRimLight) {
      scene.remove(scene.userData.plainModeRimLight);
      scene.userData.plainModeRimLight = null;
    }
    if (scene.userData.plainModeRimLight2) {
      scene.remove(scene.userData.plainModeRimLight2);
      scene.userData.plainModeRimLight2 = null;
    }
  }, 100);
}

function applyDetailedStyle() {
  globe
    .globeImageUrl(CONFIG.EARTH_IMAGE_URL)
    .bumpImageUrl(CONFIG.EARTH_TOPOLOGY_URL)
    .showGlobe(true)
    .showAtmosphere(true);
  
  // Wait for textures to load, then enhance globe material
  setTimeout(() => {
    const globeMaterial = globe.globeMaterial();
    globeMaterial.color = new THREE.Color(0xffffff); // Reset to white so texture shows properly
    globeMaterial.bumpScale = 10;
    globeMaterial.transparent = false;
    globeMaterial.opacity = 1.0;
    globeMaterial.emissive = new THREE.Color(0x000000);
    globeMaterial.emissiveIntensity = 0;
    
    new THREE.TextureLoader().load('//cdn.jsdelivr.net/npm/three-globe/example/img/earth-water.png', texture => {
      globeMaterial.specularMap = texture;
      globeMaterial.specular = new THREE.Color('grey');
      globeMaterial.shininess = 15;
      globeMaterial.needsUpdate = true;
    });
    
    // Adjust light position for better effect
    const directionalLight = globe.lights().find(light => light.type === 'DirectionalLight');
    if (directionalLight) {
      directionalLight.position.set(1, 1, 1);
    }
    
    // Remove plain mode rim lights if they exist
    const scene = globe.scene();
    if (scene.userData.plainModeRimLight) {
      scene.remove(scene.userData.plainModeRimLight);
      scene.userData.plainModeRimLight = null;
    }
    if (scene.userData.plainModeRimLight2) {
      scene.remove(scene.userData.plainModeRimLight2);
      scene.userData.plainModeRimLight2 = null;
    }
  }, 100);
}

function applyPlainStyle() {
  globe
    .globeImageUrl(null)
    .bumpImageUrl(null)
    .showGlobe(true)
    .showAtmosphere(true); // Enable atmosphere for edge glow
  
  // Wait a frame, then set solid colors for globe with edge lighting
  setTimeout(() => {
    const globeMaterial = globe.globeMaterial();
    globeMaterial.color = new THREE.Color('#0a0e12'); // Much darker for plain mode
    globeMaterial.bumpScale = 0;
    globeMaterial.specularMap = null;
    globeMaterial.specular = new THREE.Color(0x111111);
    globeMaterial.shininess = 3;
    globeMaterial.transparent = false;
    globeMaterial.opacity = 1.0;
    globeMaterial.emissive = new THREE.Color('#050608'); // Very dark emissive
    globeMaterial.emissiveIntensity = 0.2;
    globeMaterial.needsUpdate = true;
    
    // Enhance lighting for better edge definition
    const scene = globe.scene();
    const existingLights = scene.children.filter(child => child.isLight);
    
    // Add rim lighting if not already present - using white for glow effect
    if (!scene.userData.plainModeRimLight) {
      const rimLight = new THREE.DirectionalLight('#FFFFFF', 0.4);
      rimLight.position.set(-2, 1, 3);
      scene.add(rimLight);
      scene.userData.plainModeRimLight = rimLight;
      
      const rimLight2 = new THREE.DirectionalLight('#FFFFFF', 0.3);
      rimLight2.position.set(2, -1, -3);
      scene.add(rimLight2);
      scene.userData.plainModeRimLight2 = rimLight2;
    }
  }, 100);
}

export function setNightSkyBackground() {
  if (!globe) return;
  globe.backgroundImageUrl('//cdn.jsdelivr.net/npm/three-globe/example/img/night-sky.png');
}

export function removeNightSkyBackground() {
  if (!globe) return;
  globe.backgroundImageUrl(null)
       .backgroundColor('rgba(15,17,21,1)'); // Background color from palette
}

export async function loadCountryBorders() {
  if (!globe) return;
  
  try {
    // Fetch TopoJSON data
    const response = await fetch(CONFIG.COUNTRIES_GEOJSON_URL);
    const topoData = await response.json();
    
    // Convert TopoJSON to GeoJSON using topojson library
    // The world-atlas data uses TopoJSON format which needs to be converted
    const countries = topojson.feature(topoData, topoData.objects.countries);
    
    // Set the countries data to the globe
    globe.polygonsData(countries.features);
    
    console.log('Country borders loaded successfully');
  } catch (error) {
    console.error('Error loading country borders:', error);
  }
}

export function toggleCountryBorders(show) {
  if (!globe) return;
  
  if (show) {
    loadCountryBorders();
  } else {
    globe.polygonsData([]);
  }
}

