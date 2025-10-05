// Configuration   RING_PROPAGATION_SPEED: 0.5,
export const CONFIG = {
  GEOJSON_FILES: ['../data/geojson/flowering_sites.geojson'],
  CLOUDS_IMAGE_URL: '/clouds.png',
  EARTH_IMAGE_URL: '//cdn.jsdelivr.net/npm/three-globe/example/img/earth-blue-marble.jpg',
  EARTH_TOPOLOGY_URL: '//cdn.jsdelivr.net/npm/three-globe/example/img/earth-topology.png',
  
  CLOUDS_ALTITUDE: 0.004,
  CLOUDS_ROTATION_SPEED: -0.006,
  
  HEX_POLYGON_RESOLUTION: 5,
  HEX_POLYGON_MARGIN: 0.3,
  
  POINT_AREA_SCALE: 0.0000005,
  POINT_BASE_RADIUS: 0.1,
  RING_AREA_SCALE: 0.000003,
  RING_BASE_RADIUS: 0.3,
  RING_PROPAGATION_SPEED: 0.5,
  RING_REPEAT_PERIOD: 500,
  
  DEFAULT_COLOR: '#00FFAE', // Green mint (pulse of life color)
  
  INITIAL_VIEW: {
    lat: 37,
    lng: -95,
    altitude: 2
  }
};

export const COLOR_MODE = {
  DEFAULT: 'default',
  FAMILY: 'family'
};

export const DISPLAY_MODE = {
  HEX: 'hex',
  POINTS: 'points'
};
