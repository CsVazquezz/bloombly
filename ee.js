// BloomWatch: Time-Series Bloom Detection (Final Working Version)
// This version fixes the "ComputedObject" error by correctly casting the
// output of the conditional function to a FeatureCollection for visualization.

// =================================================================================
// Section 1: Setup the Area of Interest (AOI) and Time Frame
// =================================================================================

// ===================== AOI Selection =====================
// Set these to switch between global, country, state, or custom region
// AOI_TYPE: 'global', 'country', 'state', or 'region'
// AOI_NAME: country name, state/region name, or leave blank for global
var AOI_TYPE = 'global'; // 'global', 'country', 'state', or 'region'
var AOI_COUNTRY = '';
var AOI_STATE = '';
var AOI_REGION = null; // Optionally set a custom region geometry

var aoi, aoi_name, map_zoom, map_label;
if (AOI_TYPE === 'global') {
  aoi = ee.Geometry.Rectangle([-180, -85, 180, 85]); // Covers most land areas
  aoi_name = 'Global';
  map_zoom = 2;
  map_label = 'Global AOI';
} else if (AOI_TYPE === 'country' && AOI_COUNTRY === 'Mexico') {
  aoi = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
    .filter(ee.Filter.eq('country_na', AOI_COUNTRY))
    .geometry();
  aoi_name = AOI_COUNTRY;
  map_zoom = 5;
  map_label = AOI_COUNTRY + ' AOI';
} else if (AOI_TYPE === 'country' && AOI_COUNTRY === 'United States') {
  aoi = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
    .filter(ee.Filter.eq('country_na', AOI_COUNTRY))
    .geometry();
  aoi_name = AOI_COUNTRY;
  map_zoom = 4;
  map_label = AOI_COUNTRY + ' AOI';
} else if (AOI_TYPE === 'state' && AOI_COUNTRY === 'Mexico') {
  var states = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM0_NAME', AOI_COUNTRY));
  var state = states.filter(ee.Filter.eq('ADM1_NAME', AOI_STATE));
  aoi = state.geometry();
  aoi_name = AOI_STATE + '_' + AOI_COUNTRY;
  map_zoom = 7;
  map_label = AOI_STATE + ' AOI';
} else if (AOI_TYPE === 'state' && AOI_COUNTRY === 'United States') {
  var states = ee.FeatureCollection('TIGER/2018/States')
    .filter(ee.Filter.eq('NAME', AOI_STATE));
  aoi = states.geometry();
  aoi_name = AOI_STATE + '_' + AOI_COUNTRY;
  map_zoom = 6;
  map_label = AOI_STATE + ' AOI';
} else if (AOI_TYPE === 'region' && AOI_REGION !== null) {
  aoi = AOI_REGION;
  aoi_name = 'CustomRegion';
  map_zoom = 6;
  map_label = 'Custom Region AOI';
} else {
  throw 'Unsupported AOI_TYPE or AOI_NAME';
}

var START_DATE = '2024-05-01';
var END_DATE = '2024-10-01';

Map.centerObject(aoi, map_zoom);
Map.addLayer(aoi, {color: 'grey'}, map_label);

// =================================================================================
// Section 2: Helper Functions for S2 Processing
// =================================================================================

function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi');
  return image.addBands(ndvi);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  // .map(maskS2clouds) // Cloud masking disabled for debugging
  .map(addNDVI);

// =================================================================================
// Section 3: Generate Bloom Snapshots Over Time
// =================================================================================

var time_step_days = 15;

var start = ee.Date(START_DATE).millis();
var end = ee.Date(END_DATE).millis();
var list_of_dates = ee.List.sequence(start, end, 1000 * 60 * 60 * 24 * time_step_days);

var getNewBlooms = function(date) {
  date = ee.Date(date);
  var current_period_start = date;
  var current_period_end = date.advance(time_step_days, 'day');
  var previous_period_start = date.advance(-time_step_days, 'day');
  
  var current_collection = s2.filterDate(current_period_start, current_period_end);
  var previous_collection = s2.filterDate(previous_period_start, current_period_start);
  
  var current_size = current_collection.size();
  var previous_size = previous_collection.size();
  var empty_features = ee.FeatureCollection([]).set('system:time_start', date.millis());
  
  // Compute server-side objects and return a FeatureCollection result.
  var hasData = current_size.gt(0).and(previous_size.gt(0));
  var current_image = current_collection.median();
  var previous_image = previous_collection.median();
  var ndvi_change = current_image.select('ndvi').subtract(previous_image.select('ndvi'));
  var bloom_threshold = 0.1;
  var ndvi_threshold = 0.4;
  var new_blooms_raster = ndvi_change.gt(bloom_threshold)
                               .and(current_image.select('ndvi').gt(ndvi_threshold));
  var bloom_vectors = new_blooms_raster.selfMask().reduceToVectors({
    geometry: aoi,
    scale: 250,
    geometryType: 'polygon',
    eightConnected: false,
    maxPixels: 1e9
  });

  // Add custom properties to each feature
  var siteName = 'Texas_Cotton'; // You can set this dynamically if needed
  var type = 'Wild';
  var season = 'Summer'; // Or derive from date
  // Calculate area in square meters for each polygon
  bloom_vectors = bloom_vectors.map(function(f) {
    var area = f.geometry().area(1);
    return f.set({
      'Site': siteName,
      'Type': type,
      'Season': season,
      'Area': area
    });
  });

  // Calculate total bloom area for this period
  var total_bloom_area = bloom_vectors.aggregate_sum('Area');
  // Define superbloom threshold (adjust as needed)
  var superbloom_threshold = 1000000; // 1,000,000 sq meters (example)
  var isSuperbloom = ee.Number(total_bloom_area).gt(superbloom_threshold);

  // Cast geometry to MultiPolygon if needed
  bloom_vectors = bloom_vectors.map(function(f) {
    var geom = f.geometry();
    var isPolygon = ee.String(geom.type()).compareTo('Polygon').eq(0);
    var multi = ee.Algorithms.If(isPolygon, ee.Geometry.MultiPolygon([geom.coordinates()]), geom);
    return f.setGeometry(multi);
  });

  // Set superbloom properties on the FeatureCollection
  bloom_vectors = bloom_vectors.set('system:time_start', date.millis())
                               .set('total_bloom_area', total_bloom_area)
                               .set('isSuperbloom', isSuperbloom);
  var result = ee.Algorithms.If(hasData, bloom_vectors, empty_features);
  return ee.FeatureCollection(result);
};

// =================================================================================
// Section 4: Prepare Exports
// =================================================================================

// Visualize just one example to confirm the logic works.
var example_date = '2024-07-30';
// *** FINAL FIX: Cast the computed object to a FeatureCollection so the map knows how to draw it. ***
var current_period_start_dbg = ee.Date(example_date);
var current_period_end_dbg = current_period_start_dbg.advance(time_step_days, 'day');
var previous_period_start_dbg = current_period_start_dbg.advance(-time_step_days, 'day');
var current_collection_dbg = s2.filterDate(current_period_start_dbg, current_period_end_dbg);
var previous_collection_dbg = s2.filterDate(previous_period_start_dbg, current_period_start_dbg);
print('Current collection size:', current_collection_dbg.size());
print('Previous collection size:', previous_collection_dbg.size());
var current_image_dbg = current_collection_dbg.median();
var previous_image_dbg = previous_collection_dbg.median();
var ndvi_change_dbg = current_image_dbg.select('ndvi').subtract(previous_image_dbg.select('ndvi'));

// Print NDVI stats for debugging
print('Current NDVI min/max:', current_image_dbg.select('ndvi').reduceRegion({reducer: ee.Reducer.minMax(), geometry: aoi, scale: 250, maxPixels: 1e9}));
print('Previous NDVI min/max:', previous_image_dbg.select('ndvi').reduceRegion({reducer: ee.Reducer.minMax(), geometry: aoi, scale: 250, maxPixels: 1e9}));
print('NDVI change min/max:', ndvi_change_dbg.reduceRegion({reducer: ee.Reducer.minMax(), geometry: aoi, scale: 250, maxPixels: 1e9}));

var example_blooms = ee.FeatureCollection(getNewBlooms(example_date));
var bloom_count = example_blooms.size();
var superbloom_flag = example_blooms.get('isSuperbloom');
var total_bloom_area = example_blooms.get('total_bloom_area');
print('New Blooms feature count for', example_date, ':', bloom_count);
print('Total bloom area (sq m):', total_bloom_area);
print('Superbloom detected?', superbloom_flag);
// Visualize superblooms in gold, others in magenta
var superbloomVis = ee.Algorithms.If(superbloom_flag,
  {color: '#FFD700'}, // gold
  {color: '#FF00FF'}  // magenta
);
Map.addLayer(example_blooms, superbloomVis, 'New Blooms for ' + example_date);
// Warn if empty
print('Warning: FeatureCollection is empty?', bloom_count.eq(0));

// This loop PREPARES the export tasks without running them.
list_of_dates.evaluate(function(dates) {
  for (var i = 0; i < dates.length; i++) {
    var date = ee.Date(dates[i]);
    // The cast is also needed for the export to know the type.
    var blooms = ee.FeatureCollection(getNewBlooms(date));
    var date_string = date.format('YYYY-MM-dd').getInfo();
    
    Export.table.toDrive({
      collection: blooms,
      description: 'Blooms_' + aoi_name + '_' + date_string,
      fileFormat: 'GeoJSON',
      fileNamePrefix: 'Blooms_' + aoi_name + '_' + date_string
    });
  }
});

print("Script finished. Go to the 'Tasks' tab to run your exports.");