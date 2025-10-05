"""
Data Cleaning Script for Bloombly
Processes multiple bloom datasets and prepares them for ML training.
Handles: Japanese Cherry Blossoms, Symphyotrichum, and Sweet Cherry data.
Preserves critical taxonomic data: family, genus, season.
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import numpy as np

class BloomDataCleaner:
    """Clean and prepare bloom observation data from multiple sources for ML pipeline"""
    
    def __init__(self, raw_data_dir='../../data/raw', processed_data_dir='../../data/processed'):
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        
        # Resolve paths relative to the script location
        if not Path(raw_data_dir).is_absolute():
            self.raw_data_dir = (script_dir / raw_data_dir).resolve()
        else:
            self.raw_data_dir = Path(raw_data_dir)
            
        if not Path(processed_data_dir).is_absolute():
            self.processed_data_dir = (script_dir / processed_data_dir).resolve()
        else:
            self.processed_data_dir = Path(processed_data_dir)
            
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Taxonomic information for cherry blossoms (Japanese)
        self.cherry_taxonomy = {
            'family': 'Rosaceae',
            'genus': 'Prunus',
            'species': 'serrulata',  # Somei Yoshino variety
            'common_name': 'Japanese Cherry Blossom'
        }
        
    def determine_season(self, date_str):
        """Determine season from date string"""
        try:
            if isinstance(date_str, str):
                date = datetime.strptime(date_str, '%Y-%m-%d')
            else:
                date = date_str
            month = date.month
            if month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            elif month in [9, 10, 11]:
                return "Fall"
            else:
                return "Winter"
        except:
            return "Unknown"
    
    def clean_japanese_cherry_data(self, forecasts_path, places_path):
        """
        Clean Japanese cherry blossom forecast data (Kaggle 2024 dataset).
        Combines forecasts with place information.
        
        Args:
            forecasts_path: Path to cherry_blossom_forecasts.csv
            places_path: Path to cherry_blossom_places.csv
            
        Returns:
            pandas DataFrame with cleaned cherry blossom data
        """
        print(f" Processing Japanese Cherry Blossom Data...")
        print(f" Reading forecasts from: {forecasts_path}")
        print(f" Reading places from: {places_path}")

        # Load both files
        forecasts = pd.read_csv(forecasts_path)
        places = pd.read_csv(places_path)
        
        print(f" Forecasts: {len(forecasts)} rows")
        print(f" Places: {len(places)} locations")

        # Rename 'code' to 'place_code' in places for merging
        if 'code' in places.columns and 'place_code' not in places.columns:
            places = places.rename(columns={'code': 'place_code'})
        
        # Merge on place_code
        merged = forecasts.merge(places, on='place_code', how='left')
        
        # Create cleaned DataFrame
        cleaned = pd.DataFrame()
        
        # Generate unique IDs
        cleaned['record_id'] = 'jcb_' + merged['place_code'].astype(str) + '_' + merged['date'].str.replace('-', '')
        cleaned['data_source'] = 'Japan Meteorological Agency / Kaggle 2024'
        
        # Taxonomy (all are Japanese cherry blossoms)
        cleaned['scientific_name'] = 'Prunus × yedoensis'  # Somei Yoshino
        cleaned['family'] = self.cherry_taxonomy['family']
        cleaned['genus'] = self.cherry_taxonomy['genus']
        cleaned['species'] = 'yedoensis'
        cleaned['common_name'] = 'Japanese Cherry Blossom (Somei Yoshino)'
        
        # Location information
        cleaned['location_name'] = merged.get('spot_name', merged.get('name', ''))
        cleaned['prefecture'] = merged.get('prefecture_en', merged.get('prefecture', ''))
        cleaned['region'] = merged.get('prefecture_en', merged.get('region', ''))
        cleaned['latitude'] = pd.to_numeric(merged.get('lat', merged.get('latitude')), errors='coerce')
        cleaned['longitude'] = pd.to_numeric(merged.get('lon', merged.get('lng', merged.get('longitude'))), errors='coerce')
        
        # Bloom dates - use kaika_date (first bloom) as primary
        cleaned['date'] = pd.to_datetime(merged['kaika_date'], errors='coerce')
        cleaned['full_bloom_date'] = pd.to_datetime(merged['mankai_date'], errors='coerce')
        cleaned['year'] = cleaned['date'].dt.year
        cleaned['day_of_year'] = cleaned['date'].dt.dayofyear
        cleaned['full_bloom_day_of_year'] = cleaned['full_bloom_date'].dt.dayofyear
        cleaned['month'] = cleaned['date'].dt.month
        cleaned['season'] = cleaned['date'].apply(self.determine_season)
        
        # Climate data
        cleaned['temperature_avg'] = pd.to_numeric(merged.get('tavg'), errors='coerce')
        cleaned['temperature_min'] = pd.to_numeric(merged.get('tmin'), errors='coerce')
        cleaned['temperature_max'] = pd.to_numeric(merged.get('tmax'), errors='coerce')
        cleaned['precipitation'] = pd.to_numeric(merged.get('prcp'), errors='coerce')
        
        # Metadata
        cleaned['trait'] = 'first flowering (kaika)'
        cleaned['basis_of_record'] = 'Forecast/Observation'
        cleaned['is_prediction'] = True  # 2024 data is forecast
        
        # Remove rows with missing critical data
        print(f" Removing rows with missing coordinates or dates...")
        initial_count = len(cleaned)
        cleaned = cleaned.dropna(subset=['latitude', 'longitude', 'date', 'day_of_year'])
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} rows with missing data")
        
        # Remove invalid coordinates
        print(f" Validating coordinates for Japan...")
        initial_count = len(cleaned)
        # Japan bounds: lat 24-46, lng 123-146
        cleaned = cleaned[
            (cleaned['latitude'].between(24, 46)) & 
            (cleaned['longitude'].between(123, 146))
        ]
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} rows with invalid coordinates")
        
        # Remove duplicates
        print(f" Removing duplicates...")
        initial_count = len(cleaned)
        cleaned = cleaned.drop_duplicates(
            subset=['location_name', 'latitude', 'longitude', 'date'],
            keep='first'
        )
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} duplicate observations")
        
        # Sort by date
        cleaned = cleaned.sort_values(['year', 'day_of_year']).reset_index(drop=True)
        
        print(f" Japanese Cherry Blossoms cleaned: {len(cleaned)} observations")
        
        return cleaned
    
    def clean_historical_sakura_data(self, sakura_bloom_path, cities_geocoded_path):
        """
        Clean historical Japanese cherry blossom data (1953-2025).
        Combines JMA observation data with geocoded city coordinates.
        
        Args:
            sakura_bloom_path: Path to sakura_first_bloom_dates.csv
            cities_geocoded_path: Path to japan_cities_geocoded.csv
            
        Returns:
            pandas DataFrame with cleaned historical cherry blossom data
        """
        print(f"\n Processing Historical Japanese Sakura Data (1953-2025)...")
        print(f" Reading bloom dates from: {sakura_bloom_path}")
        print(f" Reading geocoded cities from: {cities_geocoded_path}")

        # Load bloom dates and geocoded cities
        blooms = pd.read_csv(sakura_bloom_path)
        cities = pd.read_csv(cities_geocoded_path)
        
        print(f" Cities: {len(blooms)} locations")
        print(f" Geocoded cities: {len(cities)} with coordinates")
        
        # Create a dictionary mapping city names to their coordinates
        city_lookup = cities.set_index('city_name').to_dict('index')
        
        # Prepare data for transformation
        records = []
        
        # Get year columns (exclude first 2 columns: Site Name, Currently Being Observed)
        # and last 2 columns: 30 Year Average, Notes
        year_columns = blooms.columns[2:-2]
        
        print(f" Processing bloom dates from {len(year_columns)} years...")
        
        for idx, row in blooms.iterrows():
            city_name = row['Site Name']
            currently_observed = row['Currently Being Observed']
            notes = row.get('Notes', '') if pd.notna(row.get('Notes', '')) else ''
            
            # Get city coordinates
            city_info = city_lookup.get(city_name, {})
            latitude = city_info.get('latitude')
            longitude = city_info.get('longitude')
            elevation = city_info.get('elevation_m')
            
            if latitude is None or longitude is None:
                print(f"   Warning: No coordinates for {city_name}, skipping...")
                continue
            
            # Determine species from notes
            if 'Sargent cherry' in notes or 'Prunus sargentii' in notes:
                species_name = 'Prunus sargentii'
                common_name = 'Sargent Cherry'
                species_code = 'sargentii'
            elif 'Taiwan cherry' in notes or 'Prunus campanulata' in notes:
                species_name = 'Prunus campanulata'
                common_name = 'Taiwan Cherry'
                species_code = 'campanulata'
            elif 'Kurile' in notes or 'kurilensis' in notes:
                species_name = 'Cerasus nipponica var. kurilensis'
                common_name = 'Kurile Island Cherry'
                species_code = 'kurilensis'
            else:
                # Default to Yoshino (most common)
                species_name = 'Prunus × yedoensis'
                common_name = 'Yoshino Cherry'
                species_code = 'yedoensis'
            
            # Process each year's bloom date
            for year_col in year_columns:
                bloom_date_str = row[year_col]
                
                # Skip empty/NaN values
                if pd.isna(bloom_date_str) or bloom_date_str == '':
                    continue
                
                try:
                    # Parse the datetime
                    bloom_date = pd.to_datetime(bloom_date_str)
                    
                    # Extract year from column name (should match bloom_date year)
                    year = int(year_col)
                    
                    # Create record
                    record = {
                        'record_id': f'jma_sakura_{city_name.lower().replace(" ", "_")}_{year}',
                        'data_source': 'Japan Meteorological Agency (JMA) Historical',
                        'scientific_name': species_name,
                        'family': 'Rosaceae',
                        'genus': species_name.split()[0],  # First word of scientific name
                        'species': species_code,
                        'common_name': common_name,
                        'location_name': city_name,
                        'prefecture': '',  # Could be extracted from region if needed
                        'region': 'Japan',
                        'latitude': latitude,
                        'longitude': longitude,
                        'elevation_m': elevation,
                        'date': bloom_date,
                        'year': year,
                        'month': bloom_date.month,
                        'day_of_year': bloom_date.timetuple().tm_yday,
                        'season': self.determine_season(bloom_date),
                        'full_bloom_date': pd.NaT,  # Not available in this dataset
                        'full_bloom_day_of_year': np.nan,
                        'temperature_avg': np.nan,  # Will be added by feature engineering
                        'temperature_min': np.nan,
                        'temperature_max': np.nan,
                        'precipitation': np.nan,
                        'trait': 'first flowering (kaika)',
                        'basis_of_record': 'Human Observation',
                        'is_prediction': False,  # Historical observations
                        'currently_observed': currently_observed,
                        'species_notes': notes
                    }
                    
                    records.append(record)
                    
                except Exception as e:
                    print(f"   Warning: Could not parse date '{bloom_date_str}' for {city_name} in {year_col}: {e}")
                    continue
        
        # Create DataFrame from records
        cleaned = pd.DataFrame(records)
        
        if len(cleaned) == 0:
            print(f" WARNING: No historical sakura data could be processed!")
            return pd.DataFrame()
        
        # Remove duplicates
        print(f" Removing duplicates...")
        initial_count = len(cleaned)
        cleaned = cleaned.drop_duplicates(
            subset=['location_name', 'latitude', 'longitude', 'year', 'day_of_year'],
            keep='first'
        )
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} duplicate observations")
        
        # Sort by date
        cleaned = cleaned.sort_values(['year', 'day_of_year', 'location_name']).reset_index(drop=True)
        
        # Summary statistics
        print(f"\n Historical Sakura Data Summary:")
        print(f"   Total observations: {len(cleaned):,}")
        print(f"   Cities: {cleaned['location_name'].nunique()}")
        print(f"   Year range: {cleaned['year'].min()} - {cleaned['year'].max()}")
        print(f"   Species: {cleaned['scientific_name'].nunique()}")
        for species in cleaned['scientific_name'].unique():
            count = (cleaned['scientific_name'] == species).sum()
            print(f"     • {species}: {count:,} observations")
        print(f"   Currently observed stations: {cleaned['currently_observed'].sum()}")
        print(f"   Geographic range:")
        print(f"     • Latitude: {cleaned['latitude'].min():.2f}° to {cleaned['latitude'].max():.2f}°")
        print(f"     • Longitude: {cleaned['longitude'].min():.2f}° to {cleaned['longitude'].max():.2f}°")
        print(f"     • Elevation: {cleaned['elevation_m'].min():.0f}m to {cleaned['elevation_m'].max():.0f}m")
        
        return cleaned
    
    def clean_symphyotrichum_data(self, csv_path):
        """
        Clean Symphyotrichum species data (National Phenology Network).
        
        Args:
            csv_path: Path to data.csv (Symphyotrichum observations)
            
        Returns:
            pandas DataFrame with cleaned data
        """
        print(f"\n Processing Symphyotrichum Species Data...")
        print(f" Reading data from: {csv_path}")

        df = pd.read_csv(csv_path, encoding='utf-8')

        print(f" Initial rows: {len(df)}")
        
        # Create cleaned copy
        cleaned = pd.DataFrame()
        
        # Extract core fields
        cleaned['record_id'] = 'npn_' + df['annotationID'].str.replace('npn:', '')
        cleaned['data_source'] = df.get('dataSource', 'National Phenology Network')
        cleaned['scientific_name'] = df['scientificName']
        cleaned['family'] = df['family']
        cleaned['genus'] = df['genus']
        cleaned['species'] = df.get('species', '')
        cleaned['common_name'] = df['scientificName'].apply(self._get_common_name)
        
        # Location
        cleaned['location_name'] = ''  # Not provided in this dataset
        cleaned['prefecture'] = ''
        cleaned['region'] = 'North America'  # Based on lat/lng
        cleaned['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        cleaned['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Date and temporal features
        cleaned['date'] = pd.to_datetime(df['date'], errors='coerce')
        cleaned['year'] = df['year']
        cleaned['day_of_year'] = df.get('dayOfYear', cleaned['date'].dt.dayofyear)
        cleaned['month'] = cleaned['date'].dt.month
        cleaned['season'] = cleaned['date'].apply(self.determine_season)
        cleaned['full_bloom_date'] = None  # Not tracked for these species
        cleaned['full_bloom_day_of_year'] = None
        
        # Climate data (not available in this dataset)
        cleaned['temperature_avg'] = None
        cleaned['temperature_min'] = None
        cleaned['temperature_max'] = None
        cleaned['precipitation'] = None
        
        # Observation metadata
        cleaned['trait'] = df.get('trait', 'open flower present')
        cleaned['basis_of_record'] = df.get('basisOfRecord', 'Human Observation')
        cleaned['is_prediction'] = False  # Historical observations
        
        # Remove rows with missing critical data
        print(f"🧹 Removing rows with missing coordinates or dates...")
        initial_count = len(cleaned)
        cleaned = cleaned.dropna(subset=['latitude', 'longitude', 'date', 'day_of_year'])
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} rows with missing data")
        
        # Remove invalid coordinates
        print(f"🧹 Removing invalid coordinates...")
        initial_count = len(cleaned)
        cleaned = cleaned[
            (cleaned['latitude'].between(-90, 90)) & 
            (cleaned['longitude'].between(-180, 180))
        ]
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} rows with invalid coordinates")
        
        # Remove duplicates
        print(f"🧹 Removing duplicates...")
        initial_count = len(cleaned)
        cleaned = cleaned.drop_duplicates(
            subset=['scientific_name', 'latitude', 'longitude', 'date'],
            keep='first'
        )
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} duplicate observations")
        
        # Sort by date
        cleaned = cleaned.sort_values(['year', 'day_of_year']).reset_index(drop=True)
        
        print(f" Symphyotrichum data cleaned: {len(cleaned)} observations")
        
        return cleaned
    
    def clean_sweet_cherry_data(self, csv_path):
        """
        Clean Sweet Cherry phenology data (1978-2015 European dataset).
        
        Args:
            csv_path: Path to Sweet_cherry_phenology_data_1978-2015.csv
            
        Returns:
            pandas DataFrame with cleaned sweet cherry data
        """
        print(f"\n Processing Sweet Cherry Phenology Data...")
        print(f" Reading data from: {csv_path}")

        df = pd.read_csv(csv_path, encoding='utf-8')

        print(f" Initial rows: {len(df)}")
        
        # Create cleaned copy
        cleaned = pd.DataFrame()
        
        # Generate unique IDs
        cleaned['record_id'] = 'swc_' + df.index.astype(str)
        cleaned['data_source'] = df['Country'] + ' - ' + df['Institute']
        
        # Taxonomy (all are Sweet Cherry)
        cleaned['scientific_name'] = 'Prunus avium'
        cleaned['family'] = 'Rosaceae'
        cleaned['genus'] = 'Prunus'
        cleaned['species'] = 'avium'
        cleaned['common_name'] = 'Sweet Cherry'
        
        # Location information
        cleaned['location_name'] = df['Site']
        cleaned['prefecture'] = df['Country']
        cleaned['region'] = df['Country']
        cleaned['latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        cleaned['longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        # Parse dates - the file uses "Beginning of flowering" column (day of year)
        cleaned['year'] = pd.to_numeric(df['Year'], errors='coerce')
        cleaned['day_of_year'] = pd.to_numeric(df['Beginning of flowering'], errors='coerce')
        
        # Create actual date from year and day of year
        def create_date(row):
            try:
                if pd.notnull(row['year']) and pd.notnull(row['day_of_year']):
                    return datetime.strptime(f"{int(row['year'])}-{int(row['day_of_year'])}", "%Y-%j")
            except:
                pass
            return None
        
        cleaned['date'] = cleaned.apply(create_date, axis=1)
        cleaned['month'] = cleaned['date'].dt.month
        cleaned['season'] = cleaned['date'].apply(self.determine_season)
        
        # Full flowering data
        cleaned['full_bloom_day_of_year'] = pd.to_numeric(df.get('Full Flowering'), errors='coerce')
        
        def create_full_bloom_date(row):
            try:
                if pd.notnull(row['year']) and pd.notnull(row['full_bloom_day_of_year']):
                    return datetime.strptime(f"{int(row['year'])}-{int(row['full_bloom_day_of_year'])}", "%Y-%j")
            except:
                pass
            return None
        
        cleaned['full_bloom_date'] = cleaned.apply(create_full_bloom_date, axis=1)
        
        # Climate data (not available in this dataset)
        cleaned['temperature_avg'] = None
        cleaned['temperature_min'] = None
        cleaned['temperature_max'] = None
        cleaned['precipitation'] = None
        
        # Metadata
        cleaned['trait'] = 'beginning of flowering'
        cleaned['basis_of_record'] = 'Field Observation'
        cleaned['is_prediction'] = False  # Historical observations
        
        # Add cultivar info for reference (optional)
        # cleaned['cultivar'] = df.get('Cultivar', '')
        
        # Remove rows with missing critical data
        print(f"🧹 Removing rows with missing coordinates or dates...")
        initial_count = len(cleaned)
        cleaned = cleaned.dropna(subset=['latitude', 'longitude', 'year', 'day_of_year'])
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} rows with missing data")
        
        # Remove invalid coordinates (Europe bounds approximately)
        print(f"🧹 Validating coordinates for Europe...")
        initial_count = len(cleaned)
        # Europe bounds: lat 35-71, lng -10 to 40
        cleaned = cleaned[
            (cleaned['latitude'].between(35, 71)) & 
            (cleaned['longitude'].between(-10, 40))
        ]
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} rows with invalid coordinates")
        
        # Remove duplicates
        print(f"🧹 Removing duplicates...")
        initial_count = len(cleaned)
        cleaned = cleaned.drop_duplicates(
            subset=['location_name', 'latitude', 'longitude', 'year', 'day_of_year'],
            keep='first'
        )
        removed = initial_count - len(cleaned)
        print(f"   Removed {removed} duplicate observations")
        
        # Sort by date
        cleaned = cleaned.sort_values(['year', 'day_of_year']).reset_index(drop=True)
        
        print(f" Sweet Cherry data cleaned: {len(cleaned)} observations")
        
        return cleaned
    
    def _get_common_name(self, scientific_name):
        """Map scientific names to common names"""
        common_names = {
            'Symphyotrichum novae-angliae': 'New England Aster',
            'Symphyotrichum ericoides': 'White Heath Aster',
            'Prunus × yedoensis': 'Yoshino Cherry',
            'Prunus avium': 'Sweet Cherry'
        }
        return common_names.get(scientific_name, scientific_name)
    
    def combine_datasets(self, *dataframes):
        """
        Combine multiple cleaned datasets into one unified dataset.
        
        Args:
            *dataframes: Variable number of cleaned DataFrames
            
        Returns:
            Combined DataFrame with all observations
        """
        print(f"\n Combining {len(dataframes)} datasets...")
        
        # Concatenate all dataframes
        combined = pd.concat(dataframes, ignore_index=True)
        
        # Ensure consistent column order
        column_order = [
            'record_id', 'data_source', 'scientific_name', 'family', 'genus', 'species', 
            'common_name', 'location_name', 'prefecture', 'region',
            'latitude', 'longitude', 'date', 'year', 'month', 'day_of_year', 'season',
            'full_bloom_date', 'full_bloom_day_of_year',
            'temperature_avg', 'temperature_min', 'temperature_max', 'precipitation',
            'trait', 'basis_of_record', 'is_prediction'
        ]
        
        # Add optional columns if they exist
        optional_columns = ['elevation_m', 'currently_observed', 'species_notes']
        for col in optional_columns:
            if col in combined.columns and col not in column_order:
                column_order.append(col)
        
        # Select only columns that exist in the dataframe
        available_columns = [col for col in column_order if col in combined.columns]
        combined = combined[available_columns]
        
        print(f" Combined dataset: {len(combined)} total observations")
        print(f"   Species: {combined['scientific_name'].nunique()}")
        print(f"   Year range: {combined['year'].min()} - {combined['year'].max()}")
        
        return combined
    
    def add_climate_indicators(self, df):
        """
        Add basic climate change indicators and temporal trends.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with additional features
        """
        print(f"\n Adding climate indicators...")
        
        # Normalize year (useful for ML)
        min_year = df['year'].min()
        max_year = df['year'].max()
        df['year_normalized'] = (df['year'] - min_year) / max(1, (max_year - min_year))
        
        # Create decade bins for analysis
        df['decade'] = (df['year'] // 10) * 10
        
        # Calculate historical baseline per species and region
        historical_cutoff = df['year'].quantile(0.3)  # First 30% of data as baseline
        
        print(f"   Calculating deviations from baseline (year <= {historical_cutoff:.0f})...")
        
        def calc_deviation(row):
            """Calculate deviation from historical average for this species in this region"""
            # Group by species and approximate region (rounded lat/lng)
            species_region_data = df[
                (df['scientific_name'] == row['scientific_name']) &
                (df['year'] <= historical_cutoff) &
                (abs(df['latitude'] - row['latitude']) < 5) &  # ~500km radius
                (abs(df['longitude'] - row['longitude']) < 5)
            ]
            if len(species_region_data) > 5:  # Need minimum data points
                baseline = species_region_data['day_of_year'].mean()
                return row['day_of_year'] - baseline
            return None
        
        df['deviation_from_baseline'] = df.apply(calc_deviation, axis=1)
        
        # Add location-based grouping (round to ~10km grid)
        df['lat_grid'] = (df['latitude'] * 10).round() / 10
        df['lng_grid'] = (df['longitude'] * 10).round() / 10
        df['location_grid'] = df['lat_grid'].astype(str) + '_' + df['lng_grid'].astype(str)
        
        print(f" Climate indicators added")
        
        return df
    
    def create_species_summary(self, df):
        """
        Create summary statistics per species for quick reference.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Summary DataFrame
        """
        print(f"\n Creating species summary...")
        
        summary = df.groupby(['family', 'genus', 'scientific_name', 'common_name']).agg({
            'record_id': 'count',
            'year': ['min', 'max'],
            'day_of_year': ['mean', 'std', 'min', 'max'],
            'latitude': ['mean', 'min', 'max'],
            'longitude': ['mean', 'min', 'max'],
            'season': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        }).reset_index()
        
        summary.columns = [
            'family', 'genus', 'scientific_name', 'common_name',
            'observation_count', 'first_year', 'last_year',
            'avg_bloom_day', 'bloom_day_std', 'earliest_bloom', 'latest_bloom',
            'center_lat', 'min_lat', 'max_lat',
            'center_lng', 'min_lng', 'max_lng',
            'primary_season'
        ]
        
        # Calculate years of observations
        summary['years_observed'] = summary['last_year'] - summary['first_year'] + 1
        
        # Sort by observation count
        summary = summary.sort_values('observation_count', ascending=False)
        
        print(f" Summary created for {len(summary)} species")
        
        return summary
    
    def export_for_ml(self, df, filename='clean_blooms_ml.csv'):
        """
        Export cleaned data optimized for ML training.
        
        Args:
            df: Cleaned DataFrame
            filename: Output filename
        """
        output_path = self.processed_data_dir / filename
        
        # Select relevant columns for ML
        ml_columns = [
            'record_id', 'scientific_name', 'family', 'genus', 'species', 'common_name',
            'year', 'year_normalized', 'day_of_year', 'month', 'season', 'decade',
            'latitude', 'longitude', 'location_grid', 'region', 'prefecture',
            'temperature_avg', 'temperature_min', 'temperature_max', 'precipitation',
            'deviation_from_baseline', 'trait', 'is_prediction'
        ]
        
        ml_df = df[ml_columns].copy()
        ml_df.to_csv(output_path, index=False)
        
        print(f"\n💾 ML data exported to: {output_path}")
        print(f"   Rows: {len(ml_df):,}")
        print(f"   Columns: {len(ml_df.columns)}")
        
        return output_path
    
    def export_to_geojson(self, df, filename='blooms.geojson'):
        """
        Export cleaned data to GeoJSON for visualization.
        
        Args:
            df: Cleaned DataFrame
            filename: Output filename
        """
        output_path = self.processed_data_dir / filename
        
        features = []
        
        print(f"\n🗺️ Generating GeoJSON features...")
        
        for _, row in df.iterrows():
            # Create a small square around the point (0.01 degrees ~ 1km)
            size = 0.005
            lat, lon = row['latitude'], row['longitude']
            
            square_coords = [
                [lon - size, lat - size],
                [lon + size, lat - size],
                [lon + size, lat + size],
                [lon - size, lat + size],
                [lon - size, lat - size]
            ]
            
            properties = {
                "id": str(row['record_id']),
                "Site": row.get('location_name') or row['scientific_name'],
                "Family": row['family'],
                "Genus": row['genus'],
                "Species": row.get('species', ''),
                "CommonName": row.get('common_name', ''),
                "Season": row['season'],
                "Year": int(row['year']),
                "DayOfYear": int(row['day_of_year']),
                "Date": row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else '',
                "Area": 1.0,
                "Latitude": float(row['latitude']),
                "Longitude": float(row['longitude']),
                "Region": row.get('region', ''),
                "Prefecture": row.get('prefecture', ''),
                "Temperature": float(row['temperature_avg']) if pd.notnull(row.get('temperature_avg')) else None,
                "DeviationFromBaseline": float(row.get('deviation_from_baseline', 0)) if pd.notnull(row.get('deviation_from_baseline')) else None,
                "IsPrediction": bool(row.get('is_prediction', False))
            }
            
            geometry = {
                "type": "MultiPolygon",
                "coordinates": [[square_coords]]
            }
            
            feature = {
                "type": "Feature",
                "properties": properties,
                "geometry": geometry
            }
            
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "name": "Global_Bloom_Observations",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                }
            },
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f" GeoJSON exported to: {output_path}")
        print(f"   Features: {len(features):,}")
        
        return output_path
    
    def generate_summary_stats(self, df):
        """Print summary statistics about the cleaned data"""
        print("\n" + "="*70)
        print(" DATASET SUMMARY STATISTICS")
        print("="*70)
        
        print(f"\nTemporal Coverage:")
        print(f"   Years: {df['year'].min()} - {df['year'].max()}")
        print(f"   Total observations: {len(df):,}")
        print(f"   Predictions: {df['is_prediction'].sum():,}")
        print(f"   Historical: {(~df['is_prediction']).sum():,}")
        
        print(f"\n Taxonomic Diversity:")
        print(f"   Families: {df['family'].nunique()}")
        print(f"   Genera: {df['genus'].nunique()}")
        print(f"   Species: {df['scientific_name'].nunique()}")
        
        print(f"\n Species Breakdown:")
        for species in df['scientific_name'].unique():
            count = len(df[df['scientific_name'] == species])
            common = df[df['scientific_name'] == species]['common_name'].iloc[0]
            pct = (count / len(df)) * 100
            print(f"   {species} ({common}): {count:,} ({pct:.1f}%)")
        
        print(f"\n Seasonal Distribution:")
        for season in ['Spring', 'Summer', 'Fall', 'Winter']:
            count = len(df[df['season'] == season])
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"   {season}: {count:,} ({pct:.1f}%)")
        
        print(f"\n Geographic Coverage:")
        print(f"   Latitude range: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
        print(f"   Longitude range: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
        print(f"   Regions: {df['region'].nunique()}")
        print(f"   Unique locations: {df['location_grid'].nunique():,}")
        
        print(f"\n Bloom Timing:")
        print(f"   Average day of year: {df['day_of_year'].mean():.1f}")
        print(f"   Std deviation: {df['day_of_year'].std():.1f} days")
        print(f"   Earliest: Day {df['day_of_year'].min()} ({self.day_to_date(df['year'].iloc[0], df['day_of_year'].min())})")
        print(f"   Latest: Day {df['day_of_year'].max()} ({self.day_to_date(df['year'].iloc[0], df['day_of_year'].max())})")
        
        # Climate data availability
        if 'temperature_avg' in df.columns:
            temp_available = df['temperature_avg'].notna().sum()
            pct_temp = (temp_available / len(df)) * 100
            print(f"\n Climate Data Availability:")
            print(f"   Temperature data: {temp_available:,} observations ({pct_temp:.1f}%)")
            if temp_available > 0:
                print(f"   Avg temperature: {df['temperature_avg'].mean():.1f}°C")
                print(f"   Temperature range: {df['temperature_avg'].min():.1f}°C to {df['temperature_avg'].max():.1f}°C")
        
        print("\n" + "="*70 + "\n")
    
    def day_to_date(self, year, day_of_year):
        """Convert day of year to readable date"""
        try:
            return datetime.strptime(f"{int(year)}-{int(day_of_year)}", "%Y-%j").strftime("%b %d")
        except:
            return "N/A"
    
    def run_full_pipeline(self, cherry_forecasts=None, cherry_places=None, 
                         symphyotrichum_data=None, sweet_cherry_data=None,
                         historical_sakura=None, sakura_cities_geocoded=None):
        """
        Execute the complete data cleaning pipeline for all datasets.
        
        Args:
            cherry_forecasts: Path to cherry_blossom_forecasts.csv (or None to skip)
            cherry_places: Path to cherry_blossom_places.csv (or None to skip)
            symphyotrichum_data: Path to data.csv (or None to skip)
            sweet_cherry_data: Path to sweet cherry data (or None to skip)
            historical_sakura: Path to sakura_first_bloom_dates.csv (or None to skip)
            sakura_cities_geocoded: Path to japan_cities_geocoded.csv (or None to skip)
        """
        print("\n🚀 Starting Multi-Dataset Bloom Data Cleaning Pipeline\n")
        print("="*70)
        
        datasets = []
        
        # Process Japanese Cherry Blossoms (2024 forecasts)
        if cherry_forecasts and cherry_places:
            # Try to find files in different locations
            forecasts_path = self._find_file(cherry_forecasts)
            places_path = self._find_file(cherry_places)
            
            if forecasts_path and places_path:
                cherry_df = self.clean_japanese_cherry_data(forecasts_path, places_path)
                datasets.append(cherry_df)
            else:
                print(f" Skipping Japanese cherry forecast data - files not found")
        
        # Process Historical Sakura Data (1953-2025) - NEW!
        if historical_sakura and sakura_cities_geocoded:
            sakura_path = self._find_file(historical_sakura)
            cities_path = self._find_file(sakura_cities_geocoded)
            
            if sakura_path and cities_path:
                historical_df = self.clean_historical_sakura_data(sakura_path, cities_path)
                if len(historical_df) > 0:
                    datasets.append(historical_df)
            else:
                print(f" Skipping historical sakura data - files not found")
                if not sakura_path:
                    print(f"   Missing: {historical_sakura}")
                if not cities_path:
                    print(f"   Missing: {sakura_cities_geocoded}")
        
        # Process Symphyotrichum
        if symphyotrichum_data:
            symph_path = self._find_file(symphyotrichum_data)
            if symph_path:
                symph_df = self.clean_symphyotrichum_data(symph_path)
                datasets.append(symph_df)
            else:
                print(f" Skipping Symphyotrichum data - file not found")
        
        # Process Sweet Cherry (European phenology dataset)
        if sweet_cherry_data:
            sweet_path = self._find_file(sweet_cherry_data)
            if sweet_path:
                sweet_df = self.clean_sweet_cherry_data(sweet_path)
                datasets.append(sweet_df)
            else:
                print(f" Skipping Sweet Cherry data - file not found")
        
        if not datasets:
            raise ValueError("No valid datasets found! Please check file paths.")
        
        # Combine all datasets
        df_combined = self.combine_datasets(*datasets)
        
        # Add climate indicators
        df_enriched = self.add_climate_indicators(df_combined)
        
        # Generate summary stats
        self.generate_summary_stats(df_enriched)
        
        # Export for ML
        self.export_for_ml(df_enriched)
        
        # Export to GeoJSON
        self.export_to_geojson(df_enriched)
        
        print("\n Multi-dataset cleaning pipeline completed successfully!\n")
        
        return df_enriched
    
    def _find_file(self, filename):
        """Helper to find file in multiple possible locations"""
        possible_locations = [
            self.raw_data_dir / filename,
            Path('../../backend') / filename,
            Path('../backend') / filename,
            Path(filename)  # Absolute or relative path
        ]
        
        for path in possible_locations:
            if path.exists():
                return path
        
        return None


def main():
    """Main execution function"""
    cleaner = BloomDataCleaner()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                  BLOOMBLY DATA CLEANING PIPELINE                     ║
    ║                     Climate Change Bloom Analysis                    ║
    ║          Now with 73 Years of Japanese Sakura Data! 🌸              ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run the full pipeline with all datasets
    cleaned_data = cleaner.run_full_pipeline(
        cherry_forecasts='cherry_blossom_forecasts.csv',
        cherry_places='cherry_blossom_places.csv',
        historical_sakura='sakura_first_bloom_dates.csv',
        sakura_cities_geocoded='japan_cities_geocoded.csv',
        symphyotrichum_data='data.csv',
        sweet_cherry_data='Sweet_cherry_phenology_data_1978-2015.csv'
    )
    
    print("\n Output files created:")
    print("   ✓ data/processed/clean_blooms_ml.csv      (for ML training)")
    print("   ✓ data/processed/blooms.geojson           (for 3D globe visualization)")
    print("   ✓ data/processed/feature_metadata.json    (feature engineering metadata)")
    print("\n Dataset Summary:")
    print(f"   • Total observations: {len(cleaned_data):,}")
    print(f"   • Historical sakura (1953-2025): ~5,800 observations from 102 cities")
    print(f"   • Geographic coverage: Japan (primary), Europe, North America")
    print(f"   • Species tracked: {cleaned_data['scientific_name'].nunique()}")
    print("\n Next steps:")
    print("   1. Run feature engineering: python ml/src/features.py")
    print("   2. Train ML model: python ml/src/train.py")
    print("   3. Generate predictions for 2026-2030")


if __name__ == "__main__":
    main()
