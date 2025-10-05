import pandas as pd
import os
import json

def get_season(doy):
    try:
        doy = int(doy)
        if 80 <= doy <= 171:
            return 'spring'
        elif 172 <= doy <= 265:
            return 'summer'
        elif 266 <= doy <= 355:
            return 'fall'
        else:
            return 'winter'
    except ValueError:
        return 'unknown'

# Read the CSV file
df = pd.read_csv('data2.csv', encoding='latin1')

# Filter for phenophase 'flower' or 'flowerend'
df = df[df['phenophase'].isin(['flower', 'flowerend'])]

# Add season column based on DOY
df['season'] = df['DOY'].apply(get_season)

# Group by functional_group (as family) and genus
grouped = df.groupby(['functional_group', 'genus'])

last_file = None
for (family, genus), group in grouped:
    # Create directory for family
    os.makedirs(family, exist_ok=True)
    
    # Select relevant columns: year, season, lat, long
    data = group[['year', 'season', 'lat', 'long']]
    
    # Save as JSON Lines for efficient processing
    filename = f'{family}/{genus}.jsonl'
    with open(filename, 'w') as f:
        for _, row in data.iterrows():
            json.dump({
                'year': int(row['year']),
                'season': row['season'],
                'lat': float(row['lat']),
                'long': float(row['long'])
            }, f)
            f.write('\n')
    
    last_file = filename

print(f'Last file created: {last_file}')
