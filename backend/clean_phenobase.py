import pandas as pd
import os
import json

# Read the CSV file
df = pd.read_csv('../data/csv/data.csv')

# Filter for traits containing 'flower'
df = df[df['trait'].str.contains('flower', case=False, na=False)]

# Filter for years between 1975 and 2025
df = df[(df['year'] >= 1975) & (df['year'] <= 2025)]

# Since there's no day information, we'll create entries for all seasons
# We'll duplicate each row 4 times, once for each season
seasons = ['spring', 'summer', 'fall', 'winter']
expanded_rows = []

for _, row in df.iterrows():
    for season in seasons:
        new_row = row.copy()
        new_row['season'] = season
        expanded_rows.append(new_row)

df = pd.DataFrame(expanded_rows)

# Group by family and genus
grouped = df.groupby(['family', 'genus'])

last_file = None
for (family, genus), group in grouped:
    # Create directory for family (lowercase)
    family_dir = family.lower()
    os.makedirs(os.path.join('../data/jsonl', family_dir), exist_ok=True)
    
    # Select relevant columns: year, season, latitude, longitude
    data = group[['year', 'season', 'latitude', 'longitude']]
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Skip if no valid data
    if data.empty:
        continue
    
    # Save as JSON Lines for efficient processing (genus capitalized)
    genus_capitalized = genus.capitalize()
    filename = os.path.join('../data/jsonl', family_dir, f'{genus_capitalized}.jsonl')
    with open(filename, 'w') as f:
        for _, row in data.iterrows():
            json.dump({
                'year': int(row['year']),
                'season': row['season'],
                'lat': float(row['latitude']),
                'long': float(row['longitude'])
            }, f)
            f.write('\n')
    
    last_file = filename

print(f'Last file created: {last_file}')