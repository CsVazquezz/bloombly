import pandas as pd
import os
import json

# Read the CSV file
df = pd.read_csv('../data/csv/clean_blooms_ml.csv')

# Filter for years between 1975 and 2025
df = df[(df['year'] >= 1975) & (df['year'] <= 2025)]

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
    
    # Check if file already exists and append, otherwise create new
    mode = 'a' if os.path.exists(filename) else 'w'
    
    with open(filename, mode) as f:
        for _, row in data.iterrows():
            json.dump({
                'year': int(row['year']),
                'season': row['season'].capitalize(),  # Capitalize season to match format
                'lat': float(row['latitude']),
                'long': float(row['longitude'])
            }, f)
            f.write('\n')
    
    last_file = filename

print(f'Last file created: {last_file}')
print(f'Total unique family/genus combinations processed: {len(grouped)}')
