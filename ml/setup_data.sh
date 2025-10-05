#!/bin/bash

# Setup script for Bloombly data cleaning
# Moves CSV files from Downloads to data/raw folder

echo "🌸 Bloombly Data Setup Script"
echo "==============================="
echo ""

# Create directories if they don't exist
mkdir -p ../data/raw
mkdir -p ../data/processed

echo " Created data directories"
echo ""

# Check for files in Downloads
DOWNLOADS_DIR="$HOME/Downloads"

if [ -f "$DOWNLOADS_DIR/cherry_blossom_forecasts.csv" ]; then
    echo "✓ Found cherry_blossom_forecasts.csv"
    cp "$DOWNLOADS_DIR/cherry_blossom_forecasts.csv" ../data/raw/
    echo "  → Copied to data/raw/"
else
    echo "✗ cherry_blossom_forecasts.csv not found in Downloads"
fi

if [ -f "$DOWNLOADS_DIR/cherry_blossom_places.csv" ]; then
    echo "✓ Found cherry_blossom_places.csv"
    cp "$DOWNLOADS_DIR/cherry_blossom_places.csv" ../data/raw/
    echo "  → Copied to data/raw/"
else
    echo "✗ cherry_blossom_places.csv not found in Downloads"
fi

if [ -f "../backend/data.csv" ]; then
    echo "✓ Found data.csv in backend folder"
    cp "../backend/data.csv" ../data/raw/
    echo "  → Copied to data/raw/"
else
    echo "✗ data.csv not found in backend folder"
fi

echo ""
echo " Files in data/raw/:"
ls -lh ../data/raw/*.csv 2>/dev/null || echo "  (no CSV files found)"

echo ""
echo " Setup complete!"
echo ""
echo "Next step: Run the cleaning script"
echo "  cd src"
echo "  python clean_data.py"
