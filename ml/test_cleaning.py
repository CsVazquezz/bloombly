#!/usr/bin/env python3
"""
Quick test script to verify data cleaning pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from clean_data import BloomDataCleaner

def test_cleaning_pipeline():
    """Test the data cleaning pipeline with your data"""
    print("🧪 Testing Bloom Data Cleaning Pipeline\n")
    
    try:
        # Initialize cleaner
        cleaner = BloomDataCleaner(
            raw_data_dir='../data/raw',
            processed_data_dir='../data/processed'
        )
        
        # Check if data.csv exists in backend folder
        backend_csv = Path('../backend/data.csv')
        if backend_csv.exists():
            print(f"✅ Found data.csv in backend folder")
            
            # Run the full pipeline
            cleaned_data = cleaner.run_full_pipeline(csv_file='data.csv')
            
            print("\n✅ Test completed successfully!")
            print(f"   Cleaned {len(cleaned_data)} observations")
            print(f"   Covering {cleaned_data['scientific_name'].nunique()} species")
            
            return True
        else:
            print(f"❌ Could not find data.csv in ../backend/")
            print(f"   Please ensure your CSV is in the correct location")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cleaning_pipeline()
    sys.exit(0 if success else 1)
