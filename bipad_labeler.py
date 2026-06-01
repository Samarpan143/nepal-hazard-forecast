import pandas as pd
import numpy as np
from datetime import datetime

class BIPADProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        
    def load_and_filter(self):
        # Read with handling for possible encoding issues
        try:
            self.df = pd.read_csv(self.csv_path)
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.csv_path, encoding='latin1')
            
        print(f"Loaded {len(self.df)} total incidents.")
        
        # Filter for Landslides, Floods, and Fires
        hazards = ['Landslide', 'Flood', 'Fire', 'Forest Fire']
        self.df = self.df[self.df['Hazard'].isin(hazards)]
        
        # Clean dates
        self.df['date'] = pd.to_datetime(self.df['Incident on'])
        
        # Limit to 2021-2023
        self.df = self.df[(self.df['date'] >= '2021-01-01') & (self.df['date'] <= '2023-12-31')]
        
        print(f"Filtered to {len(self.df)} Landslide/Flood incidents between 2021-2023.")
        return self.df

    def get_location_strings(self):
        """Returns a list of unique 'Municipality, District, Province, Nepal' strings for geocoding."""
        if self.df is None:
            return []
        
        locs = self.df[['Municipality', 'District', 'Province']].drop_duplicates()
        loc_strings = []
        for _, row in locs.iterrows():
            loc_strings.append(f"{row['Municipality']}, {row['District']}, {row['Province']}, Nepal")
        return loc_strings

if __name__ == "__main__":
    processor = BIPADProcessor('bipad_data/incidents-1775018065001.csv')
    df = processor.load_and_filter()
    locs = processor.get_location_strings()
    print(f"Found {len(locs)} unique locations to geocode.")
    # Save a temporary file for the next step (geocoding)
    df.to_csv('bipad_labels_filtered.csv', index=False)
