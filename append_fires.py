import pandas as pd
from geopy.geocoders import Nominatim
import time

def append_fires():
    # Read raw BIPAD
    df_raw = pd.read_csv('bipad_data/incidents-1775018065001.csv')
    fires = df_raw[df_raw['Hazard'] == 'Forest Fire'].copy()
    
    # Format same as bipad_labels_filtered.csv
    fires['date'] = pd.to_datetime(fires['Incident on']).dt.strftime('%Y-%m-%d')
    fires['search_str'] = fires['Municipality'] + ", " + fires['District'] + ", Nepal"
    
    print(f"Geocoding {len(fires)} Forest Fire incidents...")
    
    results = {}
    unique_searches = fires['search_str'].unique()
    geolocator = Nominatim(user_agent="nepal_hazard_prediction")
    for search_str in unique_searches:
        try:
            loc = geolocator.geocode(search_str)
            if loc:
                results[search_str] = (loc.latitude, loc.longitude)
                print(f"Success: {search_str} -> ({loc.latitude:.4f}, {loc.longitude:.4f})")
            else:
                fallback = search_str.split(',')[1].strip() + ", Nepal"
                loc = geolocator.geocode(fallback)
                if loc:
                    results[search_str] = (loc.latitude, loc.longitude)
                    print(f"Fallback: {fallback} -> ({loc.latitude:.4f}, {loc.longitude:.4f})")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            
    fires['latitude'] = fires['search_str'].map(lambda x: results.get(x, (None, None))[0])
    fires['longitude'] = fires['search_str'].map(lambda x: results.get(x, (None, None))[1])
    fires = fires.dropna(subset=['latitude', 'longitude'])
    
    print(f"Successfully geocoded {len(fires)} forest fires.")
    
    # Load existing and append
    df_existing = pd.read_csv('bipad_labels_with_coords.csv')
    df_combined = pd.concat([df_existing, fires], ignore_index=True)
    df_combined.to_csv('bipad_labels_with_coords.csv', index=False)
    print("Saved to bipad_labels_with_coords.csv")

if __name__ == '__main__':
    append_fires()
