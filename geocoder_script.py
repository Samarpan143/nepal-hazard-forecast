import pandas as pd
import geemap
import time

def geocode_locations(filtered_csv):
    df = pd.read_csv(filtered_csv)
    
    # Identify unique location strings
    locs = df[['Municipality', 'District', 'Province']].drop_duplicates()
    locs['search_str'] = locs['Municipality'] + ", " + locs['District'] + ", Nepal"
    
    print(f"Geocoding {len(locs)} unique places...")
    
    results = {}
    for i, row in locs.iterrows():
        try:
            # We use a simple geocoder. Since we are on a Mac, we might have network.
            # geemap.geocode returns a list of results.
            res = geemap.geocode(row['search_str'])
            if res:
                loc = res[0]
                lat = loc.lat
                lon = loc.lng
                results[row['search_str']] = (lat, lon)
                print(f"Success: {row['search_str']} -> ({lat:.4f}, {lon:.4f})")
            else:
                # Try fallback: just District, Nepal
                fallback = f"{row['District']}, Nepal"
                res = geemap.geocode(fallback)
                if res:
                    loc = res[0]
                    lat = loc.lat
                    lon = loc.lng
                    results[row['search_str']] = (lat, lon)
                    print(f"Fallback Success: {fallback} -> ({lat:.4f}, {lon:.4f})")
                else:
                    print(f"Failed to geocode: {row['search_str']}")
            
            # Rate limiting sleep
            time.sleep(0.1)
        except Exception as e:
            print(f"Error geocoding {row['search_str']}: {e}")
            
    # Map back to the main dataframe
    df['search_str'] = df['Municipality'] + ", " + df['District'] + ", Nepal"
    df['latitude'] = df['search_str'].map(lambda x: results.get(x, (None, None))[0])
    df['longitude'] = df['search_str'].map(lambda x: results.get(x, (None, None))[1])
    
    # Drop failures
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"Successfully mapped {len(df)} incidents to coordinates.")
    
    df.to_csv('bipad_labels_with_coords.csv', index=False)

if __name__ == "__main__":
    geocode_locations('bipad_labels_filtered.csv')
