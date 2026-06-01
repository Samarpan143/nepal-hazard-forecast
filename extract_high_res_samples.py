import ee
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize GEE
PROJECT_ID = os.getenv('PROJECT_ID', 'nd-sem-project')
ee.Initialize(project=PROJECT_ID)

def get_static_topography():
    srtm = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(srtm)
    aspect = ee.Terrain.aspect(srtm)
    tpi = srtm.subtract(srtm.focal_mean(300, 'circle', 'meters')).rename('TPI')
    return ee.Image([slope.rename('Slope'), aspect.rename('Aspect'), tpi])

def extract_vectorized(points_df, batch_size=50):
    topo = get_static_topography()
    total = len(points_df)
    print(f"Starting FINAL Robust Vectorized Extraction for {total} samples...")

    all_data = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_df = points_df.iloc[start:end]
        print(f"Processing Batch {start//batch_size + 1}/{total//batch_size + 1}...")

        features_list = []
        for idx, row in batch_df.iterrows():
            f = ee.Feature(ee.Geometry.Point([row['longitude'], row['latitude']]), {
                'date_str': row['date_str'],
                'sample_id': int(idx),
                'label': int(row['label'])
            })
            features_list.append(f)
        
        batch_fc = ee.FeatureCollection(features_list)

        def extract_sequence(feature):
            target_date = ee.Date(feature.get('date_str'))
            geom = feature.geometry()
            
            def get_daily(d_offset):
                d = target_date.advance(d_offset, 'day')
                
                # S1 padding
                s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
                    .filterBounds(geom).filterDate(d.advance(-10, 'day'), d.advance(1, 'day')) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                    .mean().addBands(ee.Image(0).rename('VV')).addBands(ee.Image(0).rename('VH')) \
                    .select(['VV', 'VH'])
                
                # S2 padding
                s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                    .filterBounds(geom).filterDate(d.advance(-20, 'day'), d.advance(1, 'day')) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE').first()
                # Use a dummy NDVI if S2 is missing
                ndvi_img = ee.Image(ee.Algorithms.If(s2, s2.normalizedDifference(['B8', 'B4']), ee.Image(0))).rename('NDVI')
                
                # NDMI (Moisture) using MOD09A1 8-day Surface Reflectance
                ndmi = ee.ImageCollection("MODIS/061/MOD09A1") \
                    .filterBounds(geom).filterDate(d.advance(-20, 'day'), d.advance(1, 'day')) \
                    .sort('system:time_start', False).first()
                ndmi_img = ee.Image(ee.Algorithms.If(ndmi, ndmi.normalizedDifference(['sur_refl_b02', 'sur_refl_b06']), ee.Image(0.5))).rename('NDMI')
                
                # Aerosol padding
                aerosol = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_AER_AI") \
                    .filterBounds(geom).filterDate(d.advance(-2, 'day'), d.advance(1, 'day')) \
                    .mean().addBands(ee.Image(0).rename('absorbing_aerosol_index')) \
                    .select('absorbing_aerosol_index').rename('Aerosol')
                
                # LST padding
                lst = ee.ImageCollection("MODIS/061/MOD11A1") \
                    .filterBounds(geom).filterDate(d.advance(-2, 'day'), d.advance(1, 'day')) \
                    .mean().addBands(ee.Image(0).rename('LST_Day_1km')) \
                    .select('LST_Day_1km').rename('LST')
                
                # Terra padding
                terra = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
                    .filterBounds(geom).filterDate(d.advance(-45, 'day'), d.advance(1, 'day')) \
                    .sort('time', False).first()
                terra_img = ee.Image(ee.Algorithms.If(terra, terra.select(['pr', 'soil']), ee.Image([0, 0]).rename(['pr', 'soil'])))
                
                # 1. Apply bilinear smoothing ONLY to low-resolution data (upsampling 4km -> 30m)
                terra_img = terra_img.resample('bilinear')
                lst = lst.resample('bilinear')
                aerosol = aerosol.resample('bilinear')
                ndmi_img = ndmi_img.resample('bilinear')
                
                # 2. Stack them without global resampling to preserve high-resolution data (s1, s2, topo)
                img = ee.Image([s1, ndvi_img, ndmi_img, aerosol, lst, terra_img, topo]).unmask(0)
                
                # 3. Create a 500m boundary (footprint) instead of a pure Point
                footprint = geom.buffer(250).bounds()
                
                # 4. Use mean() over the footprint at 30m scale to mathematically average 
                # all high-resolution pixels inside the 500m area.
                val = img.reduceRegion(ee.Reducer.mean(), footprint, scale=30)
                prefix = ee.String('T').cat(ee.Number(d_offset).add(13).toInt().format())
                
                keys = val.keys()
                def rename_keys(k): return prefix.cat('_').cat(ee.String(k))
                new_keys = keys.map(rename_keys)
                return ee.Feature(None, ee.Dictionary.fromLists(new_keys, val.values()))

            offsets = ee.List.sequence(-13, 0)
            daily_features = offsets.map(get_daily)
            
            def combine_props(f, acc):
                return ee.Feature(acc).copyProperties(f)
            
            return ee.Feature(daily_features.iterate(combine_props, feature))

        results_fc = batch_fc.map(extract_sequence)
        
        try:
            results = results_fc.getInfo()['features']
            for r in results:
                all_data.append(r['properties'])
        except Exception as e:
            print(f"Batch failed: {e}")

        pd.DataFrame(all_data).to_csv('nepal_500m_sequences_flat.csv', index=False)

    print(f"Extraction complete. Total records: {len(all_data)}")
    pd.DataFrame(all_data).to_csv('nepal_500m_sequences_2021_2023.csv', index=False)

if __name__ == "__main__":
    points_df = pd.read_csv('sample_locations_raw.csv')
    extract_vectorized(points_df)
