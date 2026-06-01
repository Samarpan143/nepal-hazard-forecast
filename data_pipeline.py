import ee
import geemap
import pandas as pd
import datetime
from ml_models import MultiHazardModeler
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Earth Engine Server
PROJECT_ID = os.getenv('PROJECT_ID', 'nd-sem-project') # Change this to your active GCP Project ID if needed

def initialize_gee():
    try:
        # Authorized using the approved user local credentials
        ee.Initialize(project=PROJECT_ID)
        print(f"Earth Engine successfully initialized with project: {PROJECT_ID}")
    except Exception as e:
        print(f"Failed to initialize Earth Engine: {e}")
        print("Please run `earthengine authenticate` in your terminal first.")
        raise e

initialize_gee()

# 2. Define ROI (Nepal Bounding Box)
# Nepal approximate bounding box coordinates: [min_lon, min_lat, max_lon, max_lat]
NEPAL_BOUNDS = [80.05, 26.34, 88.20, 30.42]
nepal_roi = ee.Geometry.Rectangle(NEPAL_BOUNDS)

# 3. Data Extractors configured for thesis variables
def get_sentinel1_data(start_date, end_date):
    """
    Sentinel-1 SAR GRD. 
    Useful for Flood and Landslide morphological prediction.
    """
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(nepal_roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .select(['VV'])
    return s1

def get_sentinel2_ndvi(start_date, end_date):
    """
    Sentinel-2 Surface Reflectance.
    Calculates NDVI for Vegetation Health (Wildfire / Landslide susceptibility).
    """
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(nepal_roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI').copyProperties(img, ['system:time_start']))
    return s2

def get_modis_lst(start_date, end_date):
    """
    MODIS LST (Land Surface Temperature).
    Used as representation for Sentinel-3/LST thermal requirements.
    """
    modis_lst = ee.ImageCollection('MODIS/061/MOD11A1') \
        .filterBounds(nepal_roi) \
        .filterDate(start_date, end_date) \
        .select(['LST_Day_1km'])
    return modis_lst

def get_sentinel5p_aod(start_date, end_date):
    """
    Sentinel-5P Aerosol Index.
    Used for Air Pollution risk / Proxy for PM2.5 and atmospheric monitoring.
    """
    s5p = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI') \
        .filterBounds(nepal_roi) \
        .filterDate(start_date, end_date) \
        .select(['absorbing_aerosol_index'])
    return s5p
    
def get_terraclimate_vars(start_date, end_date):
    """
    TerraClimate for Precipitation (Rainfall), Soil Moisture, and Wind Speed.
    """
    tc = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
        .filterBounds(nepal_roi) \
        .filterDate(start_date, end_date) \
        .select(['pr', 'soil', 'vs']) \
        .map(lambda img: img.rename(['Precipitation', 'Soil_Moisture', 'Wind_Speed']).copyProperties(img, ['system:time_start']))
    return tc

def extract_time_series(collection, band_name, scale=10000):
    """
    Reduces a GEE image collection over the Nepal ROI to a Pandas DataFrame time-series.
    Scale is generalized locally to 10km scale to comply with GEE User Memory Limits across country-wide aggregations.
    """
    def reduce_image(img):
        mean_dict = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=nepal_roi,
            scale=scale,
            maxPixels=1e9
        )
        return ee.Feature(None, mean_dict).set('system:time_start', img.get('system:time_start'))

    reduced_features = collection.map(reduce_image)
    
    # Export to list and format as Pandas DataFrame
    feature_list = reduced_features.getInfo()['features']
    
    data = []
    for f in feature_list:
        props = f['properties']
        if 'system:time_start' in props:
            # Convert milliseconds to datetime string
            date = datetime.datetime.fromtimestamp(props['system:time_start']/1000.0).strftime('%Y-%m-%d')
            val = props.get(band_name, None)
            data.append({'Date': date, band_name: val})
            
    df = pd.DataFrame(data)
    if not df.empty:
        # Aggregate by Date in case of multiple swaths per day
        df = df.groupby('Date').mean().reset_index()
    return df

def build_merged_dataset(start_date, end_date):
    """
    Builds the complete multi-source merged dataset across our variables.
    """
    print(f"Fetching data from {start_date} to {end_date}...")
    
    # 1. Fetch Collections
    s1_coll = get_sentinel1_data(start_date, end_date)
    s2_coll = get_sentinel2_ndvi(start_date, end_date)
    modis_coll = get_modis_lst(start_date, end_date)
    s5p_coll = get_sentinel5p_aod(start_date, end_date)
    climate_coll = get_terraclimate_vars(start_date, end_date)
    
    # 2. Extract to Pandas DFs safely
    print("Extracting Sentinel-1 (VV)...")
    df_s1 = extract_time_series(s1_coll, 'VV')
    
    print("Extracting Sentinel-2 (NDVI)...")
    df_s2 = extract_time_series(s2_coll, 'NDVI')
    
    print("Extracting MODIS (LST)...")
    df_lst = extract_time_series(modis_coll, 'LST_Day_1km')
    
    print("Extracting Sentinel-5P (Aerosol Index)...")
    df_s5p = extract_time_series(s5p_coll, 'absorbing_aerosol_index')
    
    print("Extracting TerraClimate (Precipitation, Soil Moisture, Wind Speed)...")
    df_climate_pr = extract_time_series(climate_coll, 'Precipitation')
    df_climate_soil = extract_time_series(climate_coll, 'Soil_Moisture')
    df_climate_wind = extract_time_series(climate_coll, 'Wind_Speed')

    # 3. Merge Logic into a continuous DateRange
    date_rng = pd.date_range(start=start_date, end=end_date+'-01', freq='D')
    # Use end_date directly for a simpler date bounds
    date_rng = pd.date_range(start=start_date, end=datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1), freq='D')
    merged_df = pd.DataFrame({'Date': [d.strftime('%Y-%m-%d') for d in date_rng]})
    
    # Iterative Left Merge
    dfs_to_merge = [df_s1, df_s2, df_lst, df_s5p, df_climate_pr, df_climate_soil, df_climate_wind]
    
    for df in dfs_to_merge:
        if not df.empty and 'Date' in df.columns:
            merged_df = pd.merge(merged_df, df, on='Date', how='left')

    # 4. Handle Missing Values / Cloud Cover Logic
    print("Applying interpolation scheme for cloud cover and temporal mismatches...")
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df = merged_df.set_index('Date')
    
    # Linear Time Interpolation allows for continuous timeseries estimates filling in missing cloud days
    merged_df = merged_df.interpolate(method='time', limit_direction='both')
    
    return merged_df

if __name__ == "__main__":
    
    # Phase 3: 3-Year Historical Fetch for Deep Learning Generalization
    START = '2021-01-01'
    END = '2023-12-31'
    DATA_CACHE = 'nepal_historical_data_2021_2023.csv'
    
    import os
    if os.path.exists(DATA_CACHE):
        print(f"--- Loading Cached ML Pipeline Training Data ({DATA_CACHE}) ---")
        # Read the preserved dataframe and ensure Date index is loaded properly
        final_df = pd.read_csv(DATA_CACHE, index_col='Date', parse_dates=True)
    else:
        print(f"--- Starting Massive ML Pipeline Fetch (Historical: {START} to {END}) ---")
        # Chunking extraction year-by-year to bypass GEE 5000-element getInfo() payload limits
        years = [2021, 2022, 2023]
        yearly_dfs = []
        for year in years:
            print(f"\n--- Initiating Fetch for Year: {year} ---")
            y_start = f"{year}-01-01"
            y_end = f"{year}-12-31"
            df_y = build_merged_dataset(y_start, y_end)
            yearly_dfs.append(df_y)
            
        final_df = pd.concat(yearly_dfs)
        # Clean potential duplicate overlapping boundary dates
        final_df = final_df[~final_df.index.duplicated(keep='first')].sort_index()
        
        # Preserve the dataset locally to save GEE credits during iterative model tuning
        final_df.to_csv(DATA_CACHE)
        print(f"\n[SUCCESS] Unified Multi-Hazard Data Pipeline Generated & Saved locally to {DATA_CACHE}.")
    
    # Trigger Phase 2: Machine Learning Execution
    print("\n--- Initializing Deep Learning Architectures ---")
    modeler = MultiHazardModeler(dataframe=final_df)
    
    # 1. Generate Ground Truth Proxy Labels 
    modeler.create_synthetic_targets()
    
    # 2. Sequential Train/Test Split
    modeler.prepare_data(target_col='Any_Hazard', test_size=0.20)
    
    # 3. Baseline & Ensemble Supervised Training
    modeler.train_baselines()
    
    # 4. Keras Deep Learning Training
    modeler.build_and_train_lstm()
    
    # 5. Full Evaluation Metrics
    modeler.evaluate_models()
    
    print(f"\n[SUCCESS] 3-Year End-to-End Prediction System Validated! Dataset stored safely at: {DATA_CACHE}")
