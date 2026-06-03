import ee
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import datetime
from dotenv import load_dotenv

# Initialize environment and Google Earth Engine
load_dotenv()
PROJECT_ID = os.getenv('PROJECT_ID', 'nd-sem-project')
try:
    ee.Initialize(project=PROJECT_ID)
    print(f"Earth Engine initialized: {PROJECT_ID}")
except Exception as e:
    print("GEE initialization failed. Run 'earthengine authenticate' first.")
    exit()


def get_static_topography(geom):
    """Returns Slope, Aspect, and TPI for a given geometry."""
    srtm = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(srtm)
    aspect = ee.Terrain.aspect(srtm)
    tpi = srtm.subtract(srtm.focal_mean(300, 'circle', 'meters')).rename('TPI')
    topo_img = ee.Image([slope.rename('Slope'), aspect.rename('Aspect'), tpi])
    return topo_img.reduceRegion(ee.Reducer.mean(), geom, scale=30).getInfo()


def extract_14day_sequence(lat, lon, end_date_str):
    """Extracts a 14-day feature sequence for the given coordinate and end date.

    Uses TerraClimate for pr and soil, and raw MODIS LST values to match
    the training pipeline in data_pipeline.py.
    Training ranges: pr=0-1063, soil=10-3102, LST=0-15767.
    """
    geom = ee.Geometry.Point([lon, lat])
    footprint = geom.buffer(250).bounds()
    target_date = ee.Date(end_date_str)

    topo_vals = get_static_topography(geom)

    # TerraClimate is monthly — fetch the month surrounding the target date
    # This matches how data_pipeline.py pulled IDAHO_EPSCOR/TERRACLIMATE during training
    tc_start = target_date.advance(-1, 'month')
    tc_end = target_date.advance(1, 'month')
    tc = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
        .filterBounds(geom).filterDate(tc_start, tc_end) \
        .select(['pr', 'soil']).mean()
    tc_vals = tc.reduceRegion(ee.Reducer.mean(), footprint, scale=1000).getInfo()

    pr_monthly = tc_vals.get('pr') if tc_vals else None
    soil_monthly = tc_vals.get('soil') if tc_vals else None
    if pr_monthly is None:
        pr_monthly = 118.0   # training mean fallback
    if soil_monthly is None:
        soil_monthly = 754.0  # training mean fallback

    print(f"  TerraClimate: pr={pr_monthly:.1f}mm, soil={soil_monthly:.1f}mm")

    sequence = []

    for i in range(-13, 1):
        date = target_date.advance(i, 'day')

        # Sentinel-1 SAR backscatter
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(geom).filterDate(date.advance(-10, 'day'), date.advance(1, 'day')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .mean().addBands(ee.Image(0).rename('VV')).addBands(ee.Image(0).rename('VH')) \
            .select(['VV', 'VH'])

        # Sentinel-2 NDVI — use least cloudy image in a 20-day window
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(geom).filterDate(date.advance(-20, 'day'), date.advance(1, 'day')) \
            .sort('CLOUDY_PIXEL_PERCENTAGE').first()
        ndvi = ee.Image(ee.Algorithms.If(s2, s2.normalizedDifference(['B8', 'B4']), ee.Image(0))).rename('NDVI')

        # Sentinel-5P aerosol index
        aerosol = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_AER_AI") \
            .filterBounds(geom).filterDate(date.advance(-2, 'day'), date.advance(1, 'day')) \
            .mean().addBands(ee.Image(0).rename('Aerosol')) \
            .select(['absorbing_aerosol_index']).rename('Aerosol')

        # MODIS LST — raw digital numbers, no 0.02 scaling applied
        # Training data used raw values in range 0-15767
        lst = ee.ImageCollection("MODIS/061/MOD11A1") \
            .filterBounds(geom).filterDate(date.advance(-2, 'day'), date.advance(1, 'day')) \
            .mean().addBands(ee.Image(0).rename('LST')) \
            .select(['LST_Day_1km']).rename('LST')

        # Merge bands and reduce to point (pr and soil come from TerraClimate, not here)
        combined = ee.Image([s1, ndvi, aerosol, lst]).unmask(0)
        daily_vals = combined.reduceRegion(ee.Reducer.mean(), footprint, scale=30).getInfo()

        lst_val = daily_vals.get('LST')
        if lst_val is None or lst_val == 0:
            lst_val = 14300.0  # training mean

        vv_val = daily_vals.get('VV')
        if vv_val is None or vv_val == 0:
            vv_val = -10.0  # training mean

        vh_val = daily_vals.get('VH')
        if vh_val is None or vh_val == 0:
            vh_val = -17.0

        ndvi_val = daily_vals.get('NDVI')
        if ndvi_val is None:
            ndvi_val = 0.3

        aerosol_val = daily_vals.get('Aerosol')
        if aerosol_val is None:
            aerosol_val = -0.5

        day_data = {
            'VV':     vv_val,
            'VH':     vh_val,
            'NDVI':   ndvi_val,
            'Aerosol': aerosol_val,
            'LST':    lst_val,
            'pr':     pr_monthly,    # TerraClimate — matches training
            'soil':   soil_monthly,  # TerraClimate — matches training
            'Slope':  topo_vals.get('Slope', 17.0),
            'Aspect': topo_vals.get('Aspect', 170.0),
            'TPI':    topo_vals.get('TPI', 0.3)
        }

        if i == 0:
            print(f"  Day 0 audit: VV={day_data['VV']:.1f}, LST={day_data['LST']:.0f}, pr={day_data['pr']:.1f}, soil={day_data['soil']:.1f}, Slope={day_data['Slope']:.1f}")

        sequence.append([day_data[f] for f in ['VV', 'VH', 'NDVI', 'Aerosol', 'LST', 'pr', 'soil', 'Slope', 'Aspect', 'TPI']])

    return np.array(sequence)


def run_prediction(lat, lon, location_name, model, scaler, silent=False, target_date_str=None):
    if target_date_str is None:
        test_date = (datetime.datetime.now() - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
    else:
        test_date = target_date_str

    if not silent:
        print(f"Analyzing {location_name} for {test_date}...")

    try:
        raw_seq = extract_14day_sequence(lat, lon, test_date)
        scaled_seq = scaler.transform(raw_seq)
        # Reshape to (1 sample, 14 timesteps, 10 features)
        input_tensor = scaled_seq.reshape(1, 14, 10)
        preds = model.predict(input_tensor, verbose=0)[0]
        # preds: [p_stable, p_landslide, p_flood, p_fire]
        p_landslide = float(preds[1])
        p_flood = float(preds[2])
        p_fire = float(preds[3]) if len(preds) > 3 else 0.0
        
        max_prob = max(p_landslide, p_flood, p_fire)

        if not silent:
            print(f"Results for {location_name}:")
            print(f"Landslide Probability: {p_landslide:.2%}")
            print(f"Flood Probability:     {p_flood:.2%}")
            print(f"Wildfire Probability:  {p_fire:.2%}")
            if max_prob > 0.5:
                print("ALERT: High Risk")
            else:
                print("LOW RISK")

        return max_prob
    except Exception as e:
        if not silent:
            print(f"Prediction failed for {location_name}: {e}")
        return 0.0


def check_cascading_risk(valley_lat, valley_lon, hill_lat, hill_lon, model, scaler, date_str):
    """Evaluates whether upstream landslide risk will amplify downstream flood risk."""
    print(f"\nCASCADING RISK ANALYSIS — {date_str}")

    # Step 1: upstream hillside — potential debris source
    print("Step 1: Analyzing upstream hillside for landslide risk...")
    hill_prob = run_prediction(hill_lat, hill_lon, "Upstream Hillside", model, scaler, silent=True, target_date_str=date_str)

    # Step 2: downstream valley — primary impact zone
    print("Step 2: Analyzing downstream valley for flood risk...")
    valley_prob = run_prediction(valley_lat, valley_lon, "Downstream Valley", model, scaler, silent=True, target_date_str=date_str)

    # Step 3: apply cascade propagation rule
    final_valley_risk = valley_prob
    if hill_prob > 0.5:
        cascade_bonus = hill_prob * 0.5
        final_valley_risk = min(1.0, valley_prob + cascade_bonus)

    print(f"\nCascade Report:")
    print(f"Hillside (source) risk:     {hill_prob:.2%}")
    print(f"Valley (direct) risk:       {valley_prob:.2%}")
    print(f"Final propagated risk:      {final_valley_risk:.2%}")

    if final_valley_risk > valley_prob:
        print("\nCASCADING EFFECT DETECTED")
        print("Upstream landslide may block the river and amplify downstream flooding.")
    else:
        print("\nNo significant cascading propagation detected.")


if __name__ == "__main__":
    print("Loading Bi-LSTM model and scaler...")
    model_path = 'nepal_hazard_model_500m.h5'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first.")
        exit()

    model = tf.keras.models.load_model(model_path)

    # Re-fit scaler on the same training data for consistency (14-step architecture)
    train_df = pd.read_csv('nepal_500m_sequences_flat.csv')
    X_train_raw = []
    base_features = ['VV', 'VH', 'NDVI', 'Aerosol', 'LST', 'pr', 'soil', 'Slope', 'Aspect', 'TPI']
    for _, row in train_df.iterrows():
        for t in range(14):
            X_train_raw.append([row.get(f'T{t}_{f}', 0) for f in base_features])

    scaler = StandardScaler().fit(np.array(X_train_raw))
    print("System ready.")

    # Historical validation: Rasuwa landslide, August 2024
    RASUWA_DATE = "2024-08-16"
    run_prediction(28.16, 85.25, "Thulo Haku (Rasuwa)", model, scaler, target_date_str=RASUWA_DATE)

    # Historical validation: Melamchi flood disaster, June 2021
    MELAMCHI_DATE = "2021-06-15"
    run_prediction(27.83, 85.55, "Melamchi Bazar", model, scaler, target_date_str=MELAMCHI_DATE)
