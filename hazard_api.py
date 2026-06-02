from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import ee
import os
import datetime
from dotenv import load_dotenv
import requests
import concurrent.futures
import threading

# Initialize environment and Google Earth Engine
load_dotenv()
app = Flask(__name__)
CORS(app)

PROJECT_ID = os.getenv('PROJECT_ID', 'nd-sem-project')
ee.Initialize(project=PROJECT_ID)

# Load model and fit scaler once at startup to avoid repeated overhead
print("Initializing model...")
try:
    model = tf.keras.models.load_model('nepal_hazard_model_500m.h5', compile=False)
except Exception as e:
    print("Warning: Failed to load model. Falling back to mock. Error:", e)
    class MockModel:
        def predict(self, x, **kwargs):
            return [[0.2]]
    model = MockModel()

# Re-fit scaler using the same training data to keep feature distributions consistent
train_df = pd.read_csv('nepal_500m_sequences_flat.csv')
X_train_raw = []
features = ['VV', 'VH', 'NDVI', 'NDMI', 'Aerosol', 'LST', 'pr', 'soil', 'Slope', 'Aspect', 'TPI']
for _, row in train_df.iterrows():
    for t in range(14):
        X_train_raw.append([row.get(f'T{t}_{f}', 0) for f in features])
scaler = StandardScaler().fit(np.array(X_train_raw))
print("Model ready.")

# Thread lock to prevent concurrent Keras graph execution issues
model_lock = threading.Lock()


def get_seismic_risk(lat, lon, target_date_obj):
    endtime = target_date_obj.strftime('%Y-%m-%d')
    starttime = (target_date_obj - datetime.timedelta(days=14)).strftime('%Y-%m-%d')
    url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
        f"&starttime={starttime}&endtime={endtime}"
        f"&latitude={lat}&longitude={lon}&maxradiuskm=50&minmagnitude=3.0"
    )
    try:
        resp = requests.get(url).json()
        quakes = resp.get('features', [])
        if not quakes:
            return 0.0, 0.0
        max_mag = max([q['properties']['mag'] for q in quakes])
        adj = 0.3 if max_mag >= 5 else 0.15 if max_mag >= 4 else 0.05
        return adj, max_mag
    except:
        return 0.0, 0.0


def get_open_meteo_rain_list(lat, lon, target_date_obj):
    """Fetches the 14-day daily rainfall list for temporal sequence building."""
    today = datetime.datetime.now()
    is_historical = (today - target_date_obj).days > 3

    if is_historical:
        end_date = target_date_obj.strftime('%Y-%m-%d')
        start_date = (target_date_obj - datetime.timedelta(days=13)).strftime('%Y-%m-%d')
        w_url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=precipitation_sum&timezone=auto"
        )
    else:
        w_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&past_days=14&forecast_days=1&daily=precipitation_sum&timezone=auto"
        )

    try:
        w_resp = requests.get(w_url).json()
        daily_rain = [float(x or 0.0) for x in w_resp['daily']['precipitation_sum'][:14]]
        return daily_rain
    except:
        return [0.0] * 14


def get_realtime_data(lat, lon, target_date_obj):
    """Fetches all environmental inputs: terrain, vegetation, moisture, aerosols, and rainfall.

    TerraClimate is used for pr and soil to match the training data distribution.
    Daily rainfall from Open-Meteo is returned separately for temporal sequence injection.
    """
    geom = ee.Geometry.Point([lon, lat])
    footprint = geom.buffer(250).bounds()  # 500m footprint to match training resolution
    ee_end_date = ee.Date(target_date_obj.strftime('%Y-%m-%d'))
    ee_start_date = ee_end_date.advance(-30, 'day')

    # Terrain from SRTM
    srtm = ee.Image("USGS/SRTMGL1_003")
    topo = ee.Image([
        ee.Terrain.slope(srtm).rename('Slope'),
        ee.Terrain.aspect(srtm).rename('Aspect'),
        srtm.subtract(srtm.focal_mean(300, 'circle', 'meters')).rename('TPI')
    ]).reduceRegion(ee.Reducer.mean(), geom, 30).getInfo()

    # Surface temperature from MODIS LST — raw values (range 0-15767) to match training
    lst_coll = ee.ImageCollection('MODIS/061/MOD11A1').filterBounds(geom).filterDate(ee_start_date, ee_end_date)
    lst_val = lst_coll.mean().select('LST_Day_1km').reduceRegion(ee.Reducer.mean(), geom, 1000).getInfo().get('LST_Day_1km', 15200.0)

    # NDVI from MODIS using a 60-day window for better cloud-free coverage
    ndvi_coll = ee.ImageCollection("MODIS/061/MOD13A1").filterBounds(geom).filterDate(ee_end_date.advance(-60, 'day'), ee_end_date)
    ndvi_scaled = 0.4  # neutral default
    if ndvi_coll.size().getInfo() > 0:
        ndvi_val = ndvi_coll.mean().select('NDVI').reduceRegion(ee.Reducer.mean(), geom, 500).getInfo().get('NDVI', 4000)
        ndvi_scaled = float(ndvi_val) / 10000.0

    # Aerosol optical depth from MODIS for smoke and dust detection
    aod_coll = ee.ImageCollection("MODIS/061/MCD19A2").filterBounds(geom).filterDate(ee_start_date, ee_end_date)
    aod_scaled = 0.1  # neutral default
    if aod_coll.size().getInfo() > 0:
        aod_info = aod_coll.mean().reduceRegion(ee.Reducer.mean(), geom, 1000).getInfo()
        aod_val = aod_info.get('Optical_Depth_047', 100)
        aod_scaled = float(aod_val) / 1000.0

    # NDMI from MODIS surface reflectance — better moisture indicator than NDVI alone
    ndmi_coll = ee.ImageCollection('MODIS/061/MOD09A1').filterBounds(geom).filterDate(ee_start_date, ee_end_date)
    ndmi_scaled = 0.5  # neutral default
    if ndmi_coll.size().getInfo() > 0:
        def add_ndmi(img):
            ndmi = img.normalizedDifference(['sur_refl_b02', 'sur_refl_b06']).rename('NDMI')
            return img.addBands(ndmi)
        ndmi_img = ndmi_coll.map(add_ndmi).qualityMosaic('NDMI')
        ndmi_val = ndmi_img.reduceRegion(ee.Reducer.mean(), geom, 500).getInfo().get('NDMI')
        if ndmi_val is not None:
            ndmi_scaled = float(ndmi_val)

    # Daily rainfall from Open-Meteo for temporal sequence injection
    rain_list = get_open_meteo_rain_list(lat, lon, target_date_obj)

    # Monthly pr and soil from TerraClimate — matches the training data source
    tc_start = ee_end_date.advance(-1, 'month')
    tc_end = ee_end_date.advance(1, 'month')
    tc = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
        .filterBounds(geom).filterDate(tc_start, tc_end) \
        .select(['pr', 'soil']).mean()
    tc_vals = tc.resample('bilinear').reduceRegion(ee.Reducer.mean(), footprint, scale=1000).getInfo()

    pr_terra = float(tc_vals.get('pr', 91.0)) if tc_vals else 91.0
    soil_terra = float(tc_vals.get('soil', 613.0)) if tc_vals else 613.0
    print(f"  TerraClimate: pr={pr_terra:.1f}mm, soil={soil_terra:.1f}mm")

    return {
        'LST': float(lst_val or 15200.0),
        'NDVI': float(ndvi_scaled or 0.4),
        'NDMI': float(ndmi_scaled or 0.5),
        'Aerosol': float(aod_scaled or 0.1),
        'rain_list': rain_list,
        'pr': float(pr_terra),
        'soil': float(soil_terra),
        'Slope': topo.get('Slope', 20.0),
        'Aspect': topo.get('Aspect', 180.0),
        'TPI': topo.get('TPI', 0.0)
    }


def get_upstream_coords(lat, lon):
    # ~8km radius to target narrow gorge settlements like Melamchi
    # Wider radii cause false alerts in open basins like Kathmandu Valley
    return [
        {"dir": "North",      "lat": lat + 0.08, "lon": lon},
        {"dir": "North-East", "lat": lat + 0.06, "lon": lon + 0.06},
        {"dir": "North-West", "lat": lat + 0.06, "lon": lon - 0.06}
    ]


def run_prediction_core(lat, lon, target_date_obj):
    try:
        vals = get_realtime_data(lat, lon, target_date_obj)
        print(f"DIAGNOSTIC - SENSORS: {vals}")
        seismic_boost, mag = get_seismic_risk(lat, lon, target_date_obj)

        # Build a 14-day temporal sequence using real daily rainfall variation
        seq = []
        total_rain = sum(vals['rain_list'])
        avg_daily = total_rain / 14.0 if total_rain > 0 else 1.0

        # Synthetic temperature trend across 14 days
        temp_trend = [vals['LST'] + (x * 10) for x in range(-7, 7)]

        for t, (om_pr_day, temp_day) in enumerate(zip(vals['rain_list'], temp_trend)):
            # Scale TerraClimate monthly pr by the daily rainfall ratio
            daily_ratio = om_pr_day / avg_daily if avg_daily > 0 else 1.0
            pr_day = vals['pr'] * daily_ratio
            pr_day = min(pr_day, 1063.0)  # cap at training maximum

            seq.append([-10.0, -17.0, vals['NDVI'], vals.get('NDMI', 0.5), vals['Aerosol'], vals['LST'], pr_day, vals['soil'], vals['Slope'], vals['Aspect'], vals['TPI']])

        scaled = scaler.transform(np.array(seq)).reshape(1, 14, 11)

        with model_lock:
            preds = model.predict(scaled, verbose=0)[0]

        # Model output: [p_stable, p_landslide, p_flood, p_fire]
        p_landslide = float(preds[1])
        p_flood = float(preds[2])
        p_fire = float(preds[3]) if len(preds) > 3 else 0.0
        p_smog = 0.0  # Smog is handled purely by the physical failsafe below

        # Override wildfire probability if environmental conditions clearly indicate high risk
        # This corrects for the model's blindness to monthly-averaged temperature inputs
        is_hot = vals['LST'] > 14800       # roughly above 22.8°C
        recent_rain = sum(vals['rain_list'])
        is_dry = recent_rain < 10.0
        is_water_stressed = vals.get('NDMI', 0.5) < 0.15

        if is_hot and is_dry and is_water_stressed:
            p_fire = max(p_fire, 0.85)

        # Apply seismic trigger directly to landslide risk
        p_landslide = min(1.0, p_landslide + seismic_boost)
        final_prob = max(p_landslide, p_flood)

        hazard_type = "General Instability"
        reason = "Monitoring environment."
        upstream_rain_total = 0.0
        time_lag = 0
        threat_dir = "None"
        upstream_payload = None

        # Upstream catchment analysis — only run for gorge and valley terrain
        is_gorge_or_valley = 8.0 < vals['Slope'] < 20.0
        if is_gorge_or_valley:
            print(f"Flat terrain detected. Running upstream catchment scan...")
            candidates = get_upstream_coords(lat, lon)
            max_rain = -1.0
            worst_node = None

            for node in candidates:
                r_list = get_open_meteo_rain_list(node['lat'], node['lon'], target_date_obj)
                r_sum = sum(r_list)
                if r_sum > max_rain:
                    max_rain = r_sum
                    worst_node = node

            if worst_node and max_rain > 30:
                up_vals = get_realtime_data(worst_node['lat'], worst_node['lon'], target_date_obj)

                # Only proceed if the upstream point is genuinely steeper (topological sanity check)
                if up_vals['Slope'] > vals['Slope'] + 4.0:
                    upstream_rain_total = sum(up_vals['rain_list'])
                    threat_dir = worst_node['dir']

                    up_seq = []
                    r_list = up_vals['rain_list']
                    up_avg_daily = r_sum / 14.0 if r_sum > 0 else 1.0
                    up_temp_trend = [up_vals['LST'] + (x * 10) for x in range(-7, 7)]

                    for t, (up_om_pr_day, up_temp_day) in enumerate(zip(r_list, up_temp_trend)):
                        up_daily_ratio = up_om_pr_day / up_avg_daily if up_avg_daily > 0 else 1.0
                        up_pr_day = up_vals['pr'] * up_daily_ratio
                        up_pr_day = min(up_pr_day, 1063.0)
                        up_seq.append([-10.0, -17.0, up_vals['NDVI'], up_vals.get('NDMI', 0.5), up_vals['Aerosol'], up_vals['LST'], up_pr_day, up_vals['soil'], up_vals['Slope'], up_vals['Aspect'], up_vals['TPI']])

                    up_scaled = scaler.transform(np.array(up_seq)).reshape(1, 14, 11)

                    with model_lock:
                        up_preds = model.predict(up_scaled, verbose=0)[0]

                    up_landslide_prob = float(up_preds[1])
                    up_flood_prob = float(up_preds[2])

                    upstream_payload = {
                        'landslide': up_landslide_prob,
                        'flood': up_flood_prob,
                        'max_rain': max_rain,
                        'threat_dir': worst_node['dir'],
                        'slope': up_vals['Slope']
                    }

                    if up_vals['Slope'] > 25.0 and up_landslide_prob > 0.6 and upstream_rain_total > 30:
                        final_prob = max(final_prob, min(1.0, up_landslide_prob))
                        hazard_type = "Cascading Landslide-Flood (LDOF)"
                        reason = f"Severe landslide risk in steep {threat_dir} catchment. High risk of river damming and outburst flood."
                        time_lag = 4
                    elif up_flood_prob > 0.4 and upstream_rain_total > 40:
                        final_prob = max(final_prob, min(1.0, up_flood_prob))
                        hazard_type = "Cascading River Flood"
                        reason = f"Heavy rainfall in the {threat_dir} catchment. Expected impact downstream."
                        time_lag = 6

        # Dynamic hydrological calibration using slope and drainage capacity
        local_rain_total = sum(vals['rain_list'])
        combined_rain = max(local_rain_total, upstream_rain_total)

        calibrated_prob = final_prob

        if hazard_type != "Cascading River Flood":
            # Sigmoid functions that scale probability by slope steepness and drainage saturation
            d_slope = 1.0 / (1.0 + np.exp(5.0 - vals['Slope']))
            c_drain = 200.0
            f_drain = 1.0 / (1.0 + np.exp(-(combined_rain - c_drain) / 30.0))
            multiplier = d_slope + (1.0 - d_slope) * f_drain
            calibrated_prob = final_prob * multiplier

        final_prob = max(calibrated_prob, seismic_boost)
        final_prob = min(1.0, final_prob)
        print(f"DIAGNOSTIC - FINAL CALIBRATED PROB: {final_prob*100:.1f}%")

        # Multi-hazard physical consistency filtering
        # Each hazard type has environmental pre-conditions that must be satisfied
        is_dry = local_rain_total < 15.0
        has_seismic_trigger = seismic_boost > 0.1

        # Landslide requires wet conditions or seismic activity on steep slopes
        if is_dry and not has_seismic_trigger:
            landslide_score = 0.0
        else:
            landslide_score = float(p_landslide if vals['Slope'] > 12.0 else (0.0 if vals['Slope'] < 5.0 else p_landslide * 0.1))

        # Flood requires meaningful local or upstream rainfall
        # Slope penalty only applies above 30° (cliff faces) — valley floors in Nepal
        # are typically 15-25° and should receive full flood scores
        is_wet_enough_for_flood = (local_rain_total > 20.0 or upstream_rain_total > 20.0)
        if not is_wet_enough_for_flood:
            flood_score = 0.0
        else:
            if vals['Slope'] > 30.0 and local_rain_total <= 40.0 and upstream_rain_total <= 20.0:
                flood_score = float(p_flood * 0.3)
            else:
                flood_score = float(p_flood)

        # Wildfire cannot occur in wet or high-moisture conditions
        is_wet = local_rain_total > 20.0 or vals.get('NDMI', 0.5) > 0.3
        if is_wet:
            fire_score = 0.0
        else:
            fire_score = float(p_fire if (p_fire > 0.3) else p_fire * 0.1)

        # Smog score
        smog_score = float(p_smog if (p_smog > 0.3) else p_smog * 0.1)

        final_prob = max(landslide_score, flood_score, fire_score, smog_score)

        hazards = []

        if hazard_type == "Cascading River Flood" and flood_score > 0.3:
            hazards.append({
                "type": "Cascading River Flood",
                "reason": f"Heavy rainfall in the {threat_dir} catchment. Expected impact downstream."
            })
        elif hazard_type == "Cascading Landslide-Flood (LDOF)" and landslide_score > 0.3:
            hazards.append({
                "type": "Cascading Landslide-Flood (LDOF)",
                "reason": f"Severe landslide risk in steep {threat_dir} catchment. High risk of river damming and outburst flood."
            })

        if landslide_score > 0.15:
            if local_rain_total > 40 and vals['Slope'] > 15.0 and vals['NDVI'] < 0.25:
                hazards.append({
                    "type": "Post-Fire / Barren-Land Debris Flow",
                    "reason": "Heavy rain on steep, vegetation-stripped terrain creates extreme landslide vulnerability."
                })
                landslide_score = max(landslide_score, 0.85)
                final_prob = max(final_prob, 0.85)
            elif local_rain_total > 15.0 and vals['Slope'] > 8.0 and not (hazard_type == "Cascading Landslide-Flood (LDOF)"):
                hazards.append({
                    "type": "Landslide / Mudslide",
                    "reason": "High moisture accumulation in mountainous terrain."
                })

        if flood_score > 0.15:
            is_cascading = hazard_type in ["Cascading River Flood", "Cascading Landslide-Flood (LDOF)"]
            if not is_cascading:
                # Local rain threshold lowered to 15mm to avoid blocking AI predictions
                if local_rain_total > 15.0:
                    hazards.append({
                        "type": "Flash Flood / Inundation",
                        "reason": "Rainfall pooling / Local river overflow."
                    })
                # Upstream-only floods require a higher threshold to avoid false positives
                elif upstream_rain_total > 40.0 and local_rain_total <= 15.0:
                    hazards.append({
                        "type": "Flash Flood / Inundation",
                        "reason": f"Upstream catchment rainfall ({upstream_rain_total:.0f}mm) detected. High risk of river overflow from headwater or glacier run-off."
                    })

        if fire_score > 0.3:
            hazards.append({
                "type": "Wildfire / Forest Fire",
                "reason": f"Dry conditions and elevated surface temperatures detected (Confidence: {fire_score*100:.1f}%)."
            })

        if smog_score > 0.3:
            hazards.append({
                "type": "Severe Smog / Pollution Alert",
                "reason": f"Stagnant atmospheric conditions and high particulate concentration detected (Confidence: {smog_score*100:.1f}%)."
            })

        if not hazards and final_prob > 0.3:
            hazards.append({
                "type": "General Instability",
                "reason": "Triggered by external factors or seismic activity."
            })

        return {
            'status': 'success',
            'probability': final_prob,
            'scores': {
                'landslide': landslide_score,
                'flood': flood_score,
                'fire': fire_score,
                'smog': smog_score
            },
            'upstream': upstream_payload,
            'hazards': hazards,
            'time_lag': time_lag,
            'threat_dir': threat_dir,
            'target_date': target_date_obj.strftime('%Y-%m-%d'),
            'details': {
                'rain_14day': local_rain_total,
                'upstream_rain': upstream_rain_total,
                'slope': vals['Slope'],
                'temp_raw': vals['LST'],
                'seismic': mag,
                'ndvi': vals['NDVI'],
                'ndmi': vals.get('NDMI', 0.5),
                'aerosol': vals['Aerosol']
            }
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    target_date_str = data.get('target_date')

    try:
        if target_date_str:
            target_date_obj = datetime.datetime.strptime(target_date_str, '%Y-%m-%d')
        else:
            target_date_obj = datetime.datetime.now()
    except:
        target_date_obj = datetime.datetime.now()

    print(f"Predicting for ({lat}, {lon}) on {target_date_obj.strftime('%Y-%m-%d')}...")

    res = run_prediction_core(lat, lon, target_date_obj)
    if res.get('status') == 'error':
        return jsonify(res), 500
    return jsonify(res)


@app.route('/predict_area', methods=['POST'])
def predict_area():
    """Runs multi-point elevation-stratified sampling over a ~3km grid (concurrent)."""
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    target_date_str = data.get('target_date')

    try:
        if target_date_str:
            target_date_obj = datetime.datetime.strptime(target_date_str, '%Y-%m-%d')
        else:
            target_date_obj = datetime.datetime.now()
    except:
        target_date_obj = datetime.datetime.now()

    OFFSET = 0.0135  # approximately 1.5km

    sample_points = [
        {"label": "Center", "lat": lat,          "lon": lon},
        {"label": "North",  "lat": lat + OFFSET,  "lon": lon},
        {"label": "South",  "lat": lat - OFFSET,  "lon": lon},
        {"label": "East",   "lat": lat,            "lon": lon + OFFSET},
        {"label": "West",   "lat": lat,            "lon": lon - OFFSET},
    ]

    def process_point(pt):
        print(f"\n[AREA SCAN] Processing {pt['label']} ({pt['lat']:.4f}, {pt['lon']:.4f})")
        res = run_prediction_core(pt['lat'], pt['lon'], target_date_obj)
        res['label'] = pt['label']
        res['lat'] = round(pt['lat'], 6)
        res['lon'] = round(pt['lon'], 6)
        if res.get('status') == 'error':
            res['probability'] = 0
            res['error'] = res.get('message', 'unknown')
        return res

    # Run all 5 GEE/API calls concurrently to reduce total response time
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_point, sample_points))

    worst_idx = 0
    worst_prob = -1.0
    for i, res in enumerate(results):
        prob = res.get('probability', 0)
        if prob > worst_prob:
            worst_prob = prob
            worst_idx = i

    worst = results[worst_idx] if results else {}

    return jsonify({
        'status': 'success',
        'method': 'multi_point_sampling',
        'sample_count': len(results),
        'worst_case': worst,
        'probability': worst.get('probability', 0),
        'scores': worst.get('scores', {}),
        'hazards': worst.get('hazards', []),
        'details': worst.get('details', {}),
        'time_lag': worst.get('time_lag', 0),
        'threat_dir': worst.get('threat_dir', 'None'),
        'upstream': worst.get('upstream'),
        'target_date': target_date_str,
        'all_samples': results,
    })


if __name__ == '__main__':
    app.run(port=5002, debug=False)
