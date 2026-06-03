import pandas as pd
import numpy as np
import datetime
import time
import json
import requests
import sys

API_URL = "http://127.0.0.1:5002/predict_area"
TIMEOUT = 120

SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def call_api(lat, lon, date_str, retries=2):
    """Call the prediction API with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, json={
                "lat": lat, "lon": lon, "target_date": date_str
            }, timeout=TIMEOUT)
            data = resp.json()
            if data.get('status') == 'success':
                return data
            else:
                print(f"    API error: {data.get('message', 'unknown')}")
                return None
        except requests.exceptions.Timeout:
            print(f"    Timeout (attempt {attempt+1}/{retries})")
            time.sleep(2)
        except Exception as e:
            print(f"    Error: {e}")
            return None
    return None


def classify_result(api_resp, expected_hazard):
    if api_resp is None:
        return False, False, 0.0, []
    
    prob = api_resp.get('probability', 0)
    hazards = [h['type'] for h in api_resp.get('hazards', [])]
    
    # Map BIPAD categories to our system's hazard strings
    hazard_mapping = {
        'Landslide': ['Landslide / Mudslide', 'Post-Fire / Barren-Land Debris Flow', 
                      'Cascading Landslide-Flood (LDOF)'],
        'Flood': ['Flash Flood / Inundation', 'Cascading River Flood',
                  'Cascading Landslide-Flood (LDOF)'],
        'Forest Fire': ['Wildfire / Forest Fire']
    }
    
    expected_types = hazard_mapping.get(expected_hazard, [])
    detected = any(h in hazards for h in expected_types)
    elevated = prob > 0.3
    
    return detected, elevated, prob, hazards


def run_backtest():
    print("\n🔬 COMPREHENSIVE BIPAD BACKTESTING — Nepal Multi-Hazard Early Warning System")
    print(f"   Testing against geocoded BIPAD ground-truth incidents (2021–2023)")
    print(f"   Sample size per hazard type: {SAMPLE_SIZE}")
    
    # Load geocoded BIPAD data
    df = pd.read_csv('bipad_labels_with_coords.csv')
    df = df[df['latitude'].notna() & df['longitude'].notna()].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\nTotal geocoded incidents: {len(df)}")
    print(f"  Landslide: {len(df[df['Hazard']=='Landslide'])}")
    print(f"  Flood:     {len(df[df['Hazard']=='Flood'])}")
    
    all_results = {}
    
    for hazard_type in ['Landslide', 'Flood', 'Forest Fire']:
        subset = df[df['Hazard'] == hazard_type].copy()
        n_available = len(subset)
        n_sample = min(SAMPLE_SIZE, n_available)
        
        #ensure we get incidents across different months/districts
        sampled = subset.sample(n=n_sample, random_state=42)
        
        emoji_map = {'Landslide': '⛰️  LANDSLIDE', 'Flood': '🌊 FLOOD', 'Forest Fire': '🔥 FOREST FIRE'}
        print(f"\n{emoji_map.get(hazard_type, '⚠️ HAZARD')} — Testing {n_sample}/{n_available} incidents")
        
        results = []
        tested = 0
        detected = 0
        elevated = 0
        errors = 0
        
        for idx, (_, row) in enumerate(sampled.iterrows()):
            location_str = f"{row['Municipality']}, {row['District']}"
            date_str = row['date'].strftime('%Y-%m-%d')
            
            print(f"  [{idx+1}/{n_sample}] {location_str} | {date_str} | ({row['latitude']:.4f}, {row['longitude']:.4f})")
            
            api_resp = call_api(row['latitude'], row['longitude'], date_str)
            
            if api_resp is None:
                errors += 1
                results.append({
                    "location": location_str,
                    "date": date_str,
                    "lat": row['latitude'],
                    "lon": row['longitude'],
                    "status": "api_error",
                    "deaths": int(row.get('Total - People Death', 0) or 0),
                })
                continue
            
            is_detected, is_elevated, prob, hazards = classify_result(api_resp, hazard_type)
            tested += 1
            if is_detected:
                detected += 1
            if is_elevated:
                elevated += 1
            
            details = api_resp.get('details', {})
            result = {
                "location": location_str,
                "date": date_str,
                "lat": row['latitude'],
                "lon": row['longitude'],
                "probability": round(prob * 100, 1),
                "rain_14day": round(details.get('rain_14day', 0), 1),
                "slope": round(details.get('slope', 0), 1),
                "predicted_hazards": hazards,
                "exact_match": is_detected,
                "elevated_risk": is_elevated,
                "deaths": int(row.get('Total - People Death', 0) or 0),
                "status": "tested"
            }
            results.append(result)
            
            icon = "✅" if is_detected else ("⚠️" if is_elevated else "❌")
            print(f"    {icon} P={prob*100:.1f}% | Rain={details.get('rain_14day',0):.0f}mm | Slope={details.get('slope',0):.1f}° | {hazards}")
        
        detection_rate = round(detected / tested * 100, 1) if tested > 0 else 0
        elevated_rate = round(elevated / tested * 100, 1) if tested > 0 else 0
        
        all_results[hazard_type] = {
            "total_available": n_available,
            "sampled": n_sample,
            "tested": tested,
            "detected": detected,
            "detection_rate": detection_rate,
            "elevated_count": elevated,
            "elevated_rate": elevated_rate,
            "errors": errors,
            "results": results
        }
        
        print(f"\n  📊 {hazard_type.upper()}: {detected}/{tested} exact matches ({detection_rate}%)")
        print(f"     Elevated risk (>30%): {elevated}/{tested} ({elevated_rate}%)")
        if errors > 0:
            print(f"     API errors: {errors}")
    
    # Overall summary logic
    total_tested = sum(r['tested'] for r in all_results.values())
    total_detected = sum(r['detected'] for r in all_results.values())
    total_elevated = sum(r['elevated_count'] for r in all_results.values())
    total_errors = sum(r['errors'] for r in all_results.values())
    overall_rate = round(total_detected / total_tested * 100, 1) if total_tested > 0 else 0
    overall_elevated = round(total_elevated / total_tested * 100, 1) if total_tested > 0 else 0
    
    print("\n📋 OVERALL BACKTESTING SUMMARY")
    print(f"{'Hazard':<20} {'BIPAD Total':<15} {'Tested':<10} {'Detected':<12} {'Det. Rate':<12} {'Elevated':<15} {'Elev. Rate':<12}")
    for name, r in all_results.items():
        print(f"{name:<20} {r['total_available']:<15} {r['tested']:<10} {r['detected']:<12} {r['detection_rate']}%{'':<7} {r['elevated_count']}/{r['tested']}{'':<8} {r['elevated_rate']}%")
    print(f"{'OVERALL':<20} {'':<15} {total_tested:<10} {total_detected:<12} {overall_rate}%{'':<7} {total_elevated}/{total_tested}{'':<8} {overall_elevated}%")
    if total_errors > 0:
        print(f"API errors: {total_errors}")
    # False negative analysis
    print("\n🔍 FALSE NEGATIVE ANALYSIS (Missed Incidents)")
    
    missed = []
    for name, r in all_results.items():
        for res in r['results']:
            if res.get('status') == 'tested' and not res.get('exact_match', False):
                missed.append({**res, 'expected_hazard': name})
    
    if missed:
        # Analyze severity of missed events
        missed_with_deaths = [m for m in missed if m.get('deaths', 0) > 0]
        missed_no_deaths = [m for m in missed if m.get('deaths', 0) == 0]
        missed_low_rain = [m for m in missed if m.get('rain_14day', 0) < 20]
        
        print(f"\nTotal missed: {len(missed)}")
        print(f"  With casualties: {len(missed_with_deaths)} ({len(missed_with_deaths)/len(missed)*100:.0f}%)")
        print(f"  Property-only (0 deaths): {len(missed_no_deaths)} ({len(missed_no_deaths)/len(missed)*100:.0f}%)")
        print(f"  Low rainfall (<20mm): {len(missed_low_rain)} ({len(missed_low_rain)/len(missed)*100:.0f}%)")
        
        if missed_with_deaths:
            print(f"\n  ⚠️ Critical missed events (with casualties):")
            for m in sorted(missed_with_deaths, key=lambda x: -x.get('deaths', 0))[:10]:
                print(f"    💀 {m['deaths']} deaths | {m['location']} | {m['date']} | P={m['probability']}% | Rain={m['rain_14day']}mm | Slope={m['slope']}°")
    else:
        print("  No false negatives — perfect detection! 🎉")
    
    # Save results
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "sample_size_per_type": SAMPLE_SIZE,
        "overall": {
            "total_tested": total_tested,
            "total_detected": total_detected,
            "overall_detection_rate": overall_rate,
            "total_elevated": total_elevated,
            "overall_elevated_rate": overall_elevated,
            "total_errors": total_errors
        },
        "per_hazard": {}
    }
    for name, r in all_results.items():
        output["per_hazard"][name] = {
            "bipad_total": r['total_available'],
            "tested": r['tested'],
            "detected": r['detected'],
            "detection_rate": r['detection_rate'],
            "elevated_count": r['elevated_count'],
            "elevated_rate": r['elevated_rate'],
            "details": r['results']
        }
    output["false_negative_analysis"] = {
        "total_missed": len(missed),
        "with_casualties": len([m for m in missed if m.get('deaths', 0) > 0]),
        "property_only": len([m for m in missed if m.get('deaths', 0) == 0]),
        "critical_misses": [m for m in missed if m.get('deaths', 0) > 0]
    }
    
    with open('backtest_bipad_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to backtest_bipad_results.json")


if __name__ == "__main__":
    run_backtest()
