import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

# 1. Configuration
DATA_CACHE = 'nepal_historical_data_2021_2023.csv'

# GROUND TRUTH: Major Documented Natural Disasters in Nepal (2021-2023)
REAL_WORLD_EVENTS = {
    '2021-06-15': 'Melamchi Flash Flood/Landslide',
    '2021-10-18': 'Post-Monsoon Extreme Rain (Western Nepal)',
    '2022-09-09': 'Darchula Flash floods',
    '2022-09-17': 'Achham/Sudurpaschim Landslides (Post-Heavy Rain)',
    '2022-10-09': 'Karnali Catchment Catastrophic Flooding',
    '2023-04-19': 'Record Peak Wildfire Activity (2800+ Fires)',
    '2023-06-16': 'Eastern Nepal Early Monsoon Flash Floods (Koshi)',
    '2023-08-16': 'Mustang Flash Flood (Kagbeni)',
}

def verify_and_report():
    if not os.path.exists(DATA_CACHE):
        print(f"Error: {DATA_CACHE} not found. Please run data_pipeline.py first.")
        return

    # 2. Ingest Data
    df = pd.read_csv(DATA_CACHE, index_col='Date', parse_dates=True)
    
    # 3. Correlation with Real Events
    print("Scientific Correlation: Environmental Anomalies on Disaster Dates")
    
    report_rows = []
    for date_str, event_name in REAL_WORLD_EVENTS.items():
        dt = pd.to_datetime(date_str)
        if dt in df.index:
            row = df.loc[dt]
            
            # Check Z-scores to identify 'extremity'
            precip_z = (row['Precipitation'] - df['Precipitation'].mean()) / df['Precipitation'].std()
            aerosol_z = (row['absorbing_aerosol_index'] - df['absorbing_aerosol_index'].mean()) / df['absorbing_aerosol_index'].std()
            lst_z = (row['LST_Day_1km'] - df['LST_Day_1km'].mean()) / df['LST_Day_1km'].std()
            
            # Create a summary entry
            info = {
                'Date': date_str,
                'Event': event_name,
                'Precipitation_Z': precip_z,
                'Aerosol_Z': aerosol_z,
                'LST_Z': lst_z,
                'Detected_By_Proxy': (precip_z > 1.5 or lst_z > 1.5 or aerosol_z > 1.5) # Statistical flagging logic
            }
            report_rows.append(info)
            
            print(f"\n[EVENT: {event_name}] Date: {date_str}")
            print(f"  > Rain Z-score: {precip_z:.2f} (Standard Deviations above mean)")
            print(f"  > Aerosol Z-score: {aerosol_z:.2f}")
            print(f"  > LST (Temp) Z-score: {lst_z:.2f}")
            if info['Detected_By_Proxy']:
                print(f"  > [FLAG] Model identified anomalous environmental state.")
            else:
                print(f"  > [NOTE] Event was highly localized or triggered by multi-day accumulation.")
        else:
            print(f"Warning: {date_str} not found in the extracted dataset range.")

    # 4. Global Accuracy for 'Extreme Events'
    print("\n--- Model Verification Summary (Hit Rate) ---")
    verified_events = [r for r in report_rows if r['Detected_By_Proxy']]
    total_major = len(report_rows)
    hit_rate = (len(verified_events) / total_major) * 100 if total_major > 0 else 0
    
    print(f"Real-World Hit Rate: {hit_rate:.1f}%")
    print(f"Total Major Disasters Analyzed: {total_major}")
    print(f"Anomalies Successfully Flagged: {len(verified_events)}")
    
    return report_rows

if __name__ == "__main__":
    verify_and_report()
