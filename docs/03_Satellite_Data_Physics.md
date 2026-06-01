# Chapter 3: Satellite Data Physics
## The Multi-Sensor Remote Sensing Paradigm

### 3.1 Sentinel-1 (C-Band SAR)
*   **The Problem:** Optical satellites (like Landsat) cannot see through the heavy monsoon clouds of Nepal.
*   **The Solution:** Active Electronic Microwave sensors.
*   **How it Works:** Synthetic Aperture Radar (SAR) pulses microwave beams toward the earth. These beams bounce back to the sensor (Backscatter).
*   **Application:** When a river overflows (Flood), the backscatter profile changes because water acts as a mirror, reflecting the radar away from the sensor.

### 3.2 Sentinel-2 (Optical Multispectral)
*   **Resolution:** 10m to 60m. Excellent for detailed mountain terrain.
*   **The NDVI Metric:** Used to measure the "Greenness" or health of vegetation.
    *   *Importance for Landslides:* Healthy root systems (B8) stabilize slopes. Sparse, dry vegetation ($NDVI < 0.4$) is a primary landslide risk factor.
    *   *Importance for Fires:* Healthy vegetation (High NIR) resists ignition. Dry, yellow vegetation (Low NIR) is a fuel source.

### 3.3 Sentinel-5 Precursor (Atmospheric Composition)
*   **The TROPOMI Sensor:** Specifically designed for air quality.
*   **The Aerosol Index (AI):** A direct measure of smoke and particulate matter in the atmosphere.
*   **Nepal Case Study:** During the record-breaking fires of April 2023, the TROPOMI AI was our primary detection variable for wildfire smoke tracking over the Kathmandu Valley.

### 3.4 MODIS (Thermal Sensing)
*   **Variable:** Land Surface Temperature (LST).
*   **The Drought-Fire Nexus:** Anomalously high LST ($> 85^{th}$ percentile) preceding the monsoon dries out forest litter, creating high-risk environments for ignition.

### 3.5 TerraClimate (Climatic Hydrology)
*   **Resolution:** 4km.
*   **Variables:** Precipitation (Rainfall), Soil Moisture, and Wind Speed.
*   **The Trigger:** While Sentinel satellites see the "current state," TerraClimate provides the "triggering variables" needed for predictive modeling.

---
*Next Chapter: [Chapter 4: The Data Pipeline ETL](04_The_Data_Pipeline_ETL.md)*
