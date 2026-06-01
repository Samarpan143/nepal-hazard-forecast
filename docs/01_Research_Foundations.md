# Chapter 1: Research Foundations
## The Multi-Hazard Challenge in the Himalayan Frontier

### 1.1 Introduction: Why Nepal?
Nepal is one of the most disaster-prone countries in the world. Its unique geography—spanning from the tropical Terai plains (60m) to the highest peaks of the Himalayas (>8,000m)—creates extreme environmental gradients. These gradients, coupled with a dense monsoon season, make Nepal a "Hotspot" for cascading natural disasters.

### 1.2 The Research Motivation
Most traditional research siloes disasters into individual categories:
1.  **Hydrological** (Floods)
2.  **Geological** (Landslides)
3.  **Meteorological** (Wildfires)
4.  **Atmospheric** (Air Pollution)

However, in reality, these hazards are **interconnected**. A massive landslide in the Sindhupalchowk district often blocks a river, leading to a dam-burst flood (GLOF or LDFC). Similarly, a pre-monsoon drought increases Land Surface Temperature (LST), which catalyzes wildfires, which in turn spikes the Aerosol Index (AI).

### 1.3 The Core Research Question
**"Can a unified Deep Learning architecture, fed by planetary-scale satellite data, effectively generalize and predict the risk of multiple, disparate hazards simultaneously across the Nepalese landscape?"**

### 1.4 Objectives
*   **Data Fusion:** Develop a backend to merge Sentinel-1 (Radar), Sentinel-2 (Optical), Sentinel-5P (Atmospheric), and MODIS (Thermal) into a single time-series.
*   **Temporal Memory:** Evaluate whether a **Long Short-Term Memory (LSTM)** model can identify the "Cumulative Moisture" required for landslide and flood triggers.
*   **Verification:** Ground-truth the model against major recorded events (e.g., Melamchi 2021 and the 2023 Wildfire peak).

### 1.5 Target Hazards Covered
*   **Landslides:** Rainfall-induced slope failures.
*   **Flash Floods:** Rapid onset riverine and glacial lake overflows.
*   **Forest Fires:** Pre-monsoon vegetation-driven anomalies.
*   **Air Pollution:** Particulate matter and aerosol loading (PM2.5).

---
*Next Chapter: [Chapter 2: GEE and Environment Setup](02_GEE_and_Environment_Setup.md)*
