# Chapter 8: Source Code Architecture

This chapter provides a detailed technical breakdown of every Python script and module used to process geospatial data, generate target labels, train the Deep Learning models, and verify the outputs for the Nepal Multi-Hazard Prediction System. 

The source code pipeline operates sequentially. Data labeling generates targets, GEE scripts extract temporal features around those locations, and Deep Learning modules compile those parameters into hazard predictions.

---

### 1. Data Preparation & Labeling 

**`bipad_labeler.py`**
* **Purpose:** Handles raw disaster event data parsing from the BIPAD system records. 
* **Key Operations:** It loads government-provided incident CSV records, corrects encoding issues, filters explicitly for **Landslide** and **Flood** reports occurring between `2021-01-01` and `2023-12-31`, and pulls out strings defining the `Municipality`, `District`, and `Province` of the damage locus.
* **Output:** `bipad_labels_filtered.csv` (contains strings to be mapped).

**`geocoder_script.py`**
* **Purpose:** Converts categorical/text location data into explicit geospatial properties.
* **Key Operations:** It relies on the `geemap` library to connect to geocoding services. It iterates over the isolated locations provided by the labeler script and matches string queries to explicit latitude (`lat`) and longitude (`lon`) combinations safely.
* **Output:** `bipad_labels_with_coords.csv` (contains raw coordinate locations of empirical disasters).

**`generate_samples.py`**
* **Purpose:** Acts as a data-balancer. Machine Learning algorithms require examples of when disasters *did not* happen to learn boundaries.
* **Key Operations:** Imports geocoded positive hazards ("1" labels) and computationally synthesizes negative "stable" observations ("0" labels). It builds a 5:1 ratio, randomly distributing these negative points spatially across Nepal's geographical bounding box and temporally across the 3-year study period.
* **Output:** `sample_locations_raw.csv` (the foundation target coordinate map for extraction).

---

### 2. Earth Engine Distributed Feature Extraction

**`extract_high_res_samples.py`**
* **Purpose:** Complex distributed temporal extraction from the Google Earth Engine cluster.
* **Key Operations:** This script performs localized remote sensing using a 500-meter buffer radius from our target GPS samples. It aggregates Topography (SRTM Slope/Aspect), Rain (TerraClimate), LST (MODIS), Aerosol (Sentinel-5P AI), Radar (Sentinel-1 VV/VH), and Vegetation (Sentinel-2 NDVI). For every target coordinate (both positive and negative instances), it creates a continuous temporal window pulling conditions **-6 days to day 0** of the event, assembling them into a strict sequence matrix.
* **Output:** `nepal_500m_sequences_flat.csv` / `nepal_500m_sequences_2021_2023.csv` 

**`data_pipeline.py`**
* **Purpose:** A country-scale generalized Extraction-Transformation-Load (ETL) pipeline.
* **Key Operations:** Unlike the hyper-localized 500m script above, this extracts general aggregated statistics over the entire territorial bounding box of Nepal using localized reductions. It performs continuous temporal merging, handles severe missing values caused by Cloud Cover via bi-directional linear interpolation mappings, and drives pipeline triggers.
* **Output:** `nepal_historical_data_2021_2023.csv`

---

### 3. Machine Learning Architectures

**`ml_models.py`**
* **Purpose:** Core object-oriented Machine Learning module (`MultiHazardModeler` class).
* **Key Operations:** Designed primarily around the baseline aggregated datasets, it formulates **Synthetic Proxy Labels** based on physically extreme thresholds (e.g., Landslide = Extreme Rain + High Slope + Low NDVI). It acts as a structural model orchestrator, formatting scalar features for standard baseline testing via Scikit-Learn `RandomForest` and `VotingClassifier` ensembles, alongside structuring internal logic for a Sequential `LSTM` using Keras to compare architectural limitations.

**`train_bilstm_500m.py`**
* **Purpose:** The dedicated, primary Deep Learning neural network model for the localized data.
* **Key Operations:** Takes the massive high-resolution multi-variable sequence CSV arrays, applies physical adjustments via Random Forests (i.e. 'Downscaling' and "Sharpening" coarse TerraClimate soil moisture features by predicting off higher-resolution Sentinel metrics limiters). It then scales out massive arrays spanning multiple shapes, balances imbalanced BIPAD weights aggressively, and constructs a robust Keras `Bi-Directional LSTM` model to establish state-of-the-art multi-hazard correlations. 
* **Output:** Generates `confusion_matrix_500m.png` visual validation outputs, and saves the final binary model format context: `nepal_hazard_model_500m.h5`.

---

### 4. Production Verifications

**`verify_results.py`**
* **Purpose:** Post-processing validation scripts directly auditing against real world historical catastrophes.
* **Key Operations:** It checks mathematical anomalies (`Z-Scores` and Standard Deviation severity boundaries) of extracted remote-sensing environmental states against empirical datasets of some of Nepal's most devastating crises (e.g., 2021 Melamchi Flood, 2023 Wildfire Outbreak). Prints statistical anomaly hits proving the capability of the early warning system against human-confirmed catastrophes.
