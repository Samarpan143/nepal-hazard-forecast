# Chapter 4: The Data Pipeline ETL
## Sourcing and Merging the Multi-Sensor Streams

### 4.1 The ROI (Region of Interest)
We define the bounding box for all of Nepal as:
```python
NEPAL_BOUNDS = [80.05, 26.34, 88.20, 30.42]
nepal_roi = ee.Geometry.Rectangle(NEPAL_BOUNDS)
```
Every satellite query is filtered through this geometry to ensure we don't process global data, which would crash our memory limits.

### 4.2 Handling GEE Quotas (The Chunking Strategy)
**The Problem:** Google Earth Engine limits a single `getInfo()` request to **5,000 elements**.
A 3-year daily request for 5 sensors exceeds 1,000 elements per sensor.
**The Solution:**
We authored a looping mechanism in `data_pipeline.py` that separates the 3-year period into **1-year chunks**.
1. Iterate over `[2021, 2022, 2023]`.
2. Perform individual 1-year extractions.
3. Use `pd.concat()` to stitch the years back together into a final DataFrame.

### 4.3 Data Cleaning & Preprocessing
Raw satellite data is noisy. We apply two critical filters:

#### Cloud Masking (Sentinel-2)
```python
.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
```
We discard any optical image with more than 20% cloud cover to ensure our $NDVI$ values are accurate and not just pixels of white fog.

#### Linear Temporal Interpolation
Clouds still cause gaps in our daily timeline.
*   **The Problem:** Most ML models (especially LSTM) cannot handle `NaN` / empty cells.
*   **The Solution:** `df.interpolate(method='time', limit_direction='both')`.
*   **How it Works:** If Day 2 is cloudy, we estimate its value by drawing a straight line between the values of Day 1 and Day 3.

### 4.4 The CSV Cache Strategy
Fetching 3 years of satellite data takes about **5 to 10 minutes**.
To optimize our research, the pipeline saves the result to `nepal_historical_data_2021_2023.csv`.
*   **First Run:** Pull from GEE -> Save to CSV.
*   **Future Runs:** Detect CSV -> Load instantly from local drive (0 seconds).

---
*Next Chapter: [Chapter 5: Deep Learning and LSTMs](05_Deep_Learning_and_LSTMs.md)*
