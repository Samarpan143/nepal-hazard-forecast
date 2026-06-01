# Chapter 7: Research Ethics and Future Work
## Scaling for the Future

### 7.1 The Spatial-Temporal Trade-off
For a BSc CSIT technical research project, we have prioritized **Temporal Memory** (LSTM).
*   **The Next Step:** **Spatial Resolution.**
*   **The Goal:** To move from 10km grid-averaging to 100m-pixel level monitoring.
*   **The Tech:** This requires **Convolutional Neural Networks (CNNs)** fused with GEE's planet-scale Tile Layers.

### 7.2 Scaling with the BIPAD Portal
Our research used "Synthetic Proxy Labels." To bring this to the next level (Master-level research or Government-level), we would manually scrape the [Nepal BIPAD Disaster Portal](https://bipadportal.gov.np/).
*   **Method:** Intersecting BIPAD GPS points with GEE Sentinel-1 backscatter.
*   **Impact:** This would create the first "Ground-Truthed" multi-hazard LSTM for all of Nepal.

### 7.3 High-Resolution Topography
Currently, we only look at environmental variables.
*   **The Missing Link:** **Slope and Aspect.**
*   **The Data:** **NASA SRTM DEM (Digital Elevation Model).**
*   **Integration:** Fusing the SLOPE degree into the LSTM features would significantly improve landslide prediction scores.

### 7.4 Ethical Considerations for Early Warning
When building AI for disasters, we must consider **Alert Fatigue**.
*   **The Problem:** If we set the Z-score threshold too low ($Z < 1.0$), we will have many False Positives.
*   **The Implication:** People will stop trusting the system.
*   **Conclusion:** We must prioritize **High Precision (Few False Alarms)** over **High Recall (Catching Everything)** for real-world trust.

### 7.5 Final Research Acknowledgement
This research marks a transition for a **Tribhuvan University (TU) graduate** into the world of Cloud Geospatial Data Science. 

It proves that the tools of the 21st century—Google Earth Engine and Deep Learning—can provide reliable, data-driven security for countries on the front lines of climate vulnerability.

---
*Back to: [Documentation Summary (Index)](SUMMARY.md)*
