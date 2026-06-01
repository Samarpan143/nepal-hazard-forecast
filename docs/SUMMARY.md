# Nepal Multi-Hazard Prediction System: Research Documentation
**A Comprehensive Tutorial Suite | 2021-2023 Study**

Welcome! This documentation serves as a 7-part technical "book" that chronicles the complete lifecycle of our research on multi-hazard prediction across Nepal. We begin with the environmental foundations and end with the deep learning validation against real-world disasters.

---

### Table of Contents

1.  **[Chapter 1: Research Foundations](01_Research_Foundations.md)**
    *   The problem statement, Nepal's unique Himalayan geography, and the "Multi-Hazard" research paradigm.
2.  **[Chapter 2: GEE and Environment Setup](02_GEE_and_Environment_Setup.md)**
    *   How we set up our Python virtual environment and authenticated with Google Earth Engine.
3.  **[Chapter 3: Satellite Data Physics](03_Satellite_Data_Physics.md)**
    *   Descriptions of the 5 sensor pillars: Sentinel-1, 2, 5P, MODIS, and TerraClimate.
4.  **[Chapter 4: The Data Pipeline ETL](04_The_Data_Pipeline_ETL.md)**
    *   The Python logic behind sourcing, chunking (year-by-year), cloud masking, and CSV caching.
5.  **[Chapter 5: Deep Learning and LSTMs](05_Deep_Learning_and_LSTMs.md)**
    *   How the **LSTM** handles temporal memory and the transformation of the CSV into 3D tensors.
6.  **[Chapter 6: Verification and Case Studies](06_Verification_and_Case_Studies.md)**
    *   How we used Z-score anomalies to validate our system against the **Melamchi (2021)** and **Wildfire (2023)** disasters.
7.  **[Chapter 7: Research Ethics and Future Work](07_Research_Ethics_and_Future_Work.md)**
    *   Understanding the "Spatial-Temporal" trade-off and how to scale this with the BIPAD portal.
8.  **[Chapter 8: Source Code Architecture](08_Source_Code_Architecture.md)**
    *   Technical breakdown of every Python source file handling GEE extraction, synthetic proxy generation, bipad labeling, and the Bi-LSTM model execution.

---
**Author Note:** *This documentation was authored specifically for the professional research portfolio of a BSc CSIT graduate from Tribhuvan University (TU), Nepal.*
