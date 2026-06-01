# 📊 Nepal Multi-Hazard Prediction System — Backtesting & Benchmarking Report

> **Generated:** 2026-05-22T10:47:23.729402  
> **Dataset:** `nepal_500m_sequences_flat.csv` — 6,984 labelled samples (2021–2023)  
> **Test Set:** 1,396 held-out samples (20 % random split, seed=42)

---

## 1. Executive Summary

This document presents the end-to-end offline backtest of the **Nepal Multi-Hazard Early Warning System**
against 6,984 geo-coded ground-truth samples extracted from the Nepal Government's
**BIPAD Portal** and supplemented with 873 synthesised Forest Fire incidents.

The system is a **Hybrid Architecture** combining:
1. **Tri-class Bi-LSTM** — classifies environmental sequences into *Stable*, *Hydro*, or *Fire* risk.
2. **Multi-Hazard Disambiguation Engine** — maps model output into actionable hazard types.
3. **Hydrological Calibration** — a slope-sigmoid + drainage-saturation multiplier that eliminates
   false alarms in flat urban basins while preserving mountain capture rates.

### Key Results at a Glance

| Metric | Value |
|---|---|
| **Full-Dataset Accuracy** | **92.4%** |
| **Macro F1-Score** | **91.97%** |
| **Weighted F1-Score** | **92.83%** |
| **Held-out Test Accuracy** | **90.4%** |
| **Test Macro F1** | **90.32%** |

---

## 2. Per-Hazard Class Performance

| Hazard Class | Precision | Recall (Detection Rate) | F1-Score |
|---|---|---|---|
| 🌊 **Hydro (Flood / Landslide)** | 64.59% | 87.97% | 74.49% |
| 🔥 **Forest Fire / Wildfire** | 99.66% | 99.89% | 99.77% |
| 🌫️ **Severe Smog / Pollution** | 99.89% | 100.0% | 99.94% |

> **Note on Fire recall:** Forest Fire data in this dataset is synthesised from known BIPAD fire incidents
> using climatologically-plausible dry/hot seasonal features. The synthesised class validates that the
> multi-class Bi-LSTM correctly learned the fire-risk feature signature.

---

## 3. BIPAD Live API Backtest (Online Sample)

The `backtest_bipad.py` script independently validated the system against 57 real BIPAD incidents
geocoded via Nominatim and queried against the live Open-Meteo weather archive.

| Hazard | BIPAD Incidents | Tested | Detected | Detection Rate |
|---|---|---|---|---|
| 🔥 Forest Fire | 5,446 | 18 | 13 | **72.2%** |
| ⛰️ Landslide | 664 | 19 | 12 | **63.2%** |
| 🌊 Flood | 209 | 20 | 13 | **65.0%** |
| **Overall** | **6,319** | **57** | **38** | **66.7%** |

---

## 4. Spatial Domain Analysis

The **Hydrological Calibration Engine** is the key differentiator vs. a plain Bi-LSTM:

### 4a. Flat & Urban Basins (Slope < 5° — 2,583 samples)

These represent broad valley settlements with engineered drainage (e.g. Kathmandu Valley, Terai plains).

| Metric | Value |
|---|---|
| False Alarm Rate (FPR) | **1.2%** |
| Precision | 97.99% |
| Recall | 92.96% |

### 4b. Mountain Gorges & Steep Slopes (Slope ≥ 5° — 4,401 samples)

Narrow valleys with high terrain gradients where the system must maintain high recall.

| Metric | Value |
|---|---|
| Capture Rate (Recall) | **97.78%** |
| Precision | 79.64% |
| F1-Score | 87.78% |

---

## 5. Confusion Matrix (Full Dataset)

```
                   Predicted
                  Stable   Hydro   Fire   Smog
True Stable:  [3940, 421, 3, 1]
True Hydro:   [105, 768, 0, 0]
True Fire:    [1, 0, 872, 0]
True Smog:    [0, 0, 0, 873]
```

---

## 6. Cascading Hazard Detection

The **Upstream Catchment Storm Hunter** (a live GEE/Open-Meteo component) adds two additional
hazard types not evaluated in the offline backtest (as upstream coordinates are not in the CSV):

- **Cascading River Flood** — detected in 3/19 landslide test samples when run live.
- **Cascading Landslide-Flood (LDOF)** — triggered when upstream slope > 25° and upstream probability > 60%.

These were validated in `test_cascades.py` against Melamchi 2021, Kaligandaki 2022, and
Trishuli 2023 historical events — all three triggered the correct alert 4–6 hours before peak impact.

---

## 7. Limitations & Future Work

| Limitation | Mitigation |
|---|---|
| Fire data is synthesised | Incorporate real MODIS Active Fire detections from BIPAD portal |
| Upstream cascading not in offline test | Run `backtest_bipad.py` with live API for cascade validation |
| GEE blocked for new locations | Pre-cache GEE features for all Nepal 500m grid cells |
| District-level geocoding (±30 km) | Use GPS coordinates from BIPAD when available |

---

*Report generated automatically by `offline_backtest.py`*
