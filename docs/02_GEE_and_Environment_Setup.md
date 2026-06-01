# Chapter 2: GEE and Environment Setup
## Building the Cloud Workbench

### 2.1 Technical Prerequisites
* **Python 3.9+** (Required for `tensorflow` and `earthengine-api`).
* **Google Earth Engine (GEE) Account:** Must be registered and approved by Google.
* **Google Cloud Project:** Created in the [Google Cloud Console](https://console.cloud.google.com). Our project ID for this research is `nd-sem-project`.

### 2.2 Local Environment Setup
We move away from the "System" Python to avoid library conflicts.

#### Creating the Virtual Environment
```bash
mkdir nepal_hazard_prediction && cd nepal_hazard_prediction
python3 -m venv .venv
source .venv/bin/activate
```

#### Installing the Machine Learning and Geospatial Stack
```bash
pip install geemap earthengine-api xarray pandas scikit-learn tensorflow
```

### 2.3 Authenticating Google Earth Engine
For a local researcher to talk to Google's supercomputers, we use the OAuth2 protocol.

#### The Command
```bash
earthengine authenticate --auth_mode localhost
```
*   **How it works:** This command triggers a browser-based login flow.
*   **The Credential File:** A token is saved locally to `~/.config/earthengine/credentials`. This token allows your Python scripts to authenticate automatically without a login screen every time.

### 2.4 Initializing the Session
In every script, we MUST initialize the specific project to start the engine:
```python
import ee
ee.Initialize(project='nd-sem-project')
```
*   **The Error:** If the Earth Engine API is not enabled on your Google Cloud Console for that project, you will receive a `403 Forbidden` error. Go to APIs & Services > Library > Earth Engine API > Enable.

---
*Next Chapter: [Chapter 3: Satellite Data Physics](03_Satellite_Data_Physics.md)*
