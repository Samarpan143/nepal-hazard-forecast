"""
Generate spatial risk map: Bi-LSTM predicted hazard probability vs actual BIPAD events.
Outputs a publication-quality map overlaying model predictions on Nepal's geography.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load data
print("Loading data...")
df = pd.read_csv('nepal_500m_sequences_flat.csv')
points = pd.read_csv('sample_locations_raw.csv')

time_steps = 7
base_features = ['VV', 'VH', 'NDVI', 'Aerosol', 'LST', 'pr', 'soil', 'Slope', 'Aspect', 'TPI']

# 2. Build X, y (same pipeline as training)
samples = []
labels = []
for i, row in df.iterrows():
    sample_seq = []
    for t in range(time_steps):
        timestep_features = []
        for f in base_features:
            col_name = f"T{t}_{f}"
            val = row.get(col_name, 0)
            if pd.isna(val): val = 0
            timestep_features.append(val)
        sample_seq.append(timestep_features)
    samples.append(sample_seq)
    labels.append(row['label'])

X = np.array(samples)
y = np.array(labels)

# 3. Scale (fit on same data)
num_samples, ts, num_feats = X.shape
X_flat = X.reshape(-1, num_feats)
scaler = StandardScaler()
X_scaled_flat = scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(num_samples, ts, num_feats)

# 4. Same train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Get the test indices to map back to coordinates
_, test_indices = train_test_split(range(len(y)), test_size=0.2, random_state=42)

# 5. Load trained model and predict on TEST set
print("Loading trained Bi-LSTM model...")
model = tf.keras.models.load_model('nepal_hazard_model_500m.h5')
y_pred_prob = model.predict(X_test).flatten()

# 6. Map test indices to coordinates
# The 'df' and 'points' may have different ordering, use df's sample_id to get coords
test_lats = []
test_lons = []
for idx in test_indices:
    row = df.iloc[idx]
    sid = row.get('sample_id', idx)
    # Try to find matching coordinates
    if 'latitude' in points.columns and sid < len(points):
        test_lats.append(points.iloc[int(sid)]['latitude'])
        test_lons.append(points.iloc[int(sid)]['longitude'])
    elif 'latitude' in df.columns:
        test_lats.append(row['latitude'])
        test_lons.append(row['longitude'])
    else:
        # Fallback: use sample_id to index into points
        test_lats.append(points.iloc[idx % len(points)]['latitude'])
        test_lons.append(points.iloc[idx % len(points)]['longitude'])

test_lats = np.array(test_lats)
test_lons = np.array(test_lons)

# 7. Create the map
print("Generating spatial risk map...")
fig, ax = plt.subplots(1, 1, figsize=(14, 8), facecolor='#0a0a1a')
ax.set_facecolor('#0a0a1a')

# Nepal boundary approximation (simplified polygon)
nepal_lon = [80.05, 80.2, 81.1, 82.7, 84.0, 85.5, 86.5, 87.8, 88.2, 88.15, 87.5, 86.5, 85.0, 83.5, 82.0, 80.5, 80.05]
nepal_lat = [28.8, 27.5, 26.4, 26.4, 26.6, 26.6, 26.4, 26.5, 27.0, 27.8, 28.0, 28.2, 28.8, 29.3, 30.0, 30.4, 28.8]
ax.fill(nepal_lon, nepal_lat, color='#1a1a2e', alpha=0.6, edgecolor='#444', linewidth=1.5, zorder=1)
ax.plot(nepal_lon, nepal_lat, color='#555', linewidth=1.2, zorder=2)

# Separate true positives, false negatives, true negatives, false positives
y_pred_binary = (y_pred_prob > 0.5).astype(int)

# Plot all test predictions as background (colored by predicted probability)
# True Negatives (predicted safe, actually safe) — dim
tn_mask = (y_test == 0) & (y_pred_binary == 0)
ax.scatter(test_lons[tn_mask], test_lats[tn_mask], 
           c=y_pred_prob[tn_mask], cmap='Blues', s=12, alpha=0.3, 
           vmin=0, vmax=1, zorder=3, edgecolors='none')

# False Positives (predicted hazard, actually safe) — orange outline
fp_mask = (y_test == 0) & (y_pred_binary == 1)
ax.scatter(test_lons[fp_mask], test_lats[fp_mask],
           c='#f0a030', s=25, alpha=0.6, marker='s', zorder=4, edgecolors='#f0a030', linewidths=0.5)

# True Positives (predicted hazard, actually hazard) — bright red
tp_mask = (y_test == 1) & (y_pred_binary == 1)
ax.scatter(test_lons[tp_mask], test_lats[tp_mask],
           c=y_pred_prob[tp_mask], cmap='YlOrRd', s=60, alpha=0.95,
           vmin=0.3, vmax=1.0, zorder=6, edgecolors='white', linewidths=0.8, marker='o')

# False Negatives (missed hazards) — large X markers in cyan
fn_mask = (y_test == 1) & (y_pred_binary == 0)
ax.scatter(test_lons[fn_mask], test_lats[fn_mask],
           c='#00ffff', s=80, alpha=0.9, marker='X', zorder=7, edgecolors='white', linewidths=0.8)

# Actual BIPAD events (ground truth positive) — diamond outline
actual_hazard = y_test == 1
ax.scatter(test_lons[actual_hazard], test_lats[actual_hazard],
           facecolors='none', edgecolors='#e94560', s=100, alpha=0.5, 
           marker='D', linewidths=1.2, zorder=5, label='BIPAD Event')

# Title and labels
ax.set_title('Nepal Multi-Hazard Prediction: Bi-LSTM Risk Score vs BIPAD Ground Truth\n(Test Set, n=1,038)',
             fontsize=14, fontweight='bold', color='white', pad=15)
ax.set_xlabel('Longitude (°E)', fontsize=11, color='#aaa')
ax.set_ylabel('Latitude (°N)', fontsize=11, color='#aaa')
ax.tick_params(colors='#888')

# Colorbar for risk score
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=mcolors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Predicted Hazard Risk Score  P̂(H|X)', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', markersize=10,
           label=f'True Positive (Detected) — {tp_mask.sum()}', linestyle='None'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#00ffff', markersize=10,
           label=f'False Negative (Missed) — {fn_mask.sum()}', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#f0a030', markersize=8,
           label=f'False Positive (False Alarm) — {fp_mask.sum()}', linestyle='None'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='none', markeredgecolor='#e94560',
           markersize=8, label=f'BIPAD Ground Truth Events — {actual_hazard.sum()}', linestyle='None'),
]
legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=9, 
                   facecolor='#1a1a2e', edgecolor='#444', labelcolor='white',
                   framealpha=0.9)

# Stats annotation
stats_text = (
    f"Recall: {tp_mask.sum()}/{actual_hazard.sum()} = {tp_mask.sum()/actual_hazard.sum():.1%}\n"
    f"Precision: {tp_mask.sum()}/{(tp_mask | fp_mask).sum():.0f} = {tp_mask.sum()/(tp_mask.sum()+fp_mask.sum()):.1%}\n"
    f"Missed Events: {fn_mask.sum()}"
)
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9, color='white',
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#e94560', alpha=0.9))

ax.set_xlim(79.5, 88.8)
ax.set_ylim(25.8, 30.8)

# Grid
ax.grid(True, alpha=0.15, color='#555')
for spine in ax.spines.values():
    spine.set_color('#333')

plt.tight_layout()
plt.savefig('spatial_risk_map.png', dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
print("Saved: spatial_risk_map.png")
plt.close()
