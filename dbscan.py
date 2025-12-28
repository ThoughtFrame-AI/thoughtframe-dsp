import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from thoughtframe.bootstrap import thoughtframe

# -------------------------
# CONFIG
# -------------------------
SENSOR_ID = "mic1"

FEATURES = [
    "rms_mean",
    "centroid_mean",
    "iforest_mean",
]

EPS = 0.8        # distance threshold (we’ll tune this)
MIN_SAMPLES = 5  # minimum windows to form a regime

# -------------------------
# LOAD WINDOWS
# -------------------------
root = thoughtframe.resolve_rooted_path(
    THOUGHTFRAME_CONFIG,
    THOUGHTFRAME_CONFIG.get("samples", "audio"),
    SENSOR_ID
)

csv_path = os.path.join(root, "timewindows.csv")
print("Loading:", csv_path)

df = pd.read_csv(csv_path)

X = df[FEATURES].dropna().values

# -------------------------
# SCALE FEATURES
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# DBSCAN
# -------------------------
db = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES
)

labels = db.fit_predict(X_scaled)
df["cluster"] = labels

print("Clusters found:", set(labels))
print("Noise points:", np.sum(labels == -1))

# -------------------------
# TIME VIEW
# -------------------------
plt.figure(figsize=(12, 4))
plt.scatter(df["start_t"], df["cluster"], s=12)
plt.yticks(sorted(set(labels)))
plt.xlabel("Time (s)")
plt.ylabel("Cluster")
plt.title("DBSCAN cluster assignment over time")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# FEATURE SPACE VIEW
# -------------------------
plt.figure(figsize=(6, 6))
scatter = plt.scatter(
    df["rms_mean"],
    df["centroid_mean"],
    c=df["cluster"],
    cmap="tab10",
    s=30
)
plt.xlabel("RMS mean")
plt.ylabel("Centroid mean (Hz)")
plt.title("Window feature space (DBSCAN)")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()
