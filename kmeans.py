import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from thoughtframe.bootstrap import thoughtframe

# =========================
# CONFIG
# =========================
SENSOR_ID = "mic1"
K = 3   # start with 2 or 3
RANDOM_STATE = 42

FEATURE_COLS = [
    "rms_mean",
    "rms_var",
    "centroid_mean",
    "centroid_var",
    "iforest_mean",
]

# =========================
# LOAD WINDOWS
# =========================
def resolve_timewindows_csv(sensor_id: str) -> str:
    root = thoughtframe.resolve_rooted_path(
        THOUGHTFRAME_CONFIG,
        THOUGHTFRAME_CONFIG.get("samples", "audio"),
        sensor_id
    )
    return os.path.join(root, "ifwindows.csv")


csv_path = resolve_timewindows_csv(SENSOR_ID)
print("Loading:", csv_path)

df = pd.read_csv(csv_path)

# Drop any incomplete windows
df = df.dropna(subset=FEATURE_COLS)

# =========================
# FEATURE MATRIX
# =========================
X = df[FEATURE_COLS].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# K-MEANS
# =========================
kmeans = KMeans(
    n_clusters=K,
    random_state=RANDOM_STATE,
    n_init="auto"
)

labels = kmeans.fit_predict(X_scaled)
df["cluster"] = labels

# =========================
# INSPECT CLUSTERS
# =========================
print("\nCluster summary (mean features):")
print(
    df.groupby("cluster")[FEATURE_COLS]
      .mean()
      .round(4)
)

print("\nCluster counts:")
print(df["cluster"].value_counts().sort_index())

# =========================
# VISUALIZATION 1:
# FEATURE SPACE
# =========================
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    df["rms_mean"],
    df["centroid_mean"],
    c=df["cluster"],
    cmap="tab10",
    s=40
)

plt.xlabel("RMS mean")
plt.ylabel("Centroid mean (Hz)")
plt.title("Window feature space (colored by cluster)")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# =========================
# VISUALIZATION 2:
# TIME ORDER
# =========================
plt.figure(figsize=(12, 3))
plt.scatter(
    df["start_t"],
    df["cluster"],
    c=df["cluster"],
    cmap="tab10",
    s=30
)

plt.xlabel("Time (s)")
plt.ylabel("Cluster")
plt.title("Cluster assignment over time")
plt.yticks(range(K))
plt.tight_layout()
plt.show()
