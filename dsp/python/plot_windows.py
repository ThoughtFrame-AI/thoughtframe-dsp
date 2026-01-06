import os
import pandas as pd
import matplotlib.pyplot as plt

from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from thoughtframe.bootstrap import thoughtframe

SENSOR_ID = "mic1"

def resolve_windows_csv(sensor_id: str) -> str:
    root = thoughtframe.resolve_rooted_path(
        THOUGHTFRAME_CONFIG,
        THOUGHTFRAME_CONFIG.get("samples", "audio"),
        sensor_id
    )
    return os.path.join(root, "timewindows.csv")

csv_path = resolve_windows_csv(SENSOR_ID)
print("Loading windows from:", csv_path)

df = pd.read_csv(csv_path)

t = df["start_t"]

plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(t, df["rms_mean"], marker="o", linewidth=1)
plt.ylabel("RMS mean")

plt.subplot(3, 1, 2)
plt.plot(t, df["centroid_mean"], marker="o", linewidth=1)
plt.ylabel("Centroid mean (Hz)")

plt.subplot(3, 1, 3)
plt.plot(t, df["iforest_mean"], marker="o", linewidth=1)
plt.ylabel("IF mean")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(
    df["rms_mean"],
    df["centroid_mean"],
    c=df["iforest_mean"],
    cmap="coolwarm",
    s=40
)
plt.xlabel("RMS mean")
plt.ylabel("Centroid mean (Hz)")
plt.colorbar(label="IF mean")
plt.title("Window feature space")
plt.show()


