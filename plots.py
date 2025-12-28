import os
import pandas as pd
import matplotlib.pyplot as plt

from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from thoughtframe.bootstrap import thoughtframe

# =========================
# HARD-CODED FOR ECLIPSE
# =========================
SENSOR_ID = "mic1"
IFOREST_THRESHOLD = -0.1


def resolve_telemetry_csv(sensor_id: str) -> str:
    root = thoughtframe.resolve_rooted_path(
        THOUGHTFRAME_CONFIG,
        THOUGHTFRAME_CONFIG.get("samples", "audio"),
        sensor_id
    )
    return os.path.join(root, "telemetry.csv")


def resolve_impulse_windows_csv(sensor_id: str) -> str:
    root = thoughtframe.resolve_rooted_path(
        THOUGHTFRAME_CONFIG,
        THOUGHTFRAME_CONFIG.get("samples", "audio"),
        sensor_id
    )
    return os.path.join(root, "ImpulseIsolator.csv")


def resolve_iforest_windows_csv(sensor_id: str) -> str:
    root = thoughtframe.resolve_rooted_path(
        THOUGHTFRAME_CONFIG,
        THOUGHTFRAME_CONFIG.get("samples", "audio"),
        sensor_id
    )
    return os.path.join(root, "IsolationForestWindowIsolator.csv")


def draw_window_boundaries(ax, windows, color="k", alpha=0.15, linewidth=1.0):
    for _, row in windows.iterrows():
        ax.axvline(row["start_t"], color=color, alpha=alpha, linewidth=linewidth)


# ---- load telemetry ----
csv_path = resolve_telemetry_csv(SENSOR_ID)
impulse_path = resolve_impulse_windows_csv(SENSOR_ID)
iforest_path = resolve_iforest_windows_csv(SENSOR_ID)

print("Loading telemetry from:", csv_path)
print("Loading impulse windows from:", impulse_path)
print("Loading IF windows from:", iforest_path)

df = pd.read_csv(csv_path)
impulse_windows = pd.read_csv(impulse_path)
iforest_windows = pd.read_csv(iforest_path)

# ---- time axis ----
t = df["t_sec"]
plt.figure(figsize=(14, 10))

# RMS
ax1 = plt.subplot(4, 1, 1)
ax1.plot(t, df["rms"], linewidth=0.8)
draw_window_boundaries(ax1, impulse_windows, color="gray", alpha=0.25)
draw_window_boundaries(ax1, iforest_windows, color="red", alpha=0.25)
ax1.set_ylabel("RMS")
ax1.set_title("Acoustic Telemetry")

# Spectral centroid
ax2 = plt.subplot(4, 1, 2)
ax2.plot(t, df["spec_centroid_hz"], linewidth=0.8)
draw_window_boundaries(ax2, impulse_windows, color="gray", alpha=0.25)
draw_window_boundaries(ax2, iforest_windows, color="red", alpha=0.25)
ax2.set_ylabel("Centroid (Hz)")

# Isolation Forest score
ax3 = plt.subplot(4, 1, 3)
ax3.plot(t, df["iforest_score"], linewidth=0.8)
ax3.axhline(IFOREST_THRESHOLD, color="r", linestyle="--")
draw_window_boundaries(ax3, impulse_windows, color="gray", alpha=0.25)
draw_window_boundaries(ax3, iforest_windows, color="red", alpha=0.25)
ax3.set_ylabel("IF score")

# Anomaly rate
ax4 = plt.subplot(4, 1, 4)
ax4.plot(t, df["anomaly_rate"], linewidth=0.8)
draw_window_boundaries(ax4, impulse_windows, color="gray", alpha=0.25)
draw_window_boundaries(ax4, iforest_windows, color="red", alpha=0.25)
ax4.set_ylabel("Anomaly rate")
ax4.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()
