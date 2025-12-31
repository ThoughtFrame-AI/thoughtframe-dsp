import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from tf_core.bootstrap import thoughtframe

# =========================
# CONFIGURATION
# =========================
SENSOR_ID = "beam_0"  # UPDATED
IFOREST_THRESHOLD = -0.1

def resolve_path(sensor_id, filename):
    root = thoughtframe.resolve_rooted_path(
        THOUGHTFRAME_CONFIG,
        THOUGHTFRAME_CONFIG.get("samples", "audio"),
        sensor_id
    )
    return os.path.join(root, filename)

# ---- Load Data ----
telemetry_path = resolve_path(SENSOR_ID, "telemetry.csv")
impulse_path = resolve_path(SENSOR_ID, "ImpulseIsolator.csv")
iforest_path = resolve_path(SENSOR_ID, "IsolationForestWindowIsolator.csv")

print(f"[{SENSOR_ID}] Loading telemetry...")
if not os.path.exists(telemetry_path):
    print(f"ERROR: Telemetry not found at {telemetry_path}")
    exit(1)

df = pd.read_csv(telemetry_path)

# Load optional CSVs safely
impulse_windows = pd.DataFrame()
if os.path.exists(impulse_path):
    impulse_windows = pd.read_csv(impulse_path)
    print(f"Loaded {len(impulse_windows)} impulse events")
else:
    print("No ImpulseIsolator.csv found (skipping overlays)")

iforest_windows = pd.DataFrame()
if os.path.exists(iforest_path):
    iforest_windows = pd.read_csv(iforest_path)
    print(f"Loaded {len(iforest_windows)} IF windows")
else:
    print("No IsolationForestWindowIsolator.csv found (skipping overlays)")

# ---- Plotting Helpers ----

def draw_impulse_markers(ax, windows):
    """Draws thin vertical lines for momentary impulses"""
    if windows.empty: return
    # Filter for valid timestamps if needed
    for _, row in windows.iterrows():
        ax.axvline(row["start_t"], color="black", alpha=0.3, linewidth=0.8, linestyle=":")

def draw_anomaly_regions(ax, windows):
    """Draws shaded RED boxes for durations where state == EVENT"""
    if windows.empty: return
    
    # Filter only for 'EVENT' state rows
    events = windows[windows["state"] == "EVENT"]
    
    for _, row in events.iterrows():
        # Use axvspan to shade the region from start_t to end_t
        start = row["start_t"]
        # If the window is still open, end_t might be missing or we use current time
        end = row.get("end_t") if not pd.isna(row.get("end_t")) else start + 5.0 
        
        ax.axvspan(start, end, color='red', alpha=0.15)
        
        # Add a darker line at the start
        ax.axvline(start, color='red', alpha=0.4, linewidth=1.0)

# ---- Layout ----
t = df["t_sec"]
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

# 1. RMS
ax1 = axes[0]
ax1.plot(t, df["rms"], linewidth=0.8, color="#1f77b4")
draw_impulse_markers(ax1, impulse_windows)
draw_anomaly_regions(ax1, iforest_windows)
ax1.set_ylabel("RMS (Amplitude)")
ax1.set_title(f"Acoustic Telemetry: {SENSOR_ID} (48kHz High-Def)")
ax1.grid(True, alpha=0.3)

# 2. Spectral Centroid (The High Frequency Detector)
ax2 = axes[1]
ax2.plot(t, df["spec_centroid_hz"], linewidth=0.8, color="#2ca02c")
draw_impulse_markers(ax2, impulse_windows)
draw_anomaly_regions(ax2, iforest_windows)
ax2.set_ylabel("Centroid (Hz)")
ax2.grid(True, alpha=0.3)

# 3. Isolation Forest Score
ax3 = axes[2]
ax3.plot(t, df["iforest_score"], linewidth=1.0, color="#d62728")
ax3.axhline(IFOREST_THRESHOLD, color="black", linestyle="--", linewidth=1.5, label="Threshold")
draw_impulse_markers(ax3, impulse_windows)
draw_anomaly_regions(ax3, iforest_windows)
ax3.set_ylabel("Anomaly Score")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3)

# 4. Anomaly Rate (Optional metric)
ax4 = axes[3]
if "anomaly_rate" in df.columns:
    ax4.plot(t, df["anomaly_rate"], linewidth=0.8, color="#9467bd")
else:
    ax4.text(0.5, 0.5, "No 'anomaly_rate' in telemetry", ha='center')
    
draw_impulse_markers(ax4, impulse_windows)
draw_anomaly_regions(ax4, iforest_windows)
ax4.set_ylabel("Anomaly Rate")
ax4.set_xlabel("Time (s)")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()