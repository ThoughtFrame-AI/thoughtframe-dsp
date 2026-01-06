import os
import pandas as pd
import matplotlib.pyplot as plt

from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from tf_core.bootstrap import thoughtframe

# =========================
# CONFIG
# =========================
SENSOR_ID = "beam_0"

OBSERVERS = {
    "baseline": "red",
    "impulses_high": "blue",
    "impulses_low": "green",
}

IFOREST_THRESHOLD = -0.1

# =========================
# PATH HELPERS
# =========================
def resolve_path(sensor_id, filename):
    root = thoughtframe.resolve_rooted_path(
        THOUGHTFRAME_CONFIG,
        THOUGHTFRAME_CONFIG.get("samples", "audio"),
        sensor_id,
    )
    return os.path.join(root, filename)

# =========================
# SAFE LOADERS
# =========================
def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    if os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_telemetry(prefix):
    if prefix == "baseline":
        return load_csv(resolve_path(SENSOR_ID, "baseline_telemetry.csv"))
    return load_csv(resolve_path(SENSOR_ID, f"{prefix}_telemetry.csv"))

def load_windows(prefix, name):
    return load_csv(resolve_path(SENSOR_ID, f"{prefix}_{name}.csv"))

# =========================
# DRAW HELPERS
# =========================
def draw_impulses(ax, windows, color):
    if windows.empty:
        return
    for _, r in windows.iterrows():
        ax.axvline(r["start_t"], color=color, linestyle=":", alpha=0.4)

def draw_windows(ax, windows, color):
    if windows.empty:
        return
    events = windows[windows["state"] == "EVENT"]
    for _, r in events.iterrows():
        start = r["start_t"]
        end = r["end_t"] if not pd.isna(r.get("end_t")) else start
        ax.axvspan(start, end, color=color, alpha=0.15)

# =========================
# OBSERVER VIEW
# =========================
def plot_observer(prefix, color):
    df = load_telemetry(prefix)
    if df.empty:
        print(f"[skip] {prefix}: no telemetry")
        return

    iforest = load_windows(prefix, "IsolationForestWindowIsolator")
    impulses = load_windows(prefix, "ImpulseIsolator")

    t = df["t_sec"]
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)

    # RMS
    axes[0].plot(t, df["rms"], linewidth=0.8)
    draw_windows(axes[0], iforest, color)
    draw_impulses(axes[0], impulses, "black")
    axes[0].set_ylabel("RMS")
    axes[0].set_title(f"{SENSOR_ID} — {prefix} observer")

    # Centroid
    axes[1].plot(t, df["spec_centroid_hz"], linewidth=0.8)
    draw_windows(axes[1], iforest, color)
    draw_impulses(axes[1], impulses, "black")
    axes[1].set_ylabel("Centroid (Hz)")

    # IF score
    axes[2].plot(t, df["iforest_score"], linewidth=0.9)
    axes[2].axhline(IFOREST_THRESHOLD, color="black", linestyle="--")
    draw_windows(axes[2], iforest, color)
    draw_impulses(axes[2], impulses, "black")
    axes[2].set_ylabel("IF score")

    # Anomaly rate
    if "anomaly_rate" in df.columns:
        axes[3].plot(t, df["anomaly_rate"], linewidth=0.9)
    axes[3].set_ylabel("Anomaly rate")
    axes[3].set_xlabel("Time (s)")
    draw_windows(axes[3], iforest, color)

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# =========================
# COMPARISON (DECISIONS ONLY)
# =========================
def plot_comparison():
    base_df = load_telemetry("baseline")
    if base_df.empty:
        print("[skip] comparison: no baseline telemetry")
        return

    t = base_df["t_sec"]
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    for prefix, color in OBSERVERS.items():
        iforest = load_windows(prefix, "IsolationForestWindowIsolator")
        impulses = load_windows(prefix, "ImpulseIsolator")

        for ax in axes:
            draw_windows(ax, iforest, color)
            draw_impulses(ax, impulses, color)

    # Base signals for reference
    axes[0].plot(t, base_df["rms"], linewidth=0.8)
    axes[1].plot(t, base_df["spec_centroid_hz"], linewidth=0.8)
    axes[2].plot(t, base_df["iforest_score"], linewidth=0.8)
    axes[2].axhline(IFOREST_THRESHOLD, color="black", linestyle="--")
    if "anomaly_rate" in base_df.columns:
        axes[3].plot(t, base_df["anomaly_rate"], linewidth=0.8)

    labels = ["RMS", "Centroid (Hz)", "IF score", "Anomaly rate"]
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[0].set_title(f"{SENSOR_ID} — decision overlays (baseline / guards)")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# =========================
# RUN
# =========================
plot_comparison()

for prefix, color in OBSERVERS.items():
    plot_observer(prefix, color)
