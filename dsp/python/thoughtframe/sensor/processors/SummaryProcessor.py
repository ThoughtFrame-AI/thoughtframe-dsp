import os
import time
import threading
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytimeparse.timeparse import timeparse

from thoughtframe.sensor.interface import AcousticChunkProcessor
from tf_core.bootstrap import thoughtframe
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG


class ForensicSummaryProcessor(AcousticChunkProcessor):

    OP_NAME = "forensic_summary"

    def __init__(self, cfg, sensor):
        self.sensor_id = sensor.sensor_id
        self.fs = sensor.fs

        self.interval_sec = timeparse(cfg.get("interval", "30s"))
        self.horizons = cfg.get("horizons", ["5m", "1h", "24h"])

        base = cfg.get("csv_name", "telemetry")
        self.prefix = cfg.get("csv_prefix", "")
        filename = f"{self.prefix}_{base}.csv" if self.prefix else f"{base}.csv"

        self.data_root = thoughtframe.resolve_rooted_path(
            THOUGHTFRAME_CONFIG,
            THOUGHTFRAME_CONFIG.get("samples", "audio"),
            self.sensor_id
        )

        self.telemetry_csv = os.path.join(self.data_root, filename)

        self.outdir = os.path.join(self.data_root, "forensics")
        os.makedirs(self.outdir, exist_ok=True)

        self._last_run = 0
        self._spectrogram_buffer = []

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    def process(self, chunk, analysis):
        max_chunks = int((5 * 60) / (len(chunk) / self.fs))
        self._spectrogram_buffer.append(chunk.copy())
        if len(self._spectrogram_buffer) > max_chunks:
            self._spectrogram_buffer.pop(0)

        now = time.time()
        if now - self._last_run >= self.interval_sec:
            self._last_run = now
            threading.Thread(
                target=self._build_summaries,
                daemon=True
            ).start()

    # ============================================================
    # Core summary builder
    # ============================================================

    def _build_summaries(self):
        if not os.path.exists(self.telemetry_csv):
            return

        rows = self._load_telemetry()
        if len(rows) < 10:
            return

        latest_t = rows[-1]["t_sec"]

        if self._spectrogram_buffer:
            self._plot_spectrogram("5m")

        for h in self.horizons:
            horizon_sec = timeparse(h)
            t_min = latest_t - horizon_sec
            window = [r for r in rows if r["t_sec"] >= t_min]

            if len(window) < 10:
                continue

            label = h.replace(" ", "")

            self._plot_baseline_with_impulses(window, label, t_min)
            self._plot_baseline_with_iforest(window, label, t_min)

            self._plot_rms(window, label)
            self._plot_iforest(window, label)
            self._plot_centroid(window, label)

            self._plot_dcmt_with_windows(window, label, t_min)

    # ============================================================
    # CSV loaders
    # ============================================================

    def _load_telemetry(self):
        rows = []
        try:
            with open(self.telemetry_csv, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if not r.get("t_sec"):
                        continue
                    try:
                        rows.append({
                            "t_sec": float(r["t_sec"]),
                            "rms": float(r.get("rms", 0)),
                            "rms_mean": float(r.get("rms_mean", 0)),
                            "spec_centroid_hz": float(r.get("spec_centroid_hz", 0)),
                            "centroid_mean": float(r.get("centroid_mean", 0)),
                            "iforest_score": float(r.get("iforest_score", 0)),
                            "anomaly_rate": float(r.get("anomaly_rate", 0)),
                            "dcmt_deviation": float(r.get("dcmt_deviation", 0)),
                            "dcmt_embedding_norm": float(r.get("dcmt_embedding_norm", 0)),
                        })
                    except Exception:
                        continue
        except Exception:
            pass
        return rows

    def _load_windows_csv(self, name):
        fname = f"{self.prefix}_{name}.csv" if self.prefix else f"{name}.csv"
        path = os.path.join(self.data_root, fname)
        rows = []

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return rows

        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        rows.append({
                            "state": r.get("state"),
                            "start_t": float(r["start_t"]),
                            "end_t": float(r["end_t"]) if r.get("end_t") else None,
                        })
                    except Exception:
                        continue
        except Exception:
            pass
        return rows

    # ============================================================
    # Window drawing helpers
    # ============================================================

    def _draw_windows(self, ax, windows, color, t_min):
        for r in windows:
            if r.get("state") != "EVENT":
                continue
            start = r["start_t"]
            end = r["end_t"] if r["end_t"] else start
            if end >= t_min:
                ax.axvspan(max(start, t_min), end, color=color, alpha=0.15)

    def _draw_dcmt_windows(self, ax, windows, t_min):
        # Stable color palette (extend if needed)
        pin_colors = [
            "#d62728",  # red
            "#1f77b4",  # blue
            "#2ca02c",  # green
            "#ff7f0e",  # orange
            "#9467bd",  # purple
            "#17becf",  # cyan
            "#8c564b",  # brown
            "#e377c2",  # pink
        ]
    
        for r in windows:
            start = r["start_t"]
            end = r["end_t"] if r["end_t"] else start
    
            if end < t_min:
                continue
    
            state = r.get("state") or ""
            if not state.startswith("PIN_"):
                continue
    
            # Extract pin id
            try:
                pin_id = int(state.split("_")[1])
            except Exception:
                continue
    
            color = pin_colors[pin_id % len(pin_colors)]
    
            ax.axvspan(
                max(start, t_min),
                end,
                color=color,
                alpha=0.22
            )


    def _get_out_path(self, type_name, label):
        fname = f"{self.prefix}_{type_name}_{label}.png" if self.prefix else f"{type_name}_{label}.png"
        return os.path.join(self.outdir, fname)

    # ============================================================
    # Spectrogram
    # ============================================================

    def _plot_spectrogram(self, label):
        data = np.concatenate(self._spectrogram_buffer)
        plt.figure(figsize=(12, 4))
        plt.specgram(data, Fs=self.fs, NFFT=1024, noverlap=512, cmap="magma")
        plt.tight_layout()
        plt.savefig(self._get_out_path("spectrogram", label))
        plt.close()

    # ============================================================
    # Scalar plots
    # ============================================================

    def _plot_rms(self, rows, label):
        plt.figure(figsize=(10, 3))
        plt.plot([r["t_sec"] for r in rows], [r["rms_mean"] for r in rows])
        plt.tight_layout()
        plt.savefig(self._get_out_path("rms", label))
        plt.close()

    def _plot_iforest(self, rows, label):
        plt.figure(figsize=(10, 3))
        plt.plot([r["t_sec"] for r in rows], [r["iforest_score"] for r in rows])
        plt.axhline(0, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(self._get_out_path("iforest", label))
        plt.close()

    def _plot_centroid(self, rows, label):
        plt.figure(figsize=(10, 3))
        plt.plot([r["t_sec"] for r in rows], [r["centroid_mean"] for r in rows])
        plt.tight_layout()
        plt.savefig(self._get_out_path("centroid", label))
        plt.close()

    # ============================================================
    # Composite forensic plots
    # ============================================================

    def _plot_baseline_with_impulses(self, rows, label, t_min):
        t = [r["t_sec"] for r in rows]
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

        axes[0].plot(t, [r["rms"] for r in rows])
        axes[1].plot(t, [r["spec_centroid_hz"] for r in rows])
        axes[2].plot(t, [r["iforest_score"] for r in rows])

        plt.tight_layout()
        plt.savefig(self._get_out_path("baseline_impulses", label))
        plt.close()

    def _plot_baseline_with_iforest(self, rows, label, t_min):
        if_windows = self._load_windows_csv("IsolationForestWindowIsolator")
        dcmt_windows = self._load_windows_csv("DcmtIsolator")

        t = [r["t_sec"] for r in rows]
        fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)

        axes[0].plot(t, [r["rms"] for r in rows])
        axes[1].plot(t, [r["spec_centroid_hz"] for r in rows])
        axes[2].plot(t, [r["iforest_score"] for r in rows])
        axes[3].plot(t, [r["anomaly_rate"] for r in rows])
        axes[4].plot(t, [r["dcmt_deviation"] for r in rows])

        for ax in axes[:4]:
            self._draw_windows(ax, if_windows, "red", t_min)

        self._draw_dcmt_windows(axes[4], dcmt_windows, t_min)
        axes[4].set_ylabel("DCMT deviation")

        plt.tight_layout()
        plt.savefig(self._get_out_path("baseline_iforest", label))
        plt.close()

    # ============================================================
    # DCMT-only plot
    # ============================================================

    def _plot_dcmt_with_windows(self, rows, label, t_min):
        windows = self._load_windows_csv("DcmtIsolator")
        t = [r["t_sec"] for r in rows]

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        axes[0].plot(t, [r["dcmt_deviation"] for r in rows])
        axes[1].plot(t, [r["dcmt_embedding_norm"] for r in rows])

        for ax in axes:
            self._draw_dcmt_windows(ax, windows, t_min)

        plt.tight_layout()
        plt.savefig(self._get_out_path("dcmt", label))
        plt.close()
