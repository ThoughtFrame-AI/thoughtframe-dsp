import os
import time
import threading
import csv

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from pytimeparse.timeparse import timeparse

from thoughtframe.sensor.interface import AcousticChunkProcessor
from tf_core.bootstrap import thoughtframe
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG


class ForensicSummaryProcessor(AcousticChunkProcessor):
    """
    Builds forensic summary plots from TelemetryLogger CSV output
    and overlays window isolator decisions on baseline telemetry.
    """

    OP_NAME = "forensic_summary"

    def __init__(self, cfg, sensor):
        self.sensor_id = sensor.sensor_id

        # How often summaries are rebuilt
        self.interval_sec = timeparse(cfg.get("interval", "30s"))

        # Time horizons to summarize
        self.horizons = cfg.get("horizons", ["5m", "1h", "24h"])

        # Match TelemetryLogger CSV naming logic
        base = cfg.get("csv_name", "telemetry")
        self.prefix = cfg.get("csv_prefix", "")
        filename = f"{self.prefix}_{base}.csv" if self.prefix else f"{base}.csv"

        # Resolve telemetry directory
        self.data_root = thoughtframe.resolve_rooted_path(
            THOUGHTFRAME_CONFIG,
            THOUGHTFRAME_CONFIG.get("samples", "audio"),
            self.sensor_id
        )

        self.telemetry_csv = os.path.join(self.data_root, filename)

        # Output directory for forensic artifacts
        self.outdir = os.path.join(self.data_root, "forensics")
        os.makedirs(self.outdir, exist_ok=True)

        self._last_run = 0

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    def process(self, chunk, analysis):
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

        for h in self.horizons:
            horizon_sec = timeparse(h)
            window = [
                r for r in rows
                if r["t_sec"] >= latest_t - horizon_sec
            ]

            if len(window) < 10:
                continue

            label = h.replace(" ", "")

            # Scalar summaries
            self._plot_rms(window, label)
            self._plot_iforest(window, label)
            self._plot_centroid(window, label)

            # Window forensic overlays
            self._plot_baseline_with_impulses(window, label)
            self._plot_baseline_with_iforest(window, label)

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
                        })
                    except (ValueError, TypeError):
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
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass
        return rows

    # ============================================================
    # Drawing primitives
    # ============================================================

    def _draw_impulses(self, ax, windows):
        for r in windows:
            ax.axvline(
                r["start_t"],
                color="black",
                linestyle=":",
                alpha=0.4
            )

    def _draw_windows(self, ax, windows, color):
        for r in windows:
            if r.get("state") != "EVENT":
                continue
            start = r["start_t"]
            end = r["end_t"] if r["end_t"] else start
            ax.axvspan(
                start,
                end,
                color=color,
                alpha=0.15
            )

    def _get_out_path(self, type_name, label):
        fname = (
            f"{self.prefix}_{type_name}_{label}.png"
            if self.prefix else
            f"{type_name}_{label}.png"
        )
        return os.path.join(self.outdir, fname)

    # ============================================================
    # Scalar plots (unchanged behavior)
    # ============================================================

    def _plot_rms(self, rows, label):
        t = [r["t_sec"] for r in rows]
        y = [r["rms_mean"] for r in rows]
        plt.figure(figsize=(10, 3))
        plt.plot(t, y, linewidth=1)
        plt.title(f"[{self.prefix}] RMS (mean) – {label}")
        plt.xlabel("Time (s)")
        plt.ylabel("RMS")
        plt.tight_layout()
        plt.savefig(self._get_out_path("rms", label))
        plt.close()

    def _plot_iforest(self, rows, label):
        t = [r["t_sec"] for r in rows]
        y = [r["iforest_score"] for r in rows]
        plt.figure(figsize=(10, 3))
        plt.plot(t, y, linewidth=1)
        plt.axhline(0, linestyle="--", alpha=0.4)
        plt.title(f"[{self.prefix}] Isolation Forest – {label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(self._get_out_path("iforest", label))
        plt.close()

    def _plot_centroid(self, rows, label):
        t = [r["t_sec"] for r in rows]
        y = [r["centroid_mean"] for r in rows]
        plt.figure(figsize=(10, 3))
        plt.plot(t, y, linewidth=1)
        plt.title(f"[{self.prefix}] Centroid (mean) – {label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Hz")
        plt.tight_layout()
        plt.savefig(self._get_out_path("centroid", label))
        plt.close()

    # ============================================================
    # Forensic window overlays (new)
    # ============================================================

    def _plot_baseline_with_impulses(self, rows, label):
        impulses = self._load_windows_csv("ImpulseIsolator")
        if not impulses:
            return

        t = [r["t_sec"] for r in rows]

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

        axes[0].plot(t, [r["rms"] for r in rows], linewidth=0.8)
        self._draw_impulses(axes[0], impulses)
        axes[0].set_ylabel("RMS")

        axes[1].plot(t, [r["spec_centroid_hz"] for r in rows], linewidth=0.8)
        self._draw_impulses(axes[1], impulses)
        axes[1].set_ylabel("Centroid (Hz)")

        axes[2].plot(t, [r["iforest_score"] for r in rows], linewidth=0.8)
        axes[2].axhline(0, linestyle="--", alpha=0.4)
        self._draw_impulses(axes[2], impulses)
        axes[2].set_ylabel("IF score")
        axes[2].set_xlabel("Time (s)")

        for ax in axes:
            ax.grid(alpha=0.3)

        plt.suptitle(f"[{self.prefix}] Baseline + Impulses — {label}")
        plt.tight_layout()
        plt.savefig(self._get_out_path("baseline_impulses", label))
        plt.close()

    def _plot_baseline_with_iforest(self, rows, label):
        windows = self._load_windows_csv("IsolationForestWindowIsolator")
        if not windows:
            return

        t = [r["t_sec"] for r in rows]

        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(t, [r["rms"] for r in rows], linewidth=0.8)
        self._draw_windows(axes[0], windows, "red")
        axes[0].set_ylabel("RMS")

        axes[1].plot(t, [r["spec_centroid_hz"] for r in rows], linewidth=0.8)
        self._draw_windows(axes[1], windows, "red")
        axes[1].set_ylabel("Centroid (Hz)")

        axes[2].plot(t, [r["iforest_score"] for r in rows], linewidth=0.8)
        axes[2].axhline(0, linestyle="--", alpha=0.4)
        self._draw_windows(axes[2], windows, "red")
        axes[2].set_ylabel("IF score")

        axes[3].plot(t, [r["anomaly_rate"] for r in rows], linewidth=0.8)
        self._draw_windows(axes[3], windows, "red")
        axes[3].set_ylabel("Anomaly rate")
        axes[3].set_xlabel("Time (s)")

        for ax in axes:
            ax.grid(alpha=0.3)

        plt.suptitle(f"[{self.prefix}] Baseline + IF Windows — {label}")
        plt.tight_layout()
        plt.savefig(self._get_out_path("baseline_iforest", label))
        plt.close()
