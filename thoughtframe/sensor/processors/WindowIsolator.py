import csv
import os
import numpy as np
from abc import ABC, abstractmethod
from pytimeparse.timeparse import timeparse

from thoughtframe.bootstrap import thoughtframe
from thoughtframe.sensor.interface import AcousticChunkProcessor
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG


class WindowIsolator(AcousticChunkProcessor, ABC):
    """
    Base class for processors that segment the stream into temporal windows
    and persist per-window statistics.

    Subclasses must implement:
      - propose_state(...)
      - _init_window_stats()
      - _update_window_stats(...)
    """

    def __init__(self, cfg, sensor):
        self.cfg = cfg
        self.sensor = sensor
        self.csv_name = cfg.get("csv_name") or f"{self.__class__.__name__}.csv"
            
        # --- window lifecycle ---
        self.state = "BASELINE"
        self.window_id = 0
        self.window_start_t = None
        self.window_stats = None

        self.min_duration_sec = timeparse(cfg.get("min_duration", "0s"))

        # --- persistence ---
        path = thoughtframe.resolve_rooted_path(
            THOUGHTFRAME_CONFIG,
            THOUGHTFRAME_CONFIG.get("samples", "audio"),
            sensor.sensor_id
        )
        os.makedirs(path, exist_ok=True)

        self.windows_path = os.path.join(path, self.csv_name)
        self.windows_file = open(self.windows_path, "a", newline="")
        self.windows_writer = None

    # ------------------------------------------------------------------
    # REQUIRED SUBCLASS HOOKS
    # ------------------------------------------------------------------

    @abstractmethod
    def propose_state(self, chunk, analysis) -> str:
        """
        Return the desired next state ("BASELINE", "EVENT", etc.)
        based on the current chunk.
        """
        pass

    @abstractmethod
    def _init_window_stats(self) -> dict:
        """
        Return processor-specific window fields only.
        Base schema is injected automatically.
        """
        pass

    @abstractmethod
    def _update_window_stats(self, stats: dict, chunk, analysis) -> None:
        """
        Update stats in-place for the current chunk.
        Must NOT update base lifecycle fields.
        """
        pass

    # ------------------------------------------------------------------
    # BASE WINDOW SCHEMA (CANONICAL)
    # ------------------------------------------------------------------

    def _base_window_stats(self) -> dict:
        """
        Canonical schema shared by *all* window CSVs.
        Subclasses may extend but must not remove or rename fields.
        """
        return {
            # identity / lifecycle
            "window_id": None,
            "state": None,
            "start_t": None,
            "end_t": None,
            "duration_sec": None,
            "num_chunks": 0,

            # acoustic aggregates (optional but standardized)
            "rms_mean": np.nan,
            "rms_var": np.nan,
            "rms_min": np.nan,
            "rms_max": np.nan,

            "centroid_mean": np.nan,
            "centroid_var": np.nan,

        }

    # ------------------------------------------------------------------
    # CORE WINDOW ENGINE (DO NOT OVERRIDE)
    # ------------------------------------------------------------------

    def process(self, chunk, analysis):
        if not hasattr(self, "window_stats"):
            raise RuntimeError(
                f"{self.__class__.__name__}.__init__ did not call super().__init__"
            )
        t_sec = analysis.node.t_sec
        proposed_state = self.propose_state(chunk, analysis)

        # --- first invocation ---
        if self.window_stats is None:
            self._start_window(t_sec, proposed_state)

        # --- state transition ---
        elif proposed_state != self.state:
            duration = t_sec - self.window_start_t
            if duration >= self.min_duration_sec:
                self._close_window(t_sec)
                self._start_window(t_sec, proposed_state)
            # else: ignore short dwell

        # --- per-chunk accounting (owned by base) ---
        self.window_stats["num_chunks"] += 1

        self._update_base_stats(self.window_stats, analysis)

        # --- subclass-specific stats ---
        self._update_window_stats(self.window_stats, chunk, analysis)


    def _update_base_stats(self, stats: dict, analysis) -> None:
        m = analysis.metadata
        n = stats["num_chunks"]
    
        # --- RMS ---
        rms = m.get("rms")
        if rms is not None:
            if n == 1:
                stats["rms_mean"] = rms
                stats["rms_var"] = 0.0
                stats["rms_min"] = rms
                stats["rms_max"] = rms
            else:
                delta = rms - stats["rms_mean"]
                stats["rms_mean"] += delta / n
                stats["rms_var"] += delta * (rms - stats["rms_mean"])
                stats["rms_min"] = min(stats["rms_min"], rms)
                stats["rms_max"] = max(stats["rms_max"], rms)
    
        # --- Spectral centroid ---
        centroid = m.get("spec_centroid_hz")
        if centroid is not None:
            if n == 1:
                stats["centroid_mean"] = centroid
                stats["centroid_var"] = 0.0
            else:
                delta_c = centroid - stats["centroid_mean"]
                stats["centroid_mean"] += delta_c / n
                stats["centroid_var"] += delta_c * (
                    centroid - stats["centroid_mean"]
                )


    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _start_window(self, t_sec: float, state: str):
        self.window_id += 1
        self.state = state
        self.window_start_t = t_sec

        # base schema first
        stats = self._base_window_stats()

        # subclass extensions
        stats.update(self._init_window_stats())

        # lifecycle fields (owned here)
        stats["window_id"] = self.window_id
        stats["state"] = state
        stats["start_t"] = t_sec

        self.window_stats = stats

    def _close_window(self, t_sec: float):
        stats = self.window_stats
        stats["end_t"] = t_sec
        stats["duration_sec"] = t_sec - self.window_start_t
        self._write_window(stats)

    def _write_window(self, stats: dict):
        if self.windows_writer is None:
            fieldnames = list(stats.keys())
            self.windows_writer = csv.DictWriter(
                self.windows_file,
                fieldnames=fieldnames
            )
            if self.windows_file.tell() == 0:
                self.windows_writer.writeheader()

        self.windows_writer.writerow(stats)
        self.windows_file.flush()
