import csv
import os
import threading
import requests
import json
import numpy as np
import logging
from abc import ABC, abstractmethod
from pytimeparse.timeparse import timeparse

from tf_core.bootstrap import thoughtframe
from thoughtframe.sensor.interface import AcousticChunkProcessor, AcousticAnalysis
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG

log = logging.getLogger(__name__)

class WindowIsolator(AcousticChunkProcessor, ABC):
    """
    Base class for processors that segment the stream into temporal windows.
    Handles:
      1. CSV Persistence (Local)
      2. HTTP Event Notification (Java Backend)
    """

    def __init__(self, cfg, sensor):
        self.cfg = cfg
        self.sensor = sensor
        self.sensor_id = getattr(sensor, "sensor_id", "unknown")

        # --- CSV Setup ---
        base = cfg.get("csv_name") or self.__class__.__name__
        prefix = cfg.get("csv_prefix", "baseline")
        self.csv_name = f"{prefix}_{base}.csv" if prefix else f"{base}.csv"
        
        # --- Notification Setup ---
        # Matches DspModule.handleWindowEvent
        self.notify_url = cfg.get("notify_url", "http://localhost:8080/thoughtframe/lab/dsp/api/handlewindowevent") 
        self.notify_enabled = cfg.get("notify", True) # Default to True for now

        # --- Window Lifecycle ---
        self.state = "BASELINE"
        self.window_id = 0
        self.window_start_t = None
        self.window_stats = None

        self.min_duration_sec = timeparse(cfg.get("min_duration", "0s"))

        # --- Persistence Paths ---
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
        Analyze chunk and return proposed state (e.g. 'BASELINE', 'EVENT').
        """
        pass

    @abstractmethod
    def _init_window_stats(self) -> dict:
        """
        Return a dictionary of custom stats to track for a new window.
        """
        pass

    @abstractmethod
    def _update_window_stats(self, stats: dict, chunk, analysis) -> None:
        """
        Update the custom stats dictionary with the current chunk's data.
        """
        pass

    # ------------------------------------------------------------------
    # BASE WINDOW SCHEMA
    # ------------------------------------------------------------------
    def _base_window_stats(self) -> dict:
        return {
            "window_id": None,
            "state": None,
            "start_t": None,
            "end_t": None,
            "start_chunk": None,
            "end_chunk": None,
            "duration_sec": None,
            "duration_chunks": None,
            "num_chunks": 0,
            "rms_mean": np.nan,
            "rms_var": np.nan,
            "rms_min": np.nan,
            "rms_max": np.nan,
            "centroid_mean": np.nan,
            "centroid_var": np.nan,
        }

    # ------------------------------------------------------------------
    # CORE WINDOW ENGINE
    # ------------------------------------------------------------------
    def process(self, chunk, analysis):
        # Safety check for subclasses that forget to call super().__init__
        if not hasattr(self, "window_stats"):
            raise RuntimeError(f"{self.__class__.__name__} missing super().__init__ call")
            
        t_sec = analysis.node.t_sec
        proposed_state = self.propose_state(chunk, analysis)

        # 1. First Invocation
        if self.window_stats is None:
            self._start_window(t_sec, analysis, proposed_state)

        # 2. State Transition (The critical moment)
        elif proposed_state != self.state:
            duration = t_sec - self.window_start_t
            
            # Filter out very short blips if needed
            if duration >= self.min_duration_sec:
                self._close_window(t_sec, analysis)  # <--- Fires CLOSE event
                self._start_window(t_sec, analysis, proposed_state) # <--- Fires OPEN event
            # else: ignore short noise, maintain current state

        # 3. Update Stats
        self.window_stats["num_chunks"] += 1
        self._update_base_stats(self.window_stats, analysis)
        self._update_window_stats(self.window_stats, chunk, analysis)

    def _update_base_stats(self, stats: dict, analysis) -> None:
        """Welford's Online Algorithm for running variance on standard metrics"""
        m = analysis.metadata
        n = stats["num_chunks"]
    
        # RMS
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
    
        # Centroid
        centroid = m.get("spec_centroid_hz")
        if centroid is not None:
            if n == 1:
                stats["centroid_mean"] = centroid
                stats["centroid_var"] = 0.0
            else:
                delta_c = centroid - stats["centroid_mean"]
                stats["centroid_mean"] += delta_c / n
                stats["centroid_var"] += delta_c * (centroid - stats["centroid_mean"])

    # ------------------------------------------------------------------
    # LIFECYCLE & NOTIFICATION LOGIC
    # ------------------------------------------------------------------

    def _start_window(self, t_sec: float, analysis: AcousticAnalysis, state: str):
        self.window_id += 1
        self.state = state
        self.window_start_t = t_sec
        self.window_start_chunk = analysis.chunk_index
        
        stats = self._base_window_stats()
        stats.update(self._init_window_stats())

        # Create Unique Window ID (Sensor + Timestamp + Count)
        # Matches Java's expectation for tracking updates
        unique_id = f"{self.sensor_id}_{int(t_sec)}_{self.window_id}"

        stats["window_id"] = unique_id 
        stats["state"] = state
        stats["start_t"] = t_sec
        stats["start_chunk"] = analysis.chunk_index
        self.window_stats = stats
        
        # --- FIRE EVENT: OPEN ---
        if self.notify_enabled:
            self._notify_java(event="OPEN", stats=stats)

    def _close_window(self, t_sec: float, analysis: AcousticAnalysis):
        stats = self.window_stats
        stats["end_t"] = t_sec
        stats["end_chunk"] = analysis.chunk_index
        stats["duration_sec"] = t_sec - self.window_start_t
        stats["duration_chunks"] = (stats["end_chunk"] - stats["start_chunk"])
        
        self._write_window(stats)
        
        # --- FIRE EVENT: CLOSE ---
        if self.notify_enabled:
            self._notify_java(event="CLOSE", stats=stats)

    def _write_window(self, stats: dict):
        if self.windows_writer is None:
            fieldnames = list(stats.keys())
            self.windows_writer = csv.DictWriter(self.windows_file, fieldnames=fieldnames)
            if self.windows_file.tell() == 0:
                self.windows_writer.writeheader()
        
        self.windows_writer.writerow(stats)
        self.windows_file.flush()

    # ------------------------------------------------------------------
    # HTTP NOTIFICATION (THREADED)
    # ------------------------------------------------------------------
    def _notify_java(self, event, stats):
        """
        Sends JSON payload to DspModule.handleWindowEvent without blocking audio.
        """
        
        # LOGIC FIX: Ensure we send the correct timestamp for the event type
        # DspManager.java uses the 't' field to update either start_t or end_t
        if event == "CLOSE":
            timestamp = stats.get("end_t")
        else:
            timestamp = stats.get("start_t")

        payload = {
            "window_id": stats["window_id"],
            "event": event, # OPEN, CLOSE
            "t": timestamp,
            "duration": stats.get("duration_sec", 0),
            
            # Source key for Java to parse "beam_id" and "isolator"
            # e.g. "beam_0_impulse_isolator"
            "source": f"{self.sensor_id}_{self.cfg.get('name', 'unknown')}"
        }

        # Fire and forget thread
        t = threading.Thread(target=self._send_request, args=(payload,), daemon=True)
        t.start()

    def _send_request(self, payload):
        try:
            headers = {'Content-Type': 'application/json'}
            # Short timeout to avoid piling up threads if Java is down
            requests.post(self.notify_url, json=payload, headers=headers, timeout=1.0)
        except Exception as e:
            # Silent fail to keep console clean during transient network issues
            # log.debug(f"Notify Error: {e}")
            pass